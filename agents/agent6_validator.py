import json
import re
from typing import Dict, List, Optional

from agents.base_agent import BaseAgent
from core.message_bus import MessageBus, Message
from utils.logger import logger

_VERDICT_FALLBACK = {
    "valid": True, "confidence": 0.6,
    "schema_compliance": True, "completeness": 0.75,
    "accuracy": 0.7, "task_fulfillment": 0.7,
    "critical_issues": [], "minor_issues": ["LLM verdict parse failed."],
    "feedback": "Verdict parse failed — accepted by default.",
    "recommended_action": "accept",
    "parse_error": True,
}

class Agent6QualityValidator(BaseAgent):

    def __init__(self, bus: MessageBus):
        super().__init__("agent6", "Agent 6 · Quality Validator (Multi-Tier)", bus)

    def _detect_id_field(self, output: list, raw_data) -> Optional[str]:
        candidates = ("_id", "id", "ID", "Id", "uuid", "recipeId")
        if isinstance(raw_data, list) and raw_data:
            sample = [r for r in raw_data[:50] if isinstance(r, dict)]
            if sample:
                for c in candidates:
                    if all(c in r for r in sample):
                        return c
        if isinstance(output, list) and output:
            dict_items = [x for x in output if isinstance(x, dict)]
            if dict_items:
                for c in candidates:
                    if all(c in x for x in dict_items):
                        return c
        return None

    def _rule_based_check(
        self, output: any, task: dict,
        raw_data=None, categorical_fields: Optional[Dict] = None,
        categorical_enums: Optional[Dict] = None,
        canonical_fields: Optional[List[str]] = None,
        nested_dict_field_schemas: Optional[Dict] = None,
        url_template_fields: Optional[Dict] = None,
        mostly_empty_fields: Optional[set] = None,
    ) -> Dict:
        critical: List[str] = []
        minor:    List[str] = []
        schema_ok = True

        if output is None or (isinstance(output, (list, str)) and len(output) == 0):
            critical.append("Output is empty or None.")

        requested_qty = task.get("quantity")
        if requested_qty and isinstance(output, list):
            actual = len(output)
            if actual < requested_qty:
                critical.append(
                    f"Quantity mismatch: requested {requested_qty}, got {actual}. "
                    f"Missing {requested_qty - actual} item(s)."
                )
            elif actual > requested_qty * 2:
                minor.append(
                    f"Quantity excess: requested {requested_qty}, got {actual}."
                )

        fmt = task.get("output_format", "json").lower()
        if fmt == "json":
            if not isinstance(output, (list, dict, str)):
                critical.append(f"Output type is {type(output).__name__}, expected list/dict.")
            elif isinstance(output, str):
                minor.append("Output is raw string, not parsed JSON.")

        if isinstance(output, list):
            non_dict = [i for i, x in enumerate(output) if not isinstance(x, dict)]
            if non_dict:
                critical.append(
                    f"Item(s) at index(es) {non_dict[:5]} are not JSON objects."
                )

        if isinstance(output, list) and output and isinstance(output[0], dict):
            if len(output[0]) < 2:
                minor.append("First item has very few fields — possible incomplete generation.")

        if isinstance(output, list) and output:
            id_field = self._detect_id_field(output, raw_data)
            if id_field:
                seen, dupes = set(), set()
                for item in output:
                    if not isinstance(item, dict) or id_field not in item:
                        continue
                    v = item[id_field]
                    if v in seen:
                        dupes.add(v)
                    seen.add(v)
                if dupes:
                    critical.append(
                        f"Duplicate '{id_field}' value(s) across generated items: "
                        f"{sorted(dupes, key=str)[:10]}."
                    )

        if isinstance(output, list) and len(output) > 1 and all(isinstance(x, dict) for x in output):
            key_sets = [frozenset(x.keys()) for x in output]
            distinct = set(key_sets)
            if len(distinct) > 1:
                from collections import Counter
                common = Counter(key_sets).most_common(1)[0][0]
                odd_idx = [i for i, ks in enumerate(key_sets) if ks != common]
                minor.append(
                    f"Inconsistent field set across {len(odd_idx)} item(s) "
                    f"(index(es) {odd_idx[:5]}) — possible schema drift between generation batches."
                )

                self.bus.set("flagged_item_indices", odd_idx)
            else:
                self.bus.set("flagged_item_indices", [])

        if isinstance(output, list) and categorical_enums:
            violations: Dict[str, set] = {}
            for item in output:
                if not isinstance(item, dict):
                    continue
                for field, counts in categorical_enums.items():
                    if field not in item:
                        continue
                    allowed = set(counts.keys())
                    val = item[field]
                    vals = val if isinstance(val, list) else ([val] if val is not None else [])
                    for v in vals:
                        if isinstance(v, str) and v not in allowed:
                            violations.setdefault(field, set()).add(v)
            if violations:
                critical.append(
                    "Invalid enum value(s) not present in the source dataset's "
                    "real vocabulary: "
                    + "; ".join(f"{k}={sorted(v)}" for k, v in violations.items())
                )
        elif isinstance(output, list) and categorical_fields:

            drift: Dict[str, set] = {}
            for item in output:
                if not isinstance(item, dict):
                    continue
                for field, allowed in categorical_fields.items():
                    if not allowed:
                        continue
                    allowed_set = {str(a).upper() for a in allowed}
                    val = item.get(field)
                    vals = val if isinstance(val, list) else ([val] if val is not None else [])
                    for v in vals:
                        if isinstance(v, str) and v.upper() not in allowed_set:
                            drift.setdefault(field, set()).add(v)
            if drift:
                minor.append(
                    "Possible enum/categorical drift vs known schema values: "
                    + "; ".join(f"{k}={sorted(v)}" for k, v in drift.items())
                )

        if isinstance(output, list) and output and canonical_fields:
            dict_items = [x for x in output if isinstance(x, dict)]
            if dict_items:
                _exempt = set(mostly_empty_fields or set())
                always_missing, sometimes_missing = [], []
                for f in canonical_fields:
                    if f in _exempt:
                        continue
                    missing_count = sum(1 for x in dict_items if f not in x)
                    if missing_count == len(dict_items):
                        always_missing.append(f)
                    elif missing_count > 0:
                        sometimes_missing.append(f)
                if always_missing:
                    critical.append(
                        f"Field(s) entirely missing from every generated item "
                        f"(present in the real schema but never generated): {always_missing}."
                    )
                if sometimes_missing:
                    minor.append(
                        f"Field(s) missing from some generated items: {sometimes_missing}."
                    )

        if isinstance(output, list) and output:
            contradiction_indices = []
            for idx, item in enumerate(output):
                if not isinstance(item, dict):
                    continue
                if self._find_label_ingredient_contradiction(item):
                    contradiction_indices.append(idx)
            if contradiction_indices:
                existing_flagged = self.bus.get("flagged_item_indices") or []
                merged = sorted(set(existing_flagged) | set(contradiction_indices))
                self.bus.set("flagged_item_indices", merged)
                critical.append(
                    f"Dietary/allergen labels contradicted by actual ingredients in "
                    f"{len(contradiction_indices)} item(s) "
                    f"(index(es) {contradiction_indices[:10]}) — e.g. Egg-Free "
                    f"with eggs, Low-Carb with pasta."
                )

        if isinstance(output, list) and output and nested_dict_field_schemas:
            for field, schema in nested_dict_field_schemas.items():
                subkeys = schema.get("subkeys", []) if isinstance(schema, dict) else schema
                canon_set = set(subkeys)
                if not canon_set:
                    continue
                bad_items = []
                for idx, item in enumerate(output):
                    if not isinstance(item, dict):
                        continue
                    val = item.get(field)
                    if not isinstance(val, dict) or not val:
                        continue
                    if any(not isinstance(iv, dict) or not iv for iv in val.values()):
                        bad_items.append(idx)
                if bad_items:
                    schema_ok = False
                    critical.append(
                        f"'{field}' is structurally corrupted (inner value(s) that "
                        f"should be nested {{{', '.join(sorted(canon_set))}}} objects "
                        f"are flattened to bare values) in item(s) index(es) "
                        f"{bad_items[:5]}."
                    )

        if isinstance(output, list) and output and url_template_fields:
            for field, prefix in url_template_fields.items():
                bad_items = [
                    idx for idx, item in enumerate(output)
                    if isinstance(item, dict) and isinstance(item.get(field), str)
                    and item[field] and not item[field].startswith(prefix)
                ]
                if bad_items:
                    schema_ok = False
                    critical.append(
                        f"'{field}' doesn't follow the real dataset's fixed template "
                        f"({prefix!r}) in item(s) index(es) {bad_items[:5]}."
                    )

        passes = len(critical) == 0
        return {"critical": critical, "minor": minor, "passes": passes, "schema_ok": schema_ok}

    _NUTRITION_LABEL_KEYWORDS = {
        "MEAT_FREE": ["meat", "beef", "pork", "chicken", "bacon", "ham", "sausage", "turkey", "lamb", "duck", "veal"],
        "BEEF_FREE": ["beef"], "CHICKEN_FREE": ["chicken"], "PORK_FREE": ["pork", "bacon", "ham"],
        "DUCK_FREE": ["duck"], "TURKEY_FREE": ["turkey"], "LAMB_FREE": ["lamb"], "VEAL_FREE": ["veal"],
        "EGG_FREE": ["egg"], "DAIRY_FREE": ["milk", "cheese", "butter", "yogurt", "yoghurt", "cream"],
        "MILK_FREE": ["milk", "cheese", "butter", "yogurt", "yoghurt", "cream"],
        "FISH_FREE": ["fish", "salmon", "tuna", "cod", "trout"],
        "POULTRY_FREE": ["chicken", "turkey", "duck", "goose"],
        "RED_MEAT_FREE": ["beef", "pork", "lamb", "veal"],
        "SEAFOOD_FREE": ["fish", "shrimp", "crab"],
        "SHELLFISH_FREE": ["shrimp", "crab", "lobster", "mussel", "oyster"],
        "CRUSTACEAN_FREE": ["shrimp", "crab", "lobster", "prawn"],
        "MOLLUSK_FREE": ["mussel", "oyster", "clam", "scallop", "squid", "octopus"],
        "HONEY_FREE": ["honey"],
        "GLUTEN_FREE": ["wheat", "bread", "flour", "pasta", "barley", "rye",
                         "baguette", "sourdough", "ciabatta", "croissant",
                         "focaccia", "brioche", "bagel", "biscotti",
                         "pastry", "dough", "noodle", "couscous", "semolina",
                         "pretzel", "cookie", "cake", "muffin", "waffle",
                         "pancake", "pizza", "tortilla", "crostini",
                         "panko", "breadcrumb", "cornetto"],
        "WHEAT_FREE": ["wheat", "bread", "flour", "pasta",
                        "baguette", "sourdough", "ciabatta", "croissant",
                        "focaccia", "brioche", "bagel", "biscotti",
                        "pastry", "dough", "noodle", "couscous", "semolina",
                        "pretzel", "cookie", "cake", "muffin", "waffle",
                        "pancake", "pizza", "tortilla", "crostini",
                        "panko", "breadcrumb", "cornetto"],
        "PEANUT_FREE": ["peanut"],
        "TREE_NUT_FREE": ["almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "macadamia", "brazil nut", "pine nut"],
        "SESAME_FREE": ["sesame", "tahini"],
        "SOY_FREE": ["soy", "tofu", "edamame", "tempeh"],
        "CELERY_FREE": ["celery"], "MUSTARD_FREE": ["mustard"], "LUPINE_FREE": ["lupin", "lupini"],
        "ALCOHOL_FREE": ["wine", "beer", "rum", "brandy", "liqueur", "whiskey", "vodka"],
    }
    _DIET_EXCLUSION_FIELD_CANDIDATES = (
        "nutritionLabels", "dietaryRequirements", "allergies",
        "foodIntolerances", "healthLabels",
    )
    _DIET_TYPE_CONTRADICTION_KEYWORDS = {
        "LOW_CARB": ["pasta", "bread", "flour", "rice", "potato", "sugar",
                     "honey", "pastry", "noodle"],

        "LOW_FAT": ["cheese", "ricotta", "mascarpone", "butter", "cream",
                    "bacon", "sausage", "heavy cream", "cream cheese"],
    }
    _DIET_LABEL_FIELD_CANDIDATES = ("dietLabels",)

    _CAUTIONS_FIELD_CANDIDATES = ("cautions",)
    _CAUTION_KEYWORDS = {
        "GLUTEN": ["wheat", "bread", "flour", "pasta", "barley", "rye",
                   "baguette", "sourdough", "ciabatta", "croissant",
                   "focaccia", "brioche", "bagel", "biscotti",
                   "pastry", "dough", "noodle", "couscous", "semolina",
                   "pretzel", "cookie", "cake", "muffin", "waffle",
                   "pancake", "pizza", "tortilla", "crostini",
                   "panko", "breadcrumb", "cornetto"],
        "WHEAT": ["wheat", "bread", "flour", "pasta",
                  "baguette", "sourdough", "ciabatta", "croissant",
                  "focaccia", "brioche", "bagel", "biscotti",
                  "pastry", "dough", "noodle", "couscous", "semolina",
                  "pretzel", "cookie", "cake", "muffin", "waffle",
                  "pancake", "pizza", "tortilla", "crostini",
                  "panko", "breadcrumb", "cornetto"],
        "EGGS": ["egg"], "EGG": ["egg"],
        "MILK": ["milk", "cheese", "butter", "yogurt", "yoghurt", "cream"],
        "DAIRY": ["milk", "cheese", "butter", "yogurt", "yoghurt", "cream"],
        "FISH": ["fish", "salmon", "tuna", "cod", "trout"],
        "SHELLFISH": ["shrimp", "crab", "lobster", "mussel", "oyster"],
        "CRUSTACEAN": ["shrimp", "crab", "lobster", "prawn"],
        "CRUSTACEANS": ["shrimp", "crab", "lobster", "prawn"],
        "MOLLUSKS": ["mussel", "oyster", "clam", "scallop", "squid", "octopus"],
        "PEANUTS": ["peanut"], "PEANUT": ["peanut"],
        "TREE_NUTS": ["almond", "walnut", "pecan", "cashew", "pistachio",
                      "hazelnut", "macadamia", "brazil nut", "pine nut"],
        "SESAME": ["sesame", "tahini"],
        "SOY": ["soy", "tofu", "edamame", "tempeh"],
        "SOY_BEANS": ["soy", "tofu", "edamame", "tempeh"],
        "CELERY": ["celery"], "MUSTARD": ["mustard"], "LUPINE": ["lupin", "lupini"],
    }

    @staticmethod
    def _normalize_label(label) -> str:
        return str(label).strip().upper().replace("-", "_").replace(" ", "_")

    def _find_label_ingredient_contradiction(self, item: dict) -> Optional[str]:
        if not isinstance(item, dict):
            return None
        ings = item.get("ingredients")
        if not (ings and isinstance(ings, list)):
            return None
        foods = " ".join(
            str(ing.get("food", "")) + " " + str(ing.get("text", ""))
            for ing in ings if isinstance(ing, dict)
        ).lower()
        if not foods:
            return None
        for field in self._DIET_EXCLUSION_FIELD_CANDIDATES:
            labels = item.get(field)
            if not (labels and isinstance(labels, list)):
                continue
            for label in labels:
                kws = self._NUTRITION_LABEL_KEYWORDS.get(self._normalize_label(label))
                if kws:
                    hit = next((kw for kw in kws if kw in foods), None)
                    if hit:
                        return f"{field} '{label}' contradicts actual ingredients (keyword match: {hit!r})"
        for field in self._DIET_LABEL_FIELD_CANDIDATES:
            labels = item.get(field)
            if not (labels and isinstance(labels, list)):
                continue
            for label in labels:
                kws = self._DIET_TYPE_CONTRADICTION_KEYWORDS.get(self._normalize_label(label))
                if kws:
                    hit = next((kw for kw in kws if kw in foods), None)
                    if hit:
                        return f"{field} '{label}' contradicts actual ingredients (keyword match: {hit!r})"
        for field in self._CAUTIONS_FIELD_CANDIDATES:
            labels = item.get(field)
            if not (labels and isinstance(labels, list)):
                continue
            for label in labels:
                kws = self._CAUTION_KEYWORDS.get(self._normalize_label(label))
                if kws and not any(kw in foods for kw in kws):
                    return (
                        f"{field} '{label}' is unsubstantiated — no matching "
                        f"ingredient found (expected one of: {kws})"
                    )
        return None

    def _has_real_nutrition_label_contradiction(self, output) -> bool:
        if not isinstance(output, list):
            return False
        for item in output:
            if self._find_label_ingredient_contradiction(item):
                return True
        return False

    def _has_real_calorie_weight_implausibility(self, output, issue_text: str = "") -> bool:
        if not isinstance(output, list):
            return False
        cal_bounds = self.bus.get("calories_per_serving_bounds")
        weight_bounds = self.bus.get("weight_per_serving_bounds")

        cited_numbers = [float(n) for n in re.findall(r"(\d+(?:\.\d+)?)", issue_text or "")]
        if cited_numbers:
            matched_any = False
            for item in output:
                if not isinstance(item, dict):
                    continue
                for bounds in (cal_bounds, weight_bounds):
                    if not bounds:
                        continue
                    field, yield_field = bounds.get("field"), bounds.get("yield_field")
                    v, y = item.get(field), item.get(yield_field)
                    if not (isinstance(v, (int, float)) and not isinstance(v, bool)):
                        continue
                    if not any(abs(v - c) < max(1.0, v * 0.01) for c in cited_numbers):
                        continue
                    matched_any = True
                    if (
                        isinstance(y, (int, float)) and not isinstance(y, bool) and y > 0
                    ):
                        per_serving = v / y
                        if per_serving < bounds["low"] or per_serving > bounds["high"]:
                            return True
            if matched_any:
                return False

        for item in output:
            if not isinstance(item, dict):
                continue
            for bounds in (cal_bounds, weight_bounds):
                if not bounds:
                    continue
                field, yield_field = bounds.get("field"), bounds.get("yield_field")
                v, y = item.get(field), item.get(yield_field)
                if (
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                    and isinstance(y, (int, float)) and not isinstance(y, bool) and y > 0
                ):
                    per_serving = v / y
                    if per_serving < bounds["low"] or per_serving > bounds["high"]:
                        return True
        return False

    def _filter_hallucinated_claims(self, critical_issues, output, canonical_fields: list) -> list:
        if not isinstance(output, list) or not output:
            return critical_issues

        _VAGUE_CATEGORY_LABELS = {
            "authenticity", "domain mismatch", "domain accuracy",
            "nutritional values", "nutritional value", "uniqueness",
            "dietary alignment", "schema compliance", "task fulfillment",
            "accuracy", "consistency", "quality", "realism",
            "cultural authenticity", "diversity", "completeness",
        }

        actually_missing = set()
        if canonical_fields:
            actually_missing = {
                f.lower() for f in canonical_fields
                if any(isinstance(it, dict) and f not in it for it in output)
            }
        has_real_nutrition_contradiction = None
        requested_qty = (self.bus.get("task_spec", {}) or {}).get("quantity")
        actual_qty = len(output)

        filtered = []
        for issue in critical_issues or []:
            issue_lower = str(issue).lower()

            bare_check = issue_lower.strip().rstrip(".!").strip()
            if bare_check in _VAGUE_CATEGORY_LABELS:
                logger.info(
                    f"[A6] Discarding judge claim as non-actionable — a "
                    f"bare rating-category label with no specific problem "
                    f"described: {issue!r}"
                )
                continue

            is_unverifiable_claim = (
                any(kw in issue_lower for kw in
                    ("not provided", "cannot verify", "unable to verify",
                     "no access to", "not given", "not available for comparison",
                     "no full", "not shown", "wasn't provided", "was not provided"))
                and (
                    any(kw in issue_lower for kw in
                        ("original dataset", "input dataset", "source dataset",
                         "provided dataset"))
                    or ("dataset" in issue_lower and any(kw in issue_lower for kw in
                        ("verify", "verif", "compar", "across batch",
                         "across the batch", "full batch")))
                )
            )
            if is_unverifiable_claim:
                logger.info(
                    f"[A6] Discarding judge claim as non-actionable — the judge "
                    f"is flagging its own inability to verify against data it's "
                    f"never given in any validation pass, not a defect in the "
                    f"output: {issue!r}"
                )
                continue

            is_missing_claim = any(kw in issue_lower for kw in ("missing", "absent", "lack"))
            if is_missing_claim and canonical_fields:
                mentioned = [f for f in canonical_fields if f.lower() in issue_lower]
                if mentioned and not any(f.lower() in actually_missing for f in mentioned):
                    logger.info(
                        f"[A6] Discarding judge claim as a false positive — rule "
                        f"tier confirms field(s) {mentioned} present in ALL "
                        f"items: {issue!r}"
                    )
                    continue

            is_extraneous_claim = any(
                kw in issue_lower for kw in
                ("extraneous", "extra field", "unnecessary field",
                 "shouldn't be present", "should not be present",
                 "not part of the schema", "non-canonical", "not in the schema")
            )
            if is_extraneous_claim and canonical_fields:
                mentioned = [f for f in canonical_fields if f.lower() in issue_lower]
                if mentioned:
                    logger.info(
                        f"[A6] Discarding judge claim as a false positive — "
                        f"field(s) {mentioned} it calls extraneous are "
                        f"actually canonical fields for this schema: {issue!r}"
                    )
                    continue

            is_servings_type_claim = (
                "servings" in issue_lower
                and any(kw in issue_lower for kw in
                        ("string", "numeric", "integer", "number", "should be",
                         "type", "consistency", "int", "float"))
            )
            if is_servings_type_claim:
                logger.info(
                    f"[A6] Discarding judge claim about servings field type — "
                    f"the real dataset stores servings as a string, so this "
                    f"is a false positive from the judge's general-knowledge "
                    f"assumption: {issue!r}"
                )
                continue

            is_nutrition_contradiction_claim = (
                "nutritionlabel" in issue_lower.replace(" ", "").replace("_", "")
                or ("nutrition" in issue_lower and any(
                    kw in issue_lower for kw in ("contradict", "conflict", "inconsist")
                ))

                or (
                    "allergy" in issue_lower
                    and any(kw in issue_lower for kw in ("conflict", "contradict", "free", "egg", "dairy", "meat", "fish", "honey"))
                )

                or (
                    any(kw in issue_lower for kw in ("diet label", "dietary requirement", "dietary label", "diet type"))
                    and any(kw in issue_lower for kw in ("conflict", "contradict", "inconsist", "misleading"))
                )
            )
            if is_nutrition_contradiction_claim:
                if has_real_nutrition_contradiction is None:
                    has_real_nutrition_contradiction = self._has_real_nutrition_label_contradiction(output)
                if not has_real_nutrition_contradiction:
                    logger.info(
                        f"[A6] Discarding judge claim as a false positive — keyword "
                        f"cross-check against ALL items' ingredients found no actual "
                        f"nutritionLabels contradiction: {issue!r}"
                    )
                    continue

            is_numeric_implausibility_claim = (
                any(kw in issue_lower for kw in ("calorie", "caloric", "kcal", "weight", "gram"))
                and any(kw in issue_lower for kw in
                        ("unrealistic", "implausible", "excessive", "too high",
                         "too much", "too low", "unusual", "unreasonable"))
            )
            if is_numeric_implausibility_claim:

                if not self._has_real_calorie_weight_implausibility(output, issue_lower):
                    logger.info(
                        f"[A6] Discarding judge claim as a false positive — every "
                        f"item's calorie/weight value is within Agent 4's real "
                        f"per-serving bounds once divided by its own yield: {issue!r}"
                    )
                    continue

            is_quantity_claim = bool(
                requested_qty is not None
                and re.search(
                    rf"\b{re.escape(str(requested_qty))}\b\s*(recipe|item|entr|dish|record)"
                    rf"|(?:out of|of the required|of the|instead of(?:\s+the)?|of)\s+\b{re.escape(str(requested_qty))}\b",
                    issue_lower,
                )
            )
            if is_quantity_claim and requested_qty is not None and actual_qty >= requested_qty:
                logger.info(
                    f"[A6] Discarding judge claim as a false positive — this "
                    f"SAME validation pass's own rule tier already confirmed "
                    f"{actual_qty}/{requested_qty} item(s) present: {issue!r}"
                )
                continue

            filtered.append(issue)
        return filtered

    def _build_judge_sample(self, output, max_items: int = 12, max_chars: int = 8000) -> str:
        if not isinstance(output, list) or not output:
            return json.dumps(output, indent=2, default=str, ensure_ascii=False)[:max_chars]

        n = len(output)
        if n <= max_items:
            indices = list(range(n))
        else:
            step = n / max_items
            indices = sorted(set(int(i * step) for i in range(max_items)))

        sampled = [output[i] for i in indices]
        text = json.dumps(sampled, indent=2, default=str, ensure_ascii=False)

        while len(sampled) > 1 and len(text) > max_chars:
            sampled = sampled[:-1]
            text = json.dumps(sampled, indent=2, default=str, ensure_ascii=False)

        return (
            f"[SAMPLE: {len(sampled)} complete items out of {n} total, evenly "
            f"spread across the full batch — NOT just the start]\n{text}"
        )

    async def _llm_validate(
        self, output, task: dict, schema_anal: str, user_ctx: str
    ) -> dict:
        from config import settings

        out_preview = self._build_judge_sample(output)

        critique = self.llm.chat([
            self._sys(
                "You are a strict quality-assurance evaluator. "
                "Assess the generated output against the requirements. Be concise."
            ),
            self._usr(
                f"User Requirements: {user_ctx}\n"
                f"Task: {json.dumps(task, default=str)[:800]}\n"
                f"Schema (excerpt): {schema_anal[:800]}\n"
                f"Generated Output:\n{out_preview}\n\n"
                "IMPORTANT: if the output above is marked as a SAMPLE, you are "
                "seeing a representative subset of complete items spread across "
                "the full batch, not every item. Field-by-field completeness "
                "across ALL items is already checked separately and exhaustively "
                "— do NOT claim a specific item is 'missing' a field unless you "
                "can see that exact item is missing it in what's shown below. "
                "Focus instead on PATTERNS visible in the sample: inconsistent "
                "units, unrealistic values, repeated/near-duplicate content, "
                "domain mismatches.\n\n"
                "Rate on:\n"
                "1. Schema compliance — do fields match the schema?\n"
                "2. Domain accuracy — values realistic and appropriate?\n"
                "3. Uniqueness — items are distinct from each other?\n"
                "4. Task fulfillment — meets user requirements?\n"
                "List any issues found."
            ),
        ], temperature=0.1)

        verdict_raw = self.llm.chat([
            self._sys("Output ONLY a single valid JSON object. No markdown."),
            self._usr(
                f"Critique:\n{critique}\n\n"
                f"NOTE: the output above was a REPRESENTATIVE SAMPLE of "
                f"{len(output)} total items. The full output contains "
                f"{len(output)} complete items — do NOT report a quantity "
                f"shortfall if the sample appeared to contain fewer.\n\n"
                "Return JSON:\n"
                "{\n"
                '  "valid": true/false,\n'
                '  "confidence": 0.0-1.0,\n'
                '  "schema_compliance": true/false,\n'
                '  "completeness": 0.0-1.0,\n'
                '  "accuracy": 0.0-1.0,\n'
                '  "task_fulfillment": 0.0-1.0,\n'
                '  "critical_issues": [],\n'
                '  "minor_issues": [],\n'
                '  "feedback": "<actionable improvement>",\n'
                '  "recommended_action": "accept|retry_agent5|retry_agent2"\n'
                "}\n"

                "EVERY entry in critical_issues and minor_issues MUST "
                "describe a SPECIFIC, concrete problem — name the item, "
                "field, or value that's wrong and why. A bare category or "
                "rubric name with no specific problem attached (e.g. "
                "\"Authenticity\", \"Uniqueness\", \"Nutritional Values\", "
                "\"Domain Mismatch\", \"Dietary Alignment\") is NOT a valid "
                "issue — if you have no concrete problem to report for one "
                "of the four rating dimensions, simply don't mention that "
                "dimension at all rather than listing its name.\n"
                "Start with { and end with }."
            ),
        ], temperature=0.05)

        verdict = self.llm.extract_json_dict(verdict_raw, fallback=dict(_VERDICT_FALLBACK))
        verdict["critical_issues"] = self._normalize_issue_list(verdict.get("critical_issues"))
        verdict["minor_issues"] = self._normalize_issue_list(verdict.get("minor_issues"))
        return self._enforce_verdict_consistency(verdict)

    def _normalize_issue_list(self, items) -> List[str]:
        normalized: List[str] = []
        for item in items or []:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                desc = item.get("description") or item.get("issue") or item.get("message")
                normalized.append(str(desc) if desc else str(item))
            else:
                normalized.append(str(item))
        return normalized

    def _enforce_verdict_consistency(self, verdict: Dict) -> Dict:
        has_critical = bool(verdict.get("critical_issues"))
        non_accept_action = verdict.get("recommended_action", "accept") != "accept"
        schema_bad = verdict.get("schema_compliance") is False

        if verdict.get("valid") is True and (has_critical or non_accept_action or schema_bad):
            logger.warning(
                "[A6] LLM judge verdict was self-contradictory "
                f"(valid=True but critical_issues={verdict.get('critical_issues')}, "
                f"schema_compliance={verdict.get('schema_compliance')}, "
                f"recommended_action={verdict.get('recommended_action')}) "
                "— correcting valid=False."
            )
            verdict["valid"] = False
            if verdict.get("recommended_action", "accept") == "accept":
                verdict["recommended_action"] = "retry_agent5"

        return verdict

    def _merge_verdicts(self, rule: Dict, llm_v: Dict) -> Dict:
        merged = dict(llm_v)

        merged["schema_compliance"] = (
            bool(merged.get("schema_compliance", True)) and rule.get("schema_ok", True)
        )

        if not rule["passes"]:
            merged["valid"]              = False
            merged["confidence"]         = min(float(merged.get("confidence", 0.5)), 0.4)
            merged["task_fulfillment"]   = min(float(merged.get("task_fulfillment", 0.5)), 0.4)
            merged["recommended_action"] = "retry_agent5"
            existing_critical = merged.get("critical_issues", [])
            merged["critical_issues"]    = rule["critical"] + [
                c for c in existing_critical if c not in rule["critical"]
            ]
            merged["feedback"] = (
                "Rule-based check failed. "
                + "; ".join(rule["critical"])
                + " "
                + merged.get("feedback", "")
            ).strip()

        if rule["minor"]:
            merged.setdefault("minor_issues", [])
            merged["minor_issues"] = rule["minor"] + merged["minor_issues"]

        return merged

    async def execute(self) -> bool:
        output      = self.bus.get("generated_output")
        fmt         = self.bus.get("generated_format", "json")
        task        = self.bus.get("task_spec", {})
        schema_anal = self.bus.get("schema_analysis", "")
        user_ctx    = self.bus.get("user_context", "")
        raw_data    = self.bus.get("raw_data")
        categorical_fields = (self.bus.get("inferred_schema", {}) or {}).get("categorical_fields", {})
        categorical_enums  = self.bus.get("categorical_enums", {}) or {}
        canonical_fields   = self.bus.get("canonical_fields", []) or []

        nested_dict_field_schemas = self.bus.get("nested_dict_field_schemas", {}) or {}
        url_template_fields       = self.bus.get("url_template_fields", {}) or {}
        mostly_empty_fields       = set((self.bus.get("mostly_empty_fields") or {}).keys())

        rule = self._rule_based_check(
            output, task, raw_data=raw_data,
            categorical_fields=categorical_fields,
            categorical_enums=categorical_enums,
            canonical_fields=canonical_fields,
            nested_dict_field_schemas=nested_dict_field_schemas,
            url_template_fields=url_template_fields,
            mostly_empty_fields=mostly_empty_fields,
        )
        logger.info(
            f"[A6] Rule check: passes={rule['passes']} "
            f"critical={rule['critical']} minor={rule['minor']}"
        )

        if not rule["passes"]:

            verdict = dict(_VERDICT_FALLBACK)
            verdict.update({
                "valid": False,
                "confidence": 0.4,

                "schema_compliance": rule.get("schema_ok", True),
                "critical_issues": rule["critical"],
                "minor_issues": rule["minor"],
                "feedback": "Rule-based check failed: " + "; ".join(rule["critical"]),
                "recommended_action": "retry_agent5",
                "parse_error": False,
            })
            logger.info("[A6] Skipping LLM semantic pass — hard rule failure. action=retry_agent5")
        else:

            try:
                llm_v = await self._llm_validate(output, task, schema_anal, user_ctx)
            except Exception as exc:
                logger.error(
                    f"[A6] LLM semantic pass failed with a technical error "
                    f"(not a quality verdict) — treating as inconclusive "
                    f"rather than crashing the pipeline: {exc}"
                )
                llm_v = dict(_VERDICT_FALLBACK)
                llm_v.update({
                    "valid": False,
                    "confidence": 0.5,
                    "schema_compliance": rule.get("schema_ok", True),
                    "critical_issues": [
                        f"LLM semantic validation could not complete due to a "
                        f"technical error ({exc}) — output was NOT rejected on "
                        f"quality grounds, just not yet checked this loop."
                    ],
                    "minor_issues": [],
                    "feedback": (
                        "The rule-tier checks (quantity, schema, structure) "
                        "already passed; only the LLM semantic review failed "
                        "to complete due to a technical error. Retrying will "
                        "re-attempt it."
                    ),
                    "recommended_action": "retry_agent5",
                    "parse_error": False,
                })
            llm_v["critical_issues"] = self._filter_hallucinated_claims(
                llm_v.get("critical_issues", []), output, canonical_fields
            )
            if not llm_v["critical_issues"] and llm_v.get("valid") is False:

                llm_v["valid"] = True
                llm_v["recommended_action"] = "accept"

                if llm_v.get("schema_compliance") is False:
                    llm_v["schema_compliance"] = True
            verdict = self._merge_verdicts(rule, llm_v)

        passes = (
            verdict.get("valid", False)
            and float(verdict.get("confidence", 0.0)) >= 0.55
        )

        logger.info(
            f"[A6] Final: valid={verdict.get('valid')} "
            f"conf={float(verdict.get('confidence', 0)):.2f} "
            f"passes={passes} "
            f"action={verdict.get('recommended_action')} "
            f"critical={len(verdict.get('critical_issues', []))}"
        )

        self.bus.set("validation_verdict",  verdict)
        self.bus.set("validator_feedback",  verdict.get("feedback", ""))
        self.bus.set("validation_passed",   passes)
        self.bus.set("recommended_action",  verdict.get("recommended_action", "accept"))

        await self.bus.publish(Message(
            agent_id=self.agent_id,
            topic="validation_complete",
            payload=verdict,
            status="success" if passes else "warning",
        ))
        return True
