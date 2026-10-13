import asyncio
import hashlib
import json
import re
from typing import Dict, Optional

from agents.base_agent import BaseAgent
from core.message_bus import MessageBus, Message
from utils.logger import logger

_VOLATILE_KEY_HINTS = ("id", "date", "created", "updated", "timestamp", "uuid", "author", "_class")

_IDENTITY_KEY_HINTS = ("title", "name")

def _normalize_label(label) -> str:
    return str(label).strip().upper().replace("-", "_").replace(" ", "_")

_DIET_EXCLUSION_FIELD_CANDIDATES = (
    "nutritionLabels", "dietaryRequirements", "allergies",
    "foodIntolerances", "healthLabels",
)

_NUTRITION_LABEL_KEYWORDS: dict = {
    "MEAT_FREE":    ["meat", "beef", "pork", "chicken", "bacon", "ham",
                     "sausage", "turkey", "lamb", "duck", "veal"],
    "BEEF_FREE":    ["beef"],
    "CHICKEN_FREE": ["chicken"],
    "PORK_FREE":    ["pork", "bacon", "ham"],
    "DUCK_FREE":    ["duck"],
    "TURKEY_FREE":  ["turkey"],
    "LAMB_FREE":    ["lamb"],
    "VEAL_FREE":    ["veal"],
    "EGG_FREE":     ["egg"],
    "DAIRY_FREE":   ["milk", "cheese", "butter", "yogurt", "yoghurt", "cream"],
    "MILK_FREE":    ["milk", "cheese", "butter", "yogurt", "yoghurt", "cream"],
    "FISH_FREE":    ["fish", "salmon", "tuna", "cod", "trout"],
    "POULTRY_FREE": ["chicken", "turkey", "duck", "goose"],
    "RED_MEAT_FREE":["beef", "pork", "lamb", "veal"],
    "SEAFOOD_FREE": ["fish", "shrimp", "crab"],
    "SHELLFISH_FREE":["shrimp", "crab", "lobster", "mussel", "oyster"],
    "CRUSTACEAN_FREE":["shrimp", "crab", "lobster", "prawn"],
    "MOLLUSK_FREE": ["mussel", "oyster", "clam", "scallop", "squid", "octopus"],
    "HONEY_FREE":   ["honey"],
    "GLUTEN_FREE":  ["wheat", "bread", "flour", "pasta", "barley", "rye",
                      "baguette", "sourdough", "ciabatta", "croissant",
                      "focaccia", "brioche", "bagel", "biscotti",
                      "pastry", "dough", "noodle", "couscous", "semolina",
                      "pretzel", "cookie", "cake", "muffin", "waffle",
                      "pancake", "pizza", "tortilla", "crostini",
                      "panko", "breadcrumb", "cornetto"],
    "WHEAT_FREE":   ["wheat", "bread", "flour", "pasta",
                      "baguette", "sourdough", "ciabatta", "croissant",
                      "focaccia", "brioche", "bagel", "biscotti",
                      "pastry", "dough", "noodle", "couscous", "semolina",
                      "pretzel", "cookie", "cake", "muffin", "waffle",
                      "pancake", "pizza", "tortilla", "crostini",
                      "panko", "breadcrumb", "cornetto"],
    "PEANUT_FREE":  ["peanut"],
    "TREE_NUT_FREE":["almond", "walnut", "pecan", "cashew", "pistachio",
                      "hazelnut", "macadamia", "brazil nut", "pine nut"],
    "SESAME_FREE":  ["sesame", "tahini"],
    "SOY_FREE":     ["soy", "tofu", "edamame", "tempeh"],
    "CELERY_FREE":  ["celery"],
    "MUSTARD_FREE": ["mustard"],
    "LUPINE_FREE":  ["lupin", "lupini"],
    "ALCOHOL_FREE": ["wine", "beer", "rum", "brandy", "liqueur", "whiskey", "vodka"],
}

_DIET_LABEL_FIELD_CANDIDATES = ("dietLabels",)
_DIET_TYPE_CONTRADICTION_KEYWORDS: dict = {
    "LOW_CARB": ["pasta", "bread", "flour", "rice", "potato", "sugar",
                 "honey", "pastry", "noodle"],
    "LOW_FAT": ["cheese", "ricotta", "mascarpone", "butter", "cream",
                "bacon", "sausage", "heavy cream", "cream cheese"],
}

_CAUTIONS_FIELD_CANDIDATES = ("cautions",)
_CAUTION_KEYWORDS: dict = {
    "GLUTEN":     _NUTRITION_LABEL_KEYWORDS["GLUTEN_FREE"],
    "WHEAT":      _NUTRITION_LABEL_KEYWORDS["WHEAT_FREE"],
    "EGGS":       _NUTRITION_LABEL_KEYWORDS["EGG_FREE"],
    "EGG":        _NUTRITION_LABEL_KEYWORDS["EGG_FREE"],
    "MILK":       _NUTRITION_LABEL_KEYWORDS["DAIRY_FREE"],
    "DAIRY":      _NUTRITION_LABEL_KEYWORDS["DAIRY_FREE"],
    "FISH":       _NUTRITION_LABEL_KEYWORDS["FISH_FREE"],
    "SHELLFISH":  _NUTRITION_LABEL_KEYWORDS["SHELLFISH_FREE"],
    "CRUSTACEAN": _NUTRITION_LABEL_KEYWORDS["CRUSTACEAN_FREE"],
    "CRUSTACEANS":_NUTRITION_LABEL_KEYWORDS["CRUSTACEAN_FREE"],
    "MOLLUSKS":   _NUTRITION_LABEL_KEYWORDS["MOLLUSK_FREE"],
    "PEANUTS":    _NUTRITION_LABEL_KEYWORDS["PEANUT_FREE"],
    "PEANUT":     _NUTRITION_LABEL_KEYWORDS["PEANUT_FREE"],
    "TREE_NUTS":  _NUTRITION_LABEL_KEYWORDS["TREE_NUT_FREE"],
    "SESAME":     _NUTRITION_LABEL_KEYWORDS["SESAME_FREE"],
    "SOY":        _NUTRITION_LABEL_KEYWORDS["SOY_FREE"],
    "SOY_BEANS":  _NUTRITION_LABEL_KEYWORDS["SOY_FREE"],
    "CELERY":     _NUTRITION_LABEL_KEYWORDS["CELERY_FREE"],
    "MUSTARD":    _NUTRITION_LABEL_KEYWORDS["MUSTARD_FREE"],
    "LUPINE":     _NUTRITION_LABEL_KEYWORDS["LUPINE_FREE"],
}

def _item_ingredient_text(item: dict) -> str:
    ings = item.get("ingredients")
    if not (ings and isinstance(ings, list)):
        return ""
    return " ".join(
        str(ing.get("food", "")) + " " + str(ing.get("text", ""))
        for ing in ings if isinstance(ing, dict)
    ).lower()

def _find_label_ingredient_contradiction(item: dict) -> Optional[str]:
    foods = _item_ingredient_text(item)
    if not foods:
        return None
    for field in _DIET_EXCLUSION_FIELD_CANDIDATES:
        labels = item.get(field)
        if not (labels and isinstance(labels, list)):
            continue
        for label in labels:
            kws = _NUTRITION_LABEL_KEYWORDS.get(_normalize_label(label))
            if kws:
                hit = next((kw for kw in kws if kw in foods), None)
                if hit:
                    return f"{field} '{label}' contradicts actual ingredients (keyword match: {hit!r})"
    for field in _DIET_LABEL_FIELD_CANDIDATES:
        labels = item.get(field)
        if not (labels and isinstance(labels, list)):
            continue
        for label in labels:
            kws = _DIET_TYPE_CONTRADICTION_KEYWORDS.get(_normalize_label(label))
            if kws:
                hit = next((kw for kw in kws if kw in foods), None)
                if hit:
                    return f"{field} '{label}' contradicts actual ingredients (keyword match: {hit!r})"
    for field in _CAUTIONS_FIELD_CANDIDATES:
        labels = item.get(field)
        if not (labels and isinstance(labels, list)):
            continue
        for label in labels:
            kws = _CAUTION_KEYWORDS.get(_normalize_label(label))
            if kws and not any(kw in foods for kw in kws):
                return (
                    f"{field} '{label}' is unsubstantiated — no matching "
                    f"ingredient found (expected one of: {kws})"
                )
    return None

_LARGE_UNIT_MULTIPLIERS = {
    "kilogram": 1000, "kilograms": 1000, "kg": 1000,
    "litre": 1000, "litres": 1000, "liter": 1000, "liters": 1000, "l": 1000,
    "tablespoon": 15, "tablespoons": 15, "tbsp": 15,
    "teaspoon": 5, "teaspoons": 5, "tsp": 5,
    "cup": 240, "cups": 240,
}

_COUNT_MEASURE_HINTS = {
    "unit", "units", "piece", "pieces", "pc", "pcs", "each", "whole",
    "clove", "cloves", "slice", "slices", "loaf", "loaves", "can", "cans",
    "bag", "bags", "bunch", "bunches", "stick", "sticks", "bar", "bars",
    "jar", "jars", "bottle", "bottles", "package", "packages", "pack", "packs",
}

_INGREDIENT_UNIT_WEIGHTS_G = {
    "egg": 50,
    "banana": 120,
    "apple": 180,
    "orange": 130,
    "garlic clove": 3,
    "clove": 3,
    "onion": 110,
    "tomato": 120,
    "cucumber": 200,
    "bread": 30,
    "toast": 30,
    "sausage": 50,
    "bacon": 15,
    "ham": 25,
    "cheese": 20,
    "pepper": 120,
}

_ALLERGY_KEYWORDS = {
    "GLUTEN_FREE": ["wheat", "flour", "bread", "pasta", "barley", "rye", "phyllo", "filo"],
    "EGG_FREE": ["egg"],
    "FISH_FREE": ["fish", "salmon", "tuna", "cod", "trout"],
    "CRUSTACEAN_FREE": ["shrimp", "crab", "lobster", "crayfish"],
    "SHELLFISH_FREE": ["shrimp", "crab", "lobster", "mussel", "oyster", "clam", "scallop"],
    "MOLLUSK_FREE": ["mussel", "oyster", "clam", "scallop", "squid", "octopus"],
    "PEANUT_FREE": ["peanut"],
    "SOY_FREE": ["soy", "tofu", "edamame"],
    "NUTS_TREE_NUTS_FREE": ["almond", "walnut", "hazelnut", "pistachio", "cashew", "pecan"],
    "SESAME_FREE": ["sesame", "tahini"],
    "MUSHROOM_FREE": ["mushroom"],
    "TOMATO_FREE": ["tomato"],
}
_FOOD_INTOLERANCE_KEYWORDS = {
    "LACTOSE_FREE": ["milk", "cheese", "butter", "yogurt", "yoghurt", "cream"],
}

class Agent5OutputGenerator(BaseAgent):
    def __init__(self, bus: MessageBus):
        super().__init__("agent5", "Agent 5 · Output Generator", bus)

    async def execute(self) -> bool:
        from config import settings

        task         = self.bus.get("task_spec", {})
        features     = self.bus.get("extracted_features", {})
        rag_ctx      = self.bus.get("rag_context", "")
        schema_anal  = self.bus.get("schema_analysis", "")
        user_ctx     = self.bus.get("user_context", "")
        fmt          = self.bus.get("output_format", "json").lower()
        retry_count  = self.bus.get("retry_count", 0)
        feedback     = self.bus.get("validator_feedback", "")
        examples     = self.bus.get("data_examples", [])

        retry_note = ""
        if retry_count > 0 and feedback:
            retry_note = (
                f"\n\nRETRY #{retry_count} — Fix ALL issues from previous attempt:\n"
                f"{feedback}\nEvery issue MUST be resolved."
            )

        if fmt in ("json",):
            result = await self._gen_json_batched(
                task, features, rag_ctx, schema_anal,
                user_ctx, retry_note, examples,
            )
        elif fmt == "csv":
            result = await self._gen_csv(task, features, rag_ctx, user_ctx, retry_note)
        elif fmt in ("text", "txt", "markdown", "md"):
            result = await self._gen_text(task, features, rag_ctx, user_ctx, retry_note)
        elif fmt == "image":
            result = await self._gen_image(task, features, rag_ctx, user_ctx)
        else:
            result = await self._gen_text(task, features, rag_ctx, user_ctx, retry_note)

        self.bus.set("generated_output", result)
        self.bus.set("generated_format", fmt)

        await self.bus.publish(Message(
            agent_id=self.agent_id,
            topic="output_generated",
            payload={
                "format": fmt,
                "count":  len(result) if isinstance(result, list) else 1,
                "size":   len(str(result)),
            },
        ))
        return True

    async def _gen_json_batched(
        self, task, features, rag_ctx, schema_anal,
        user_ctx, retry_note, examples,
    ) -> list:
        from config import settings

        qty = task.get("quantity") or 1

        canonical_fields  = self.bus.get("canonical_fields", []) or []
        categorical_enums = self.bus.get("categorical_enums", {}) or {}
        mostly_empty_fields = self.bus.get("mostly_empty_fields", {}) or {}
        url_template_fields = self.bus.get("url_template_fields", {}) or {}
        numeric_bounds      = self.bus.get("numeric_bounds", {}) or {}
        ingredient_conventions = self.bus.get("ingredient_conventions", {}) or {}
        title_field = next(
            (f for f in ("title", "label", "name", "recipeName", "recipeTitle")
             if f in canonical_fields),
            None,
        )
        ingredient_weight_lookup = self.bus.get("ingredient_weight_lookup", {}) or {}
        ingredient_energy_field  = self.bus.get("ingredient_energy_field")
        ingredient_energy_lookup = self.bus.get("ingredient_energy_lookup", {}) or {}
        nutrient_label_lookup    = self.bus.get("nutrient_label_lookup", {}) or {}
        ingredient_canonical_fields = self.bus.get("ingredient_canonical_fields", []) or []
        ingredient_field_types = self.bus.get("ingredient_field_types", {}) or {}
        total_weight_field = self.bus.get("total_weight_field")
        nested_dict_field_schemas = self.bus.get("nested_dict_field_schemas", {}) or {}
        calories_per_serving_bounds = self.bus.get("calories_per_serving_bounds")
        weight_per_serving_bounds = self.bus.get("weight_per_serving_bounds")
        raw_data_for_id = self.bus.get("raw_data")
        id_field = self._detect_id_field(raw_data_for_id)

        all_items: list = list(self.bus.get("carried_items", []) or [])
        if all_items:
            logger.info(f"[A5] Resuming with {len(all_items)} item(s) carried over from a previous attempt.")
            all_items = self._reassign_ids(all_items, self.bus.get("raw_data"))
            all_items = self._normalize_field_sets(all_items, self._canonical_field_set(self.bus.get("raw_data")))
            all_items = self._normalize_enums(all_items, categorical_enums, task)
            all_items = self._normalize_field_shapes(all_items, self.bus.get("raw_data"))
            all_items = self._normalize_system_fields(
                all_items, mostly_empty_fields, url_template_fields, numeric_bounds,
                ingredient_conventions
            )
            all_items, resume_hollow_recovered = self._recover_hollow_nested_dict_entries(
                all_items, nested_dict_field_schemas
            )
            if resume_hollow_recovered:
                logger.info(
                    f"[A5] Recovered {resume_hollow_recovered} completely-empty "
                    f"nested sub-value(s) in carried-over items using real-data-"
                    f"grounded per-nutrient defaults."
                )
            all_items, resume_weight_fixed = self._reconcile_ingredient_weights_with_lookup(
                all_items, ingredient_weight_lookup
            )
            if resume_weight_fixed:
                logger.info(
                    f"[A5] Backfilled/corrected {resume_weight_fixed} per-ingredient "
                    f"weight(s) in carried-over items against the real per-food lookup."
                )
            all_items, resume_wq_fixed = self._reconcile_weight_to_quantity(
                all_items,
                only_when_mirrors=ingredient_conventions.get("weight_mirrors_quantity"),
            )
            if resume_wq_fixed:
                logger.info(
                    f"[A5] Synced {resume_wq_fixed} per-ingredient weight(s) "
                    f"to match their stated quantity in carried-over items."
                )
            all_items, resume_energy_fixed = self._reconcile_ingredient_energy_with_lookup(
                all_items, ingredient_energy_field, ingredient_energy_lookup
            )
            if resume_energy_fixed:
                logger.info(
                    f"[A5] Corrected {resume_energy_fixed} per-ingredient "
                    f"'{ingredient_energy_field}' value(s) in carried-over items "
                    f"against the real per-food kcal/gram lookup."
                )
            all_items, resume_nutrient_labels_fixed = self._reconcile_nutrient_labels_with_lookup(
                all_items, nutrient_label_lookup
            )
            if resume_nutrient_labels_fixed:
                logger.info(
                    f"[A5] Corrected {resume_nutrient_labels_fixed} nutrient "
                    f"label code(s) in carried-over items against the real "
                    f"name->label lookup."
                )
            all_items, resume_labels_removed = self._remove_contradicted_dietary_labels(
                all_items
            )
            if resume_labels_removed:
                logger.info(
                    f"[A5] Removed {resume_labels_removed} dietary/allergen label(s) "
                    f"contradicted by their own ingredients in carried-over items."
                )
            all_items, dropped_values, dropped_reasons = self._drop_invalid_items(
                all_items, canonical_fields, categorical_enums,
                ingredient_canonical_fields, ingredient_field_types, mostly_empty_fields,
                nested_dict_field_schemas, total_weight_field,
            )
            if dropped_reasons:
                logger.info(
                    f"[A5] Dropped {len(dropped_reasons)} invalid carried-over "
                    f"item(s) — will regenerate. Reasons: {'; '.join(dropped_reasons[:5])}"
                    + (f" (+{len(dropped_reasons) - 5} more)" if len(dropped_reasons) > 5 else "")
                )

            if retry_note and len(all_items) >= qty:
                force_n = min(len(all_items), max(1, settings.GENERATION_CONCURRENCY * settings.GENERATION_BATCH_SIZE))
                flagged = self.bus.get("flagged_item_indices", []) or []
                flagged = sorted(set(i for i in flagged if 0 <= i < len(all_items)))
                if flagged:
                    to_drop = set(flagged[:force_n]) if len(flagged) > force_n else set(flagged)
                    logger.info(
                        f"[A5] Retry has semantic feedback — targeting "
                        f"{len(to_drop)} specifically-flagged item(s) "
                        f"(index(es) {sorted(to_drop)}) for regeneration."
                    )
                    all_items = [item for i, item in enumerate(all_items) if i not in to_drop]
                else:
                    ranked = self._rank_items_by_implausibility(
                        all_items, calories_per_serving_bounds, weight_per_serving_bounds,
                        url_template_fields,
                    )
                    to_drop = {idx for idx, score in ranked[:force_n] if score > 0}
                    if to_drop:
                        logger.info(
                            f"[A5] Retry has semantic feedback ('unrealistic'/"
                            f"'inconsistent' values) but no flagged_item_indices "
                            f"— targeting the {len(to_drop)} item(s) with the "
                            f"worst per-serving calorie/weight plausibility "
                            f"score instead of an arbitrary trim."
                        )
                        all_items = [item for i, item in enumerate(all_items) if i not in to_drop]
                    else:
                        safe_n = min(force_n, max(1, len(all_items) // 3))
                        logger.info(
                            f"[A5] Retry has semantic feedback but no structurally-invalid "
                            f"items were found to replace — forcing regeneration of "
                            f"{safe_n} item(s) (capped from {force_n} to protect "
                            f"a near-complete set) so the feedback can be applied."
                        )
                        all_items = all_items[:-safe_n] if safe_n < len(all_items) else []

        signatures = {self._signature(i) for i in all_items if isinstance(i, dict)}
        seen_titles = set()
        if title_field:
            seen_titles = {
                str(i[title_field]).strip().lower()
                for i in all_items
                if isinstance(i, dict) and i.get(title_field)
            }
        rejected_values: dict = {}
        rejected_reasons: list = []

        batch_size   = settings.GENERATION_BATCH_SIZE
        min_batch    = max(1, getattr(settings, "MIN_GEN_BATCH_SIZE", 1))
        concurrency  = max(1, getattr(settings, "GENERATION_CONCURRENCY", 3))
        backend      = self._pick_backend(settings)

        required_fields_note = ""
        if canonical_fields:
            required_fields_note = (
                "\n\nREQUIRED TOP-LEVEL FIELDS — every generated item MUST include "
                "ALL of the following fields (with realistic, appropriate values; "
                "do not omit any of them):\n"
                + ", ".join(canonical_fields)
            )
        if id_field:
            required_fields_note += (
                f"\n\nABOUT THE \"{id_field}\" FIELD: do NOT invent a complex or "
                f"\"realistic-looking\" unique identifier for it (e.g. never a "
                f"MongoDB ObjectId-style hex string). A simple placeholder "
                f"integer (or omitting the field entirely) is fine — the "
                f"pipeline automatically assigns the real, guaranteed-unique "
                f"value afterward. Whatever you put there just needs to be a "
                f"plain JSON number, not a quoted or unquoted string."
            )
        if ingredient_canonical_fields:
            required_fields_note += (
                "\n\nREQUIRED INGREDIENT SUB-FIELDS — every object inside every "
                "item's \"ingredients\" array MUST include ALL of these keys "
                "with real, non-empty values, using the SAME value type as a "
                "real dataset would (e.g. a unit like \"cup\" is text, not a "
                "bare number; never an empty {} placeholder, and never a "
                "different key name than these):\n"
                + ", ".join(ingredient_canonical_fields)
            )
        if nested_dict_field_schemas:
            lines = []
            for field, schema in nested_dict_field_schemas.items():
                subkeys = schema.get("subkeys", [])
                constants = schema.get("constants", {})
                line = (
                    f'  "{field}": every key must map to an object with fields '
                    + ", ".join(subkeys) + " (never a bare number, never an empty {})"
                )
                if constants:
                    const_desc = ", ".join(f'"{k}" is ALWAYS {v!r}' for k, v in constants.items())
                    line += f" — {const_desc}"
                lines.append(line)
            required_fields_note += (
                "\n\nREQUIRED NESTED OBJECT STRUCTURE:\n" + "\n".join(lines)
            )
            required_fields_note += (
                "\n\nKEEP THESE FIELDS CONCISE: for the nested-object "
                "fields above, include roughly 10-15 of the most "
                "important entries (e.g. energy, total fat, saturated "
                "fat, carbohydrates, fiber, sugar, protein, cholesterol, "
                "sodium, calcium, iron, vitamin C) rather than trying to "
                "list every nutrient the real dataset happens to have — "
                "real records can have 25-35+ entries, but you do NOT "
                "need to match that count; a shorter, still-realistic "
                "set is correct and expected."
            )
            required_fields_note += (
                "\n\nDO NOT SPLIT ONE RECIPE ACROSS MULTIPLE ARRAY "
                "ELEMENTS: totalNutrients, totalDaily, cuisine, "
                "mealType, dishType, and every other field belong "
                "INSIDE the SAME single { } object as uri/label — "
                "never close a recipe's { } early and continue its "
                "remaining fields as a new, separate { } in the array. "
                "Each array element must be one COMPLETE recipe with "
                "every field together; the array should contain "
                "exactly one object per recipe, never more."
            )

        enum_note = ""
        if categorical_enums:
            enum_lines = []
            for field, counts in categorical_enums.items():
                allowed = ", ".join(list(counts.keys())[:15])
                enum_lines.append(f"  - {field}: MUST be chosen ONLY from [{allowed}]")
            enum_note = (
                "\n\nSTRICT ENUM CONSTRAINTS — these fields may ONLY use the exact "
                "values listed (matching case exactly; do not invent new forms, "
                "tenses, or synonyms):\n" + "\n".join(enum_lines)
            )

        schema_ctx = schema_anal[:settings.MAX_LLM_CONTEXT_CHARS]
        feat_ctx   = json.dumps(
            {k: v for k, v in features.items() if not k.startswith("_")},
            indent=2, default=str
        )[:settings.MAX_LLM_CONTEXT_CHARS]
        rag_short  = rag_ctx[:settings.MAX_LLM_CONTEXT_CHARS]

        style_examples = examples[:1]
        style_text = json.dumps(
            [self._abbreviate(e) for e in style_examples],
            indent=2, default=str, ensure_ascii=False
        )[:2200]

        round_num = 0
        consecutive_empty_rounds = 0
        tried_simplified_fallback = False

        while len(all_items) < qty and round_num < settings.MAX_GEN_ATTEMPTS:
            if self.bus.get("llm_backend_exhausted"):
                logger.warning(
                    "[A5] Stopping rounds immediately — a non-retryable backend "
                    "error (billing/quota) was already detected this run, so "
                    "further rounds would only repeat the same guaranteed failure."
                )
                break
            round_num += 1
            remaining = qty - len(all_items)

            n_batches = min(concurrency, max(1, -(-remaining // batch_size)))
            readable_existing = self._readable_descriptions(all_items)[-30:]
            avoid_note = self._build_avoid_note(rejected_values, rejected_reasons)

            use_simplified = consecutive_empty_rounds >= 2 and not tried_simplified_fallback
            if use_simplified:
                tried_simplified_fallback = True
                logger.warning(
                    "[A5] 2 consecutive empty rounds — trying one simplified "
                    "(reduced-context) attempt before giving up."
                )
            round_feat_ctx = "" if use_simplified else feat_ctx
            round_rag_short = "" if use_simplified else rag_short

            logger.info(
                f"[A5] Round {round_num}: firing {n_batches} concurrent batch(es) "
                f"of ≤{batch_size} item(s) (have {len(all_items)}/{qty}, backend={backend})…"
            )

            batch_tasks = [
                self._gen_single_batch(
                    n=min(batch_size, remaining + 2),
                    batch_num=f"{round_num}.{i + 1}",
                    schema_ctx=schema_ctx, feat_ctx=round_feat_ctx, rag_short=round_rag_short,
                    style_text=style_text, user_ctx=user_ctx, retry_note=retry_note,
                    existing_desc=readable_existing, task=task, backend=backend,
                    temp_jitter=i,
                    required_fields_note=required_fields_note,
                    enum_note=enum_note + avoid_note,
                )
                for i in range(n_batches)
            ]
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            round_new, round_dupes = 0, 0
            new_batch_items = []
            for res in results:
                if isinstance(res, Exception):
                    logger.warning(f"[A5] Batch error: {res}")
                    continue
                for item in res:
                    if not isinstance(item, dict):
                        continue
                    sig = self._signature(item)
                    if sig in signatures:
                        round_dupes += 1
                        continue
                    if title_field and item.get(title_field):
                        norm_title = str(item[title_field]).strip().lower()
                        if norm_title in seen_titles:
                            round_dupes += 1
                            continue
                        seen_titles.add(norm_title)
                    signatures.add(sig)
                    new_batch_items.append(item)

            raw_data = self.bus.get("raw_data")
            new_batch_items = self._reassign_ids(new_batch_items, raw_data)
            new_batch_items = self._normalize_field_sets(new_batch_items, self._canonical_field_set(raw_data))
            new_batch_items = self._normalize_enums(new_batch_items, categorical_enums, task)
            new_batch_items = self._normalize_field_shapes(new_batch_items, raw_data)
            new_batch_items = self._normalize_system_fields(
                new_batch_items, mostly_empty_fields, url_template_fields, numeric_bounds,
                ingredient_conventions
            )
            new_batch_items, hollow_recovered = self._recover_hollow_nested_dict_entries(
                new_batch_items, nested_dict_field_schemas
            )
            if hollow_recovered:
                logger.info(
                    f"[A5] Recovered {hollow_recovered} completely-empty nested "
                    f"sub-value(s) using real-data-grounded per-nutrient defaults, "
                    f"before they could cost a whole item its acceptance."
                )
            new_batch_items, early_weight_fixed = self._reconcile_ingredient_weights_with_lookup(
                new_batch_items, ingredient_weight_lookup
            )
            if early_weight_fixed:
                logger.info(
                    f"[A5] Backfilled/corrected {early_weight_fixed} per-ingredient "
                    f"weight(s) against the real per-food lookup, before they could "
                    f"cost a whole item its acceptance."
                )
            new_batch_items, early_wq_fixed = self._reconcile_weight_to_quantity(
                new_batch_items,
                only_when_mirrors=ingredient_conventions.get("weight_mirrors_quantity"),
            )
            if early_wq_fixed:
                logger.info(
                    f"[A5] Synced {early_wq_fixed} per-ingredient weight(s) to match "
                    f"their stated quantity, before they could cost a whole item its "
                    f"acceptance."
                )
            new_batch_items, early_energy_fixed = self._reconcile_ingredient_energy_with_lookup(
                new_batch_items, ingredient_energy_field, ingredient_energy_lookup
            )
            if early_energy_fixed:
                logger.info(
                    f"[A5] Corrected {early_energy_fixed} per-ingredient "
                    f"'{ingredient_energy_field}' value(s) against the real per-food "
                    f"kcal/gram lookup, before they could cost a whole item its "
                    f"acceptance."
                )
            new_batch_items, early_nutrient_labels_fixed = self._reconcile_nutrient_labels_with_lookup(
                new_batch_items, nutrient_label_lookup
            )
            if early_nutrient_labels_fixed:
                logger.info(
                    f"[A5] Corrected {early_nutrient_labels_fixed} nutrient label "
                    f"code(s) against the real name->label lookup, before they "
                    f"could cost a whole item its acceptance."
                )
            new_batch_items, early_labels_removed = self._remove_contradicted_dietary_labels(
                new_batch_items
            )
            if early_labels_removed:
                logger.info(
                    f"[A5] Removed {early_labels_removed} dietary/allergen label(s) "
                    f"contradicted by their own ingredients, before they could cost "
                    f"a whole item its acceptance."
                )
            new_batch_items, round_dropped, round_dropped_reasons = self._drop_invalid_items(
                new_batch_items, canonical_fields, categorical_enums,
                ingredient_canonical_fields, ingredient_field_types, mostly_empty_fields,
                nested_dict_field_schemas, total_weight_field,
            )
            for field, val in round_dropped:
                rejected_values.setdefault(field, set()).add(val)
            rejected_reasons.extend(round_dropped_reasons)

            for item in new_batch_items:
                if len(all_items) >= qty:
                    break
                all_items.append(item)
                round_new += 1

            logger.info(
                f"[A5] Round {round_num}: +{round_new} new, {round_dupes} duplicate(s), "
                f"{len(round_dropped_reasons)} invalid item(s) rejected (have {len(all_items)}/{qty})."
                + (
                    f" Reasons: {'; '.join(round_dropped_reasons[:3])}"
                    + (f" (+{len(round_dropped_reasons) - 3} more)" if len(round_dropped_reasons) > 3 else "")
                    if round_dropped_reasons else ""
                )
            )

            self.bus.set("carried_items", list(all_items))

            if round_new == 0:
                consecutive_empty_rounds += 1
                if batch_size > min_batch:
                    batch_size = max(min_batch, batch_size // 2)
                    logger.warning(f"[A5] No new items this round — shrinking batch size to {batch_size}.")
                if consecutive_empty_rounds >= 3:
                    logger.warning("[A5] 3 consecutive empty rounds — stopping early.")
                    break
            else:
                consecutive_empty_rounds = 0

        logger.info(f"[A5] Generated {len(all_items)}/{qty} items in {round_num} round(s).")
        all_items = all_items[:qty] if len(all_items) > qty else all_items

        raw_data = self.bus.get("raw_data")
        all_items = self._reassign_ids(all_items, raw_data)
        all_items = self._normalize_field_sets(all_items, self._canonical_field_set(raw_data))
        all_items = self._normalize_enums(all_items, categorical_enums, task)
        all_items = self._normalize_field_shapes(all_items, raw_data)
        all_items = self._normalize_system_fields(
            all_items, mostly_empty_fields, url_template_fields, numeric_bounds,
            ingredient_conventions
        )
        all_items, fake_precision_fixed = self._clean_fake_precision(all_items)
        if fake_precision_fixed:
            logger.info(
                f"[A5] Rounded {fake_precision_fixed} value(s) with obviously-fake "
                f"sequential-digit decimal precision (e.g. \"...1234567890123\") "
                f"down to 2 decimal places. Done before weight reconciliation below "
                f"so that step works from cleaned inputs."
            )
        all_items, weight_fixed = self._reconcile_weight_field(all_items, total_weight_field)
        if weight_fixed:
            logger.info(
                f"[A5] Reconciled '{total_weight_field}' against summed "
                f"ingredient weights on {weight_fixed} item(s)."
            )
        all_items, const_fixed = self._apply_nested_dict_constants(
            all_items, nested_dict_field_schemas
        )
        if const_fixed:
            logger.info(
                f"[A5] Force-corrected {const_fixed} nested sub-value(s) "
                f"to their known-constant real-data value (e.g. totalDaily unit=%)."
            )
        all_items, lookup_fixed = self._reconcile_ingredient_weights_with_lookup(
            all_items, ingredient_weight_lookup
        )
        if lookup_fixed:
            logger.info(
                f"[A5] Corrected {lookup_fixed} per-ingredient weight(s) that were "
                f"wildly implausible against the real per-food weight lookup."
            )
        all_items, wq_fixed = self._reconcile_weight_to_quantity(
            all_items,
            only_when_mirrors=ingredient_conventions.get("weight_mirrors_quantity"),
        )
        if wq_fixed:
            logger.info(
                f"[A5] Synced {wq_fixed} per-ingredient weight(s) to match "
                f"their stated quantity (real dataset convention: weight=quantity)."
            )
        all_items, energy_lookup_fixed = self._reconcile_ingredient_energy_with_lookup(
            all_items, ingredient_energy_field, ingredient_energy_lookup
        )
        if energy_lookup_fixed:
            logger.info(
                f"[A5] Corrected {energy_lookup_fixed} per-ingredient "
                f"'{ingredient_energy_field}' value(s) that were wildly implausible "
                f"against the real per-food kcal/gram lookup."
            )
        all_items, nutrient_labels_fixed = self._reconcile_nutrient_labels_with_lookup(
            all_items, nutrient_label_lookup
        )
        if nutrient_labels_fixed:
            logger.info(
                f"[A5] Corrected {nutrient_labels_fixed} nutrient label code(s) "
                f"against the real name->label lookup."
            )
        all_items, per_serving_fixed = self._clamp_per_serving(
            all_items, calories_per_serving_bounds
        )
        if per_serving_fixed:
            logger.info(
                f"[A5] Clamped {per_serving_fixed} item(s) whose calories were "
                f"implausible once normalized per serving (not just in absolute terms)."
            )
        all_items, weight_per_serving_fixed = self._clamp_per_serving(
            all_items, weight_per_serving_bounds
        )
        if weight_per_serving_fixed:
            logger.info(
                f"[A5] Clamped {weight_per_serving_fixed} item(s) whose "
                f"totalWeight was implausible once normalized per serving, "
                f"even though it matched its own ingredients' sum."
            )
        all_items, calories_reconciled = self._reconcile_calories_with_nutrients(all_items)
        if calories_reconciled:
            logger.info(
                f"[A5] Reconciled totalNutrients' energy value against the "
                f"top-level 'calories' field on {calories_reconciled} item(s) "
                f"that had drifted out of sync."
            )
        all_items, daily_energy_fixed = self._reconcile_energy_daily_value(all_items)
        if daily_energy_fixed:
            logger.info(
                f"[A5] Recomputed totalDaily's energy percentage directly "
                f"from 'calories' (2000-kcal reference, confirmed against "
                f"real data) on {daily_energy_fixed} item(s)."
            )

        all_items, filled = self._fill_missing_categorical_defaults(all_items, categorical_enums)
        if filled:
            logger.info(f"[A5] Backfilled missing categorical field(s) with dataset-majority values: {filled}")

        if canonical_fields:
            missing_counts: dict = {}
            for item in all_items:
                if not isinstance(item, dict):
                    continue
                for f in canonical_fields:
                    if f not in item:
                        missing_counts[f] = missing_counts.get(f, 0) + 1
            if missing_counts:
                logger.warning(
                    f"[A5] Field completeness gap vs canonical schema "
                    f"(field: #items missing it): {missing_counts}"
                )
            self.bus.set("field_completeness_gap", missing_counts)

        return all_items

    async def _gen_single_batch(
        self, n, batch_num, schema_ctx, feat_ctx,
        rag_short, style_text, user_ctx, retry_note,
        existing_desc, task, backend, temp_jitter=0,
        required_fields_note="", enum_note="",
    ) -> list:
        from config import settings

        constraints = "\n".join(f"  - {c}" for c in task.get("constraints", []))
        existing_note = (
            "\nALREADY GENERATED — do not repeat these; vary ingredients, "
            "cooking method, and dish type from every one of them:\n"
            + "\n".join(f"  - {d}" for d in existing_desc)
            if existing_desc else ""
        )

        messages = [
            self._sys(
                "You are a precise structured data generator. "
                "Output ONLY a valid JSON array containing exactly the number of items requested. "
                "Start with [ and end with ]. "
                "Each item must fully conform to the schema with ALL required fields present. "
                "Values must be realistic, domain-appropriate, and diverse. "
                "Do NOT include markdown fences or explanation."
            ),
            self._usr(
                f"User Instructions: {user_ctx}\n"
                f"Task: {task.get('summary', '')}\n"
                f"Constraints:\n{constraints}\n"
                f"{retry_note}\n"
                f"{required_fields_note}\n"
                f"{enum_note}\n"
                f"{existing_note}\n\n"
                f"=== SCHEMA ANALYSIS ===\n{schema_ctx}\n\n"
                f"=== DOMAIN FEATURES & PATTERNS ===\n{feat_ctx}\n\n"
                f"=== STYLE GUIDE (2 abbreviated examples showing ALL real fields "
                f"— do NOT copy values, but DO match this field structure) ===\n{style_text}\n\n"
                f"=== REFERENCE CONTEXT ===\n{rag_short}\n\n"
                f"Generate EXACTLY {n} unique, complete, schema-compliant item(s) "
                f"as a JSON array. Every item must include every field listed under "
                f"REQUIRED TOP-LEVEL FIELDS above, and every enum field must use only "
                f"values from STRICT ENUM CONSTRAINTS above. "
                f"This is batch {batch_num}. "
                f"Output ONLY the JSON array starting with [ and ending with ]."
            ),
        ]

        temperature = min(0.95, 0.7 + 0.05 * temp_jitter)

        if backend == "huggingface":
            floor = 8192
            batch_tokens_ceiling = getattr(settings, "AGENT5_HF_BATCH_TOKENS_CEILING", 16384)
        else:
            floor = 4096
            batch_tokens_ceiling = getattr(settings, "AGENT5_BATCH_TOKENS_CEILING", 8192)
        dynamic_max_tokens = min(batch_tokens_ceiling, max(floor, n * 2400 + 1200))
        logger.info(
            f"[A5] Batch {batch_num}: requesting {n} item(s), "
            f"max_tokens capped at {dynamic_max_tokens} (ceiling {batch_tokens_ceiling})."
        )

        try:
            raw = await self.llm.achat(
                messages,
                temperature=temperature,
                max_tokens=dynamic_max_tokens,
                backend=backend,
            )
        except Exception as exc:
            logger.warning(f"[A5] Batch {batch_num} LLM call failed: {exc}")
            exc_str = str(exc)
            exc_lower = exc_str.lower()
            if (
                ("max_tokens" in exc_lower or "max tokens" in exc_lower
                 or "context_length" in exc_lower or "context length" in exc_lower)
                and ("exceed" in exc_lower or "too large" in exc_lower or "too long" in exc_lower
                     or "400" in exc_str)
            ):
                logger.error(
                    f"[A5] Batch {batch_num}: provider rejected the requested "
                    f"max_tokens ({dynamic_max_tokens}) outright — this looks "
                    f"like a per-request output-token limit on the current "
                    f"backend/provider, not a transient error. Consider "
                    f"lowering AGENT5_BATCH_TOKENS_CEILING. Error: {exc_str}"
                )
            if "402" in exc_str and "payment" in exc_str.lower():
                self.bus.set("llm_backend_exhausted", True)
                self.bus.set("llm_backend_exhausted_reason", exc_str)
                logger.error(
                    "[A5] Backend billing/quota error detected (HTTP 402) — "
                    "this will not be fixed by retrying. Stopping generation "
                    "for this run rather than continuing to burn retries."
                )
            elif (
                "429" in exc_str and (
                    "tokens_per_day" in exc_str.lower()
                    or "daily" in exc_str.lower()
                    or "exceeded" in exc_str.lower() and "day" in exc_str.lower()
                )
            ):
                self.bus.set("llm_backend_exhausted", True)
                self.bus.set("llm_backend_exhausted_reason", exc_str)
                logger.error(
                    "[A5] Groq DAILY token quota exhausted (HTTP 429 with "
                    "tokens_per_day / daily limit signal) — this will not "
                    "recover until the quota resets. Stopping generation for "
                    "this run to avoid burning more retries on a permanent "
                    "failure. Consider switching to llama-3.1-8b-instant "
                    "(14.4K req/day, 500K tokens/day) for iteration-heavy "
                    "runs, or wait for the daily reset."
                )
            return []

        try:
            data = self.llm.extract_json(raw)
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                logger.warning(f"[A5] Batch {batch_num}: unexpected type {type(data)}")
                return []
            data, orphans_merged = self._merge_orphan_item_fragments(data)
            if orphans_merged:
                logger.warning(
                    f"[A5] Batch {batch_num}: merged {orphans_merged} orphaned "
                    f"field fragment(s) back into the item they were split "
                    f"off from (model closed a recipe's object early and "
                    f"continued its remaining fields as separate array "
                    f"entries)."
                )
            return data
        except (ValueError, Exception) as exc:
            logger.warning(f"[A5] Batch {batch_num} parse failed: {exc}")
            if self._looks_truncated(raw, dynamic_max_tokens) and dynamic_max_tokens < batch_tokens_ceiling:
                retry_tokens = min(batch_tokens_ceiling, dynamic_max_tokens * 2)
                logger.warning(
                    f"[A5] Batch {batch_num}: response looks CUT OFF at the "
                    f"{dynamic_max_tokens}-token cap (ends mid-value, not just "
                    f"malformed) — retrying once with a larger budget "
                    f"({retry_tokens} tokens) before giving up on this batch."
                )
                try:
                    raw2 = await self.llm.achat(
                        messages, temperature=temperature,
                        max_tokens=retry_tokens, backend=backend,
                    )
                    data2 = self.llm.extract_json(raw2)
                    if isinstance(data2, dict):
                        data2 = [data2]
                    if isinstance(data2, list):
                        data2, _ = self._merge_orphan_item_fragments(data2)
                        return data2
                except Exception as exc2:
                    logger.warning(
                        f"[A5] Batch {batch_num} retry-with-larger-budget also failed: {exc2}"
                    )
            return []

    @staticmethod
    def _merge_orphan_item_fragments(items: list) -> tuple:
        if not isinstance(items, list) or len(items) < 2:
            return items, 0
        identity_fields = ("uri", "label", "title", "name", "recipeName", "recipeTitle")
        merged = []
        fixed = 0
        for item in items:
            if not isinstance(item, dict):
                merged.append(item)
                continue
            has_identity = any(item.get(f) for f in identity_fields)
            if (
                not has_identity and merged and isinstance(merged[-1], dict)
                and not any(k in merged[-1] for k in item.keys())
            ):
                merged[-1].update(item)
                fixed += 1
                continue
            merged.append(item)
        return merged, fixed

    @staticmethod
    def _looks_truncated(raw: str, max_tokens: int) -> bool:
        stripped = (raw or "").rstrip()
        if not stripped or stripped.endswith(("]", "}")):
            return False
        approx_tokens = len(stripped) / 3.5
        if approx_tokens >= max_tokens * 0.85:
            return True
        lead = stripped.lstrip()[:1]
        return lead in ("[", "{") and '"' in stripped and ":" in stripped

    def _signature(self, item: dict) -> str:
        def _clean(obj):
            if isinstance(obj, dict):
                return {
                    k: _clean(v) for k, v in sorted(obj.items())
                    if not any(h in k.lower() for h in _VOLATILE_KEY_HINTS)
                }
            if isinstance(obj, list):
                return [_clean(x) for x in obj]
            return obj

        cleaned = _clean(item)
        blob = json.dumps(cleaned, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _detect_id_field(self, raw_data) -> Optional[str]:
        if not isinstance(raw_data, list) or not raw_data:
            return None
        sample = [r for r in raw_data[:50] if isinstance(r, dict)]
        if not sample:
            return None
        for candidate in ("_id", "id", "ID", "Id", "uuid", "recipeId"):
            present = [r for r in sample if candidate in r]
            if len(present) < len(sample) * 0.9:
                continue
            numeric = [r[candidate] for r in present if isinstance(r[candidate], (int, float))]
            if len(numeric) >= len(present) * 0.8:
                return candidate
        return None

    def _reassign_ids(self, items: list, raw_data) -> list:
        id_field = self._detect_id_field(raw_data)
        if not id_field:
            return items

        existing_ids = set()
        if isinstance(raw_data, list):
            for r in raw_data:
                if isinstance(r, dict) and isinstance(r.get(id_field), (int, float)):
                    existing_ids.add(int(r[id_field]))

        used = set(existing_ids)
        next_id = (max(used) + 1) if used else 1
        reassigned = 0

        for item in items:
            if not isinstance(item, dict):
                continue
            cur = item.get(id_field)
            if isinstance(cur, (int, float)) and int(cur) not in used:
                used.add(int(cur))
                continue
            while next_id in used:
                next_id += 1
            item[id_field] = next_id
            used.add(next_id)
            next_id += 1
            reassigned += 1

        if reassigned:
            logger.info(
                f"[A5] Reassigned {reassigned} colliding/missing '{id_field}' "
                f"value(s) to guarantee uniqueness against the source dataset."
            )
        return items

    def _canonical_field_set(self, raw_data, min_presence: float = 0.9):
        if not isinstance(raw_data, list) or not raw_data:
            return None
        sample = [r for r in raw_data[:300] if isinstance(r, dict)]
        if not sample:
            return None
        counts: dict = {}
        for r in sample:
            for k in r.keys():
                counts[k] = counts.get(k, 0) + 1
        n = len(sample)
        return {k for k, c in counts.items() if c / n >= min_presence}

    def _normalize_field_sets(self, items: list, canonical_fields) -> list:
        if not canonical_fields:
            return items
        canonical_set = set(canonical_fields)
        dropped: dict = {}
        renamed: dict = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            for k in [k for k in item.keys() if k not in canonical_set]:
                candidates = {"_" + k, k.lstrip("_")}
                match = next(
                    (c for c in candidates if c in canonical_set and c not in item),
                    None,
                )
                if match:
                    item[match] = item.pop(k)
                    key = f"{k}->{match}"
                    renamed[key] = renamed.get(key, 0) + 1
                else:
                    dropped[k] = dropped.get(k, 0) + 1
                    del item[k]
        if renamed:
            logger.info(f"[A5] Renamed near-miss field name(s) to canonical form: {renamed}")
        if dropped:
            logger.warning(
                f"[A5] Normalized schema drift — dropped non-canonical field(s) "
                f"not present in the source schema: {dropped}"
            )
        return items

    _ENUM_SUFFIX_REWRITES = [
        ("IED", "YING"),
        ("ED", "ING"),
    ]

    def _closest_enum_match(self, value: str, allowed: list) -> Optional[str]:
        if not isinstance(value, str) or not allowed:
            return None
        v_upper = value.upper().strip()
        allowed_upper = {a.upper(): a for a in allowed}
        if v_upper in allowed_upper:
            return allowed_upper[v_upper]

        for suffix, replacement in self._ENUM_SUFFIX_REWRITES:
            if v_upper.endswith(suffix):
                stem = v_upper[: -len(suffix)]
                candidate = stem + replacement
                if candidate in allowed_upper:
                    return allowed_upper[candidate]

        for a_upper, a_original in allowed_upper.items():
            if a_upper in v_upper or v_upper in a_upper:
                return a_original

        return None

    def _infer_required_meal_type(self, task: dict, categorical_enums: dict) -> Optional[str]:
        if "mealType" not in categorical_enums:
            return None
        allowed = {v.upper(): v for v in categorical_enums["mealType"]}
        search_text = " ".join([
            str(task.get("summary", "")),
            " ".join(str(c) for c in task.get("constraints", [])),
            str(task.get("action", "")),
        ]).upper()

        priority_order = [
            "AFTERNOON_SNACK", "MORNING_SNACK", "AFTER_DINNER_SNACK",
            "BRUNCH",
            "BREAKFAST", "LUNCH", "DINNER", "SNACK", "TEATIME",
        ]
        for keyword in priority_order:
            if keyword in allowed and keyword in search_text:
                return allowed[keyword]
        return None

    def _normalize_enums(self, items: list, categorical_enums: dict,
                         task: Optional[dict] = None) -> list:
        if not categorical_enums:
            return items

        required_meal_type = self._infer_required_meal_type(task or {}, categorical_enums)
        corrections: dict = {}
        dropped: dict = {}

        for item in items:
            if not isinstance(item, dict):
                continue

            if required_meal_type and "mealType" in item:
                val = item["mealType"]
                current = [v for v in val] if isinstance(val, list) else ([val] if val else [])
                if current and not any(str(v).upper() == required_meal_type.upper() for v in current):
                    item["mealType"] = [required_meal_type]
                    corrections[f"mealType:task_constraint->{required_meal_type}"] = (
                        corrections.get(f"mealType:task_constraint->{required_meal_type}", 0) + 1
                    )

            for field, counts in categorical_enums.items():
                if field not in item:
                    continue
                allowed = list(counts.keys())
                val = item[field]

                if isinstance(val, list):
                    new_list = []
                    for v in val:
                        if not isinstance(v, str):
                            new_list.append(v)
                            continue
                        if v in allowed:
                            new_list.append(v)
                            continue
                        match = self._closest_enum_match(v, allowed)
                        if match:
                            corrections[f"{field}:{v}->{match}"] = (
                                corrections.get(f"{field}:{v}->{match}", 0) + 1
                            )
                            new_list.append(match)
                        else:
                            dropped[f"{field}:{v}"] = dropped.get(f"{field}:{v}", 0) + 1
                    item[field] = new_list
                elif isinstance(val, str) and val not in allowed:
                    match = self._closest_enum_match(val, allowed)
                    if match:
                        corrections[f"{field}:{val}->{match}"] = (
                            corrections.get(f"{field}:{val}->{match}", 0) + 1
                        )
                        item[field] = match

        if corrections:
            logger.info(f"[A5] Auto-corrected enum drift: {corrections}")
        if dropped:
            logger.info(
                f"[A5] Dropped {sum(dropped.values())} unresolvable enum value(s) "
                f"from list fields (no match in real vocabulary): {dropped}"
            )
        return items

    def _detect_createdat_shape(self, raw_data) -> Optional[str]:
        if not isinstance(raw_data, list):
            return None
        for r in raw_data[:50]:
            if not isinstance(r, dict):
                continue
            ca = r.get("createdAt")
            if isinstance(ca, dict) and len(ca) == 1:
                return next(iter(ca.keys()))
        return None

    def _detect_ingredient_value_type(self, raw_data, field: str) -> Optional[str]:
        if not isinstance(raw_data, list):
            return None
        counts = {"str": 0, "number": 0}
        for r in raw_data[:100]:
            if not isinstance(r, dict):
                continue
            for ing in r.get("ingredients", []) or []:
                if not isinstance(ing, dict) or field not in ing:
                    continue
                v = ing[field]
                if isinstance(v, str):
                    counts["str"] += 1
                elif isinstance(v, (int, float)):
                    counts["number"] += 1
        if counts["str"] == 0 and counts["number"] == 0:
            return None
        return "str" if counts["str"] >= counts["number"] else "number"

    def _normalize_field_shapes(self, items: list, raw_data) -> list:
        date_key = self._detect_createdat_shape(raw_data)
        qty_type = self._detect_ingredient_value_type(raw_data, "quantity")
        weight_type = self._detect_ingredient_value_type(raw_data, "weight")

        fixed_dates, fixed_qty, fixed_weight = 0, 0, 0

        for item in items:
            if not isinstance(item, dict):
                continue

            if date_key and "createdAt" in item:
                ca = item["createdAt"]
                if isinstance(ca, dict):
                    if date_key not in ca:
                        if len(ca) == 1:
                            value = next(iter(ca.values()))
                            item["createdAt"] = {date_key: value}
                            fixed_dates += 1
                elif isinstance(ca, str):
                    item["createdAt"] = {date_key: ca}
                    fixed_dates += 1

            if (qty_type or weight_type) and isinstance(item.get("ingredients"), list):
                for ing in item["ingredients"]:
                    if not isinstance(ing, dict):
                        continue
                    if qty_type and "quantity" in ing and ing["quantity"] is not None:
                        v = ing["quantity"]
                        if qty_type == "str" and not isinstance(v, str):
                            ing["quantity"] = str(v)
                            fixed_qty += 1
                        elif qty_type == "number" and isinstance(v, str):
                            try:
                                ing["quantity"] = float(v) if "." in v else int(v)
                                fixed_qty += 1
                            except ValueError:
                                pass
                    if weight_type and "weight" in ing and ing["weight"] is not None:
                        v = ing["weight"]
                        if weight_type == "str" and not isinstance(v, str):
                            ing["weight"] = str(v)
                            fixed_weight += 1
                        elif weight_type == "number" and isinstance(v, str):
                            try:
                                ing["weight"] = float(v) if "." in v else int(v)
                                fixed_weight += 1
                            except ValueError:
                                pass

        if fixed_dates or fixed_qty or fixed_weight:
            logger.info(
                f"[A5] Normalized field shapes — createdAt fixed={fixed_dates}, "
                f"ingredient.quantity type fixed={fixed_qty}, "
                f"ingredient.weight type fixed={fixed_weight}."
            )
        return items

    def _normalize_system_fields(
        self, items: list, mostly_empty_fields: dict, url_template_fields: dict,
        numeric_bounds: dict, ingredient_conventions: dict = None, clamp_multiplier: float = 2.0,
    ) -> list:
        import uuid

        emptied, templated, clamped, ingredients_fixed, allergy_labels_removed = 0, 0, 0, 0, 0
        seen_template_values: Dict[str, set] = {}
        ingredient_conventions = ingredient_conventions or {}
        dominant_measure = ingredient_conventions.get("dominant_measure")
        weight_mirrors_quantity = ingredient_conventions.get("weight_mirrors_quantity")

        for item in items:
            if not isinstance(item, dict):
                continue

            ingredient_foods = " ".join(
                str(ing.get("food", "")) + " " + str(ing.get("text", ""))
                for ing in item.get("ingredients", []) or []
                if isinstance(ing, dict)
            ).lower()
            if ingredient_foods:
                for field, keyword_map in (
                    ("allergies", _ALLERGY_KEYWORDS),
                    ("foodIntolerances", _FOOD_INTOLERANCE_KEYWORDS),
                ):
                    labels = item.get(field)
                    if not isinstance(labels, list):
                        continue
                    kept = []
                    for label in labels:
                        kws = keyword_map.get(str(label).upper())
                        if kws and any(kw in ingredient_foods for kw in kws):
                            allergy_labels_removed += 1
                        else:
                            kept.append(label)
                    if len(kept) != len(labels):
                        item[field] = kept

            for field, default in (mostly_empty_fields or {}).items():
                if field in item and item[field] not in (None, "", [], {}):
                    item[field] = default
                    emptied += 1

            for field, prefix in (url_template_fields or {}).items():
                val = item.get(field)
                seen = seen_template_values.setdefault(field, set())
                is_missing = field not in item or not item.get(field)
                is_invalid_hex_suffix = False
                if (
                    not is_missing
                    and isinstance(val, str)
                    and val.startswith(prefix)
                    and not prefix.endswith("/")
                ):
                    suffix = val[len(prefix):]
                    if "." in suffix and suffix.rsplit(".", 1)[-1].isalnum():
                        suffix = suffix.rsplit(".", 1)[0]
                    is_invalid_hex_suffix = bool(suffix) and not all(
                        c in "0123456789abcdefABCDEF" for c in suffix
                    )
                needs_new = is_missing or is_invalid_hex_suffix or (
                    isinstance(val, str) and val and (not val.startswith(prefix) or val in seen)
                )
                if needs_new:
                    ext = ""
                    if isinstance(val, str) and "." in val.rsplit("/", 1)[-1]:
                        candidate_ext = val.rsplit(".", 1)[-1]
                        if candidate_ext.isalnum() and 1 <= len(candidate_ext) <= 5:
                            ext = "." + candidate_ext
                    if prefix.endswith("/"):
                        item[field] = f"{prefix}{field}-{uuid.uuid4().hex}{(ext or '.jpeg')}"
                    else:
                        item[field] = f"{prefix}{uuid.uuid4().hex}{ext}"
                    templated += 1
                    seen.add(item[field])
                elif isinstance(val, str) and val:
                    seen.add(val)

            for field, bound in (numeric_bounds or {}).items():
                val = item.get(field)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    low, high = bound["low"], bound["high"]
                    span = high - low if high > low else max(high, 1)
                    if low > 0:
                        hard_low = low / clamp_multiplier
                    else:
                        hard_low = low - span * (clamp_multiplier - 1)
                    hard_high = (
                        high * clamp_multiplier if high > 0
                        else high + span * (clamp_multiplier - 1)
                    )
                    if val < hard_low:
                        item[field] = low
                        clamped += 1
                    elif val > hard_high:
                        item[field] = high
                        clamped += 1

            if (dominant_measure or weight_mirrors_quantity) and isinstance(item.get("ingredients"), list):
                for ing in item["ingredients"]:
                    if not isinstance(ing, dict):
                        continue
                    changed = False
                    measure_raw = str(ing.get("measure", "")).strip().lower()
                    food_name = str(ing.get("food") or ing.get("text") or "").lower()

                    is_count_measure = measure_raw in _COUNT_MEASURE_HINTS
                    try:
                        current_weight = float(ing.get("weight"))
                    except (TypeError, ValueError):
                        current_weight = None
                    needs_estimate = current_weight is None or (
                        current_weight < 5 and measure_raw not in ("pinch", "pinches")
                    )

                    if is_count_measure and not needs_estimate:
                        continue

                    if is_count_measure or needs_estimate:
                        per_unit = next(
                            (g for kw, g in _INGREDIENT_UNIT_WEIGHTS_G.items() if kw in food_name),
                            None,
                        )
                        if per_unit is not None:
                            try:
                                count = float(ing.get("quantity", 1))
                                if count <= 0:
                                    count = 1
                            except (TypeError, ValueError):
                                count = 1
                            estimated = round(count * per_unit, 1)
                            weight_is_str = isinstance(ing.get("weight"), str)
                            qty_is_str = isinstance(ing.get("quantity"), str)
                            new_weight = str(estimated) if weight_is_str else estimated
                            new_quantity = str(estimated) if qty_is_str else estimated
                            if ing.get("weight") != new_weight:
                                ing["weight"] = new_weight
                                changed = True
                            if weight_mirrors_quantity and ing.get("quantity") != new_quantity:
                                ing["quantity"] = new_quantity
                                changed = True
                            if dominant_measure and ing.get("measure") != dominant_measure:
                                ing["measure"] = dominant_measure
                                changed = True
                            if changed:
                                ingredients_fixed += 1
                            continue
                        elif is_count_measure:
                            continue

                    multiplier = _LARGE_UNIT_MULTIPLIERS.get(measure_raw, 1)
                    if multiplier != 1:
                        for key in ("quantity", "weight"):
                            val = ing.get(key)
                            try:
                                numeric = float(val)
                                if numeric < 10:
                                    converted = numeric * multiplier
                                    ing[key] = str(converted) if isinstance(val, str) else converted
                                    changed = True
                            except (TypeError, ValueError):
                                pass

                    if dominant_measure and ing.get("measure") != dominant_measure:
                        ing["measure"] = dominant_measure
                        changed = True
                    if weight_mirrors_quantity and "quantity" in ing and (needs_estimate or multiplier != 1):
                        qty = ing["quantity"]
                        weight_is_str = isinstance(ing.get("weight"), str)
                        try:
                            target = str(qty) if weight_is_str else float(qty)
                        except (TypeError, ValueError):
                            target = qty
                        if ing.get("weight") != target:
                            ing["weight"] = target
                            changed = True
                    if changed:
                        ingredients_fixed += 1

        if emptied or templated or clamped or ingredients_fixed or allergy_labels_removed:
            logger.info(
                f"[A5] Normalized system fields — mostly-empty defaults applied={emptied}, "
                f"URL-template corrections={templated}, numeric bound clamps={clamped}, "
                f"ingredient measure/weight corrections={ingredients_fixed}, "
                f"allergy/intolerance label corrections={allergy_labels_removed}."
            )
        return items

    def _is_item_valid(
        self, item: dict, canonical_fields: list, categorical_enums: dict,
        ingredient_canonical_fields: Optional[list] = None,
        ingredient_field_types: Optional[dict] = None,
        mostly_empty_fields: Optional[dict] = None,
        nested_dict_field_schemas: Optional[dict] = None,
        total_weight_field: Optional[str] = None,
        max_missing_fields: int = 5,
    ) -> tuple:
        if not isinstance(item, dict):
            return False, "item is not a JSON object"

        missing = [f for f in canonical_fields if f not in item]

        if mostly_empty_fields is not None:
            hollow_now = [
                f for f in canonical_fields
                if f not in missing and f not in mostly_empty_fields
                and isinstance(item.get(f), dict) and item[f] == {}
            ]
            if hollow_now:
                return False, f"hollow (empty {{}}) field(s): {', '.join(sorted(hollow_now))}"

        identity_fields = [
            f for f in canonical_fields
            if any(h in f.lower() for h in _IDENTITY_KEY_HINTS)
        ]
        if identity_fields and any(f in missing for f in identity_fields):
            return False, "missing identity field (title/name-equivalent)"

        if len(missing) > max_missing_fields:
            return False, f"missing {len(missing)} top-level field(s): {', '.join(sorted(missing))}"

        for field, counts in categorical_enums.items():
            if field not in item:
                continue
            allowed = set(counts.keys())
            val = item[field]
            vals = val if isinstance(val, list) else ([val] if val is not None else [])
            for v in vals:
                if isinstance(v, str) and v not in allowed:
                    return False, f"invalid enum value for '{field}': {v!r}"

        if ingredient_canonical_fields:
            ings = item.get("ingredients")
            if isinstance(ings, list) and ings:
                canon_set = set(ingredient_canonical_fields)
                ftypes = ingredient_field_types or {}

                def _key_ok(ing: dict, k: str) -> bool:
                    if k not in ing:
                        return False
                    v = ing[k]
                    expected = ftypes.get(k)
                    if expected == "str":
                        return isinstance(v, str) and v != ""
                    if expected == "number":
                        return isinstance(v, (int, float)) and not isinstance(v, bool)
                    return True

                min_overlap = max(1, round(len(canon_set) * 0.4))
                bad_ingredient = None
                malformed = 0
                for ing in ings:
                    is_bad = (
                        not isinstance(ing, dict) or not ing
                        or sum(1 for k in canon_set if _key_ok(ing, k)) < min_overlap
                    )
                    if is_bad:
                        malformed += 1
                        if bad_ingredient is None:
                            bad_ingredient = ing
                if malformed / len(ings) > 0.5:
                    example_str = f" — e.g. {bad_ingredient!r}" if bad_ingredient is not None else ""
                    return False, (
                        f"{malformed}/{len(ings)} ingredient(s) don't match the "
                        f"real schema's shape (expected keys: "
                        f"{', '.join(sorted(canon_set))}){example_str}"
                    )

                _MAX_PLAUSIBLE_QUANTITY = 1e6
                absurd_qty_ing = None
                for ing in ings:
                    if not isinstance(ing, dict):
                        continue
                    qty = ing.get("quantity")
                    try:
                        fqty = float(qty) if isinstance(qty, (str, int, float)) and qty not in (None, "") else None
                        if fqty is not None and abs(fqty) > _MAX_PLAUSIBLE_QUANTITY:
                            absurd_qty_ing = (ing.get("food", "?"), qty)
                            break
                    except (ValueError, TypeError):
                        pass
                if absurd_qty_ing:
                    return False, (
                        f"absurd ingredient quantity value: "
                        f"{absurd_qty_ing[0]!r} has quantity={absurd_qty_ing[1]!r} "
                        f"(exceeds plausible recipe range of 0–{_MAX_PLAUSIBLE_QUANTITY:.0e})"
                    )

                if total_weight_field:
                    valid_weight_count = sum(
                        1 for ing in ings
                        if isinstance(ing, dict)
                        and isinstance(ing.get("weight"), (int, float))
                        and not isinstance(ing.get("weight"), bool)
                    )
                    if valid_weight_count / len(ings) < 0.5:
                        return False, (
                            f"only {valid_weight_count}/{len(ings)} ingredient(s) "
                            f"have a usable numeric 'weight' value"
                        )

        if nested_dict_field_schemas:
            for field, schema in nested_dict_field_schemas.items():
                subkeys = schema.get("subkeys", []) if isinstance(schema, dict) else schema
                val = item.get(field)
                if not isinstance(val, dict) or not val:
                    continue
                canon_set = set(subkeys)
                min_overlap = max(1, round(len(canon_set) * 0.4))
                bad_example = None
                malformed = 0
                non_dict_count = 0
                for k, iv in val.items():
                    if not isinstance(iv, dict) or not iv:
                        non_dict_count += 1
                        malformed += 1
                        if bad_example is None:
                            bad_example = (k, iv)
                        continue
                    if len(set(iv.keys()) & canon_set) < min_overlap:
                        malformed += 1
                        if bad_example is None:
                            bad_example = (k, iv)
                inner_vals = list(val.values())
                if inner_vals and (non_dict_count > 0 or malformed / len(inner_vals) > 0.5):
                    example_str = (
                        f" — e.g. {bad_example[0]!r}: {bad_example[1]!r}"
                        if bad_example is not None else ""
                    )
                    return False, (
                        f"'{field}': {malformed}/{len(inner_vals)} inner value(s) "
                        f"aren't proper nested objects (expected sub-keys: "
                        f"{', '.join(sorted(canon_set))}){example_str}"
                    )

        contradiction = _find_label_ingredient_contradiction(item)
        if contradiction:
            return False, contradiction

        return True, None

    def _drop_invalid_items(
        self, items: list, canonical_fields: list, categorical_enums: dict,
        ingredient_canonical_fields: Optional[list] = None,
        ingredient_field_types: Optional[dict] = None,
        mostly_empty_fields: Optional[dict] = None,
        nested_dict_field_schemas: Optional[dict] = None,
        total_weight_field: Optional[str] = None,
    ) -> tuple:
        valid = []
        dropped_values = []
        dropped_reasons = []
        for item in items:
            ok, reason = self._is_item_valid(
                item, canonical_fields, categorical_enums,
                ingredient_canonical_fields, ingredient_field_types, mostly_empty_fields,
                nested_dict_field_schemas, total_weight_field,
            )
            if ok:
                valid.append(item)
                continue
            dropped_reasons.append(reason or "invalid item")
            if isinstance(item, dict):
                for field, counts in categorical_enums.items():
                    val = item.get(field)
                    vals = val if isinstance(val, list) else ([val] if val is not None else [])
                    allowed = set(counts.keys())
                    for v in vals:
                        if isinstance(v, str) and v not in allowed:
                            dropped_values.append((field, v))
        return valid, dropped_values, dropped_reasons

    def _reconcile_weight_field(
        self, items: list, weight_field: Optional[str],
        array_field: str = "ingredients", tolerance: float = 0.25,
    ) -> tuple:
        if not weight_field:
            return items, 0
        fixed = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            ings = item.get(array_field)
            if not isinstance(ings, list) or not ings:
                continue
            weights = [
                ing.get("weight") for ing in ings
                if isinstance(ing, dict) and isinstance(ing.get("weight"), (int, float))
                and not isinstance(ing.get("weight"), bool)
            ]
            if len(weights) < max(1, len(ings) * 0.5):
                continue
            computed = sum(weights)
            if computed <= 0:
                continue
            current = item.get(weight_field)
            if (
                not isinstance(current, (int, float)) or isinstance(current, bool)
                or abs(current - computed) / computed > tolerance
            ):
                item[weight_field] = round(computed, 2)
                fixed += 1
        return items, fixed

    def _recover_hollow_nested_dict_entries(
        self, items: list, nested_dict_field_schemas: Optional[dict],
    ) -> tuple:
        recovered = 0
        for field, schema in (nested_dict_field_schemas or {}).items():
            if not isinstance(schema, dict):
                continue
            constants = schema.get("constants") or {}
            per_key_constants = schema.get("per_key_constants") or {}
            per_key_numeric_fallback = schema.get("per_key_numeric_fallback") or {}
            if not per_key_constants and not per_key_numeric_fallback:
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                val = item.get(field)
                if not isinstance(val, dict) or not val:
                    continue
                for outer_key, inner_val in val.items():
                    if not isinstance(inner_val, dict):
                        continue
                    key_constants = per_key_constants.get(outer_key, {})
                    key_fallback = per_key_numeric_fallback.get(outer_key, {})
                    if not key_constants and not key_fallback:
                        continue
                    for k, v in {**constants, **key_constants}.items():
                        if not inner_val.get(k):
                            inner_val[k] = v
                            recovered += 1
                    for k, v in key_fallback.items():
                        if not isinstance(inner_val.get(k), (int, float)) or isinstance(inner_val.get(k), bool):
                            inner_val[k] = v
                            recovered += 1
        return items, recovered

    def _apply_nested_dict_constants(
        self, items: list, nested_dict_field_schemas: Optional[dict],
    ) -> tuple:
        fixed = 0
        for field, schema in (nested_dict_field_schemas or {}).items():
            if not isinstance(schema, dict):
                continue
            constants = schema.get("constants") or {}
            per_key_constants = schema.get("per_key_constants") or {}
            if not constants and not per_key_constants:
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                val = item.get(field)
                if not isinstance(val, dict) or not val:
                    continue
                for outer_key, inner_val in val.items():
                    if not isinstance(inner_val, dict):
                        continue
                    for k, const_v in constants.items():
                        if inner_val.get(k) != const_v:
                            inner_val[k] = const_v
                            fixed += 1
                    key_constants = per_key_constants.get(outer_key)
                    if key_constants:
                        for k, const_v in key_constants.items():
                            if k in constants:
                                continue
                            if inner_val.get(k) != const_v:
                                inner_val[k] = const_v
                                fixed += 1
        return items, fixed

    @staticmethod
    def _normalize_food_name(name: str) -> str:
        name = (name or "").lower().strip()
        idx = name.find("(")
        if idx > 0:
            name = name[:idx].strip()
        if name.endswith("s") and len(name) > 3:
            name = name[:-1]
        return name

    def _reconcile_ingredient_weights_with_lookup(
        self, items: list, ingredient_weight_lookup: Optional[dict],
        array_field: str = "ingredients", tolerance_factor: float = 5.0,
    ) -> tuple:
        if not ingredient_weight_lookup:
            return items, 0
        fixed = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            ings = item.get(array_field)
            if not isinstance(ings, list):
                continue
            for ing in ings:
                if not isinstance(ing, dict):
                    continue
                combined = " ".join(
                    str(ing.get(k, "")) for k in ("food", "text")
                ).lower()
                if not combined.strip():
                    continue
                best_food, best_len = None, 0
                for food_name in ingredient_weight_lookup:
                    if not food_name:
                        continue
                    pattern = r"\b" + re.escape(food_name) + r"s?\b"
                    if len(food_name) > best_len and re.search(pattern, combined):
                        best_food, best_len = food_name, len(food_name)
                if not best_food:
                    continue
                known_weight = ingredient_weight_lookup[best_food]
                if not isinstance(known_weight, (int, float)) or known_weight <= 0:
                    continue

                try:
                    qty_float = float(ing.get("quantity") or 0)
                    if qty_float > 0:
                        try:
                            w_existing_float = float(ing.get("weight") or 0)
                        except (TypeError, ValueError):
                            w_existing_float = 0
                        if w_existing_float > 0 and abs(qty_float - w_existing_float) / qty_float < 0.1:
                            pass
                        elif known_weight < qty_float / 10:
                            continue
                except (TypeError, ValueError):
                    pass

                w_raw = ing.get("weight")
                if w_raw in (None, ""):
                    write_as_str = any(
                        isinstance(other.get("weight"), str) and other.get("weight") not in (None, "")
                        for other in ings if isinstance(other, dict) and other is not ing
                    )
                    corrected = round(known_weight, 2)
                    ing["weight"] = str(corrected) if write_as_str else corrected
                    fixed += 1
                    continue
                write_as_str = False
                if isinstance(w_raw, bool):
                    continue
                elif isinstance(w_raw, (int, float)):
                    w = w_raw
                elif isinstance(w_raw, str):
                    try:
                        w = float(w_raw)
                    except ValueError:
                        continue
                    write_as_str = True
                else:
                    continue
                if w <= 0:
                    continue

                if w > known_weight * tolerance_factor or w < known_weight / tolerance_factor:
                    corrected = round(known_weight, 2)
                    ing["weight"] = str(corrected) if write_as_str else corrected
                    fixed += 1
        return items, fixed

    def _reconcile_weight_to_quantity(
        self, items: list, array_field: str = "ingredients",
        only_when_mirrors: Optional[bool] = None,
    ) -> tuple:
        if not only_when_mirrors:
            return items, 0
        fixed = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            ings = item.get(array_field)
            if not isinstance(ings, list):
                continue
            for ing in ings:
                if not isinstance(ing, dict):
                    continue
                q_raw = ing.get("quantity")
                w_raw = ing.get("weight")
                if q_raw is None:
                    continue
                try:
                    q = float(q_raw)
                except (TypeError, ValueError):
                    continue
                if q <= 0:
                    continue

                if w_raw is None or w_raw == "" or w_raw == 0:
                    write_as_str = isinstance(q_raw, str)
                    ing["weight"] = str(round(q, 2)) if write_as_str else round(q, 2)
                    fixed += 1
                    continue

                try:
                    w = float(w_raw)
                except (TypeError, ValueError):
                    continue
                if w <= 0:
                    continue
                if abs(q - w) / q > 0.05:
                    write_as_str = isinstance(w_raw, str)
                    corrected = round(q, 2)
                    ing["weight"] = str(corrected) if write_as_str else corrected
                    fixed += 1
        return items, fixed

    def _reconcile_numeric_string_fields(
        self, items: list, numeric_fields: Optional[list] = None,
    ) -> tuple:
        if not numeric_fields or not isinstance(items, list):
            return items, 0
        fixed = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            for field in numeric_fields:
                val = item.get(field)
                if isinstance(val, str):
                    try:
                        item[field] = float(val) if '.' in val else int(val)
                        fixed += 1
                    except (TypeError, ValueError):
                        pass
        return items, fixed

    def _reconcile_ingredient_energy_with_lookup(
        self, items: list, energy_field: Optional[str], energy_ratio_lookup: Optional[dict],
        array_field: str = "ingredients", tolerance_factor: float = 3.0,
    ) -> tuple:
        if not energy_field or not energy_ratio_lookup:
            return items, 0
        fixed = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            ings = item.get(array_field)
            if not isinstance(ings, list):
                continue
            for ing in ings:
                if not isinstance(ing, dict):
                    continue
                combined = " ".join(
                    str(ing.get(k, "")) for k in ("food", "text")
                ).lower()
                if not combined.strip():
                    continue
                best_food, best_len = None, 0
                for food_name in energy_ratio_lookup:
                    if not food_name:
                        continue
                    pattern = r"\b" + re.escape(food_name) + r"s?\b"
                    if len(food_name) > best_len and re.search(pattern, combined):
                        best_food, best_len = food_name, len(food_name)
                if not best_food:
                    continue
                ratio = energy_ratio_lookup[best_food]
                if not isinstance(ratio, (int, float)) or ratio <= 0:
                    continue

                w_raw = ing.get("weight")
                try:
                    w = float(w_raw)
                except (TypeError, ValueError):
                    continue
                if isinstance(w_raw, bool) or w <= 0:
                    continue
                expected = ratio * w

                e_raw = ing.get(energy_field)
                e = None
                if not isinstance(e_raw, bool):
                    try:
                        e = float(e_raw)
                    except (TypeError, ValueError):
                        e = None

                if e is None or e <= 0 or e > expected * tolerance_factor or e < expected / tolerance_factor:
                    write_as_str = isinstance(e_raw, str)
                    corrected = round(expected, 2)
                    ing[energy_field] = str(corrected) if write_as_str else corrected
                    fixed += 1
        return items, fixed

    @staticmethod
    def _normalize_nutrient_name(name: str) -> str:
        n = (name or "").strip().lower()
        n = re.sub(r"\s*\([^)]*\)\s*$", "", n).strip()
        if n.endswith("s") and len(n) > 3:
            n = n[:-1]
        return n

    def _reconcile_nutrient_labels_with_lookup(
        self, items: list, nutrient_label_lookup: Optional[dict],
        array_field: str = "nutrients",
    ) -> tuple:
        if not nutrient_label_lookup:
            return items, 0
        fixed = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            nutrients = item.get(array_field)
            if not isinstance(nutrients, list):
                continue
            for n in nutrients:
                if not isinstance(n, dict):
                    continue
                name = n.get("name")
                if not isinstance(name, str) or not name:
                    continue
                real_label = nutrient_label_lookup.get(name.strip().lower())
                if not real_label:
                    real_label = nutrient_label_lookup.get(
                        self._normalize_nutrient_name(name)
                    )
                if not real_label:
                    continue
                if n.get("label") != real_label:
                    n["label"] = real_label
                    fixed += 1
        return items, fixed

    def _remove_contradicted_dietary_labels(self, items: list) -> tuple:
        removed = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            foods = _item_ingredient_text(item)
            if not foods:
                continue
            for field in _DIET_EXCLUSION_FIELD_CANDIDATES:
                labels = item.get(field)
                if not (labels and isinstance(labels, list)):
                    continue
                kept = []
                for label in labels:
                    kws = _NUTRITION_LABEL_KEYWORDS.get(_normalize_label(label))
                    if kws and any(kw in foods for kw in kws):
                        removed += 1
                    else:
                        kept.append(label)
                if len(kept) != len(labels):
                    item[field] = kept
            for field in _DIET_LABEL_FIELD_CANDIDATES:
                labels = item.get(field)
                if not (labels and isinstance(labels, list)):
                    continue
                kept = []
                for label in labels:
                    kws = _DIET_TYPE_CONTRADICTION_KEYWORDS.get(_normalize_label(label))
                    if kws and any(kw in foods for kw in kws):
                        removed += 1
                    else:
                        kept.append(label)
                if len(kept) != len(labels):
                    item[field] = kept
            for field in _CAUTIONS_FIELD_CANDIDATES:
                labels = item.get(field)
                if not (labels and isinstance(labels, list)):
                    continue
                kept = []
                for label in labels:
                    kws = _CAUTION_KEYWORDS.get(_normalize_label(label))
                    if kws and not any(kw in foods for kw in kws):
                        removed += 1
                    else:
                        kept.append(label)
                if len(kept) != len(labels):
                    item[field] = kept
        return items, removed

    def _clamp_per_serving(
        self, items: list, per_serving_bounds: Optional[dict], clamp_multiplier: float = 1.5,
    ) -> tuple:
        if not per_serving_bounds:
            return items, 0
        field = per_serving_bounds["field"]
        yield_field = per_serving_bounds["yield_field"]
        low, high = per_serving_bounds["low"], per_serving_bounds["high"]
        hard_low = low / clamp_multiplier if low > 0 else low
        hard_high = high * clamp_multiplier
        fixed = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            y = item.get(yield_field)
            v = item.get(field)
            if not (isinstance(y, (int, float)) and not isinstance(y, bool) and y > 0):
                continue
            if not (isinstance(v, (int, float)) and not isinstance(v, bool)):
                continue
            ratio = v / y
            new_v = None
            if ratio < hard_low:
                new_v = round(low * y, 2)
            elif ratio > hard_high:
                new_v = round(high * y, 2)
            if new_v is not None:
                item[field] = new_v
                fixed += 1
                self._sync_nested_energy_value(item, v, new_v)
        return items, fixed

    @staticmethod
    def _sync_nested_energy_value(item: dict, old_v, new_v) -> None:
        if not (isinstance(old_v, (int, float)) and old_v and not isinstance(old_v, bool)):
            return
        ratio = new_v / old_v
        for container_field, is_percentage in (("totalNutrients", False), ("totalDaily", True)):
            container = item.get(container_field)
            if not isinstance(container, dict):
                continue
            for entry in container.values():
                if not isinstance(entry, dict):
                    continue
                if "energy" not in str(entry.get("label", "")).lower():
                    continue
                qty = entry.get("quantity")
                if not (isinstance(qty, (int, float)) and not isinstance(qty, bool)):
                    continue
                entry["quantity"] = round(new_v, 2) if not is_percentage else round(qty * ratio, 2)

    def _reconcile_calories_with_nutrients(self, items: list, tolerance: float = 0.02) -> tuple:
        fixed = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            calories = item.get("calories")
            if not (isinstance(calories, (int, float)) and not isinstance(calories, bool)):
                continue
            nutrients = item.get("totalNutrients")
            if not isinstance(nutrients, dict):
                continue
            for entry in nutrients.values():
                if not isinstance(entry, dict) or "energy" not in str(entry.get("label", "")).lower():
                    continue
                qty = entry.get("quantity")
                is_numeric = isinstance(qty, (int, float)) and not isinstance(qty, bool)
                mismatched = (not is_numeric) or qty == 0 or (
                    abs(qty - calories) / max(abs(calories), 1.0) > tolerance
                )
                if mismatched:
                    if is_numeric and qty:
                        self._sync_nested_energy_value(item, qty, calories)
                    else:
                        entry["quantity"] = round(calories, 2)
                    fixed += 1
                break
        return items, fixed

    def _reconcile_energy_daily_value(
        self, items: list, daily_reference_kcal: float = 2000.0, tolerance: float = 0.05,
    ) -> tuple:
        if not isinstance(items, list):
            return items, 0
        fixed = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            calories = item.get("calories")
            if not (isinstance(calories, (int, float)) and not isinstance(calories, bool)):
                continue
            daily = item.get("totalDaily")
            if not isinstance(daily, dict):
                continue
            for entry in daily.values():
                if not isinstance(entry, dict) or "energy" not in str(entry.get("label", "")).lower():
                    continue
                expected = round(calories / daily_reference_kcal * 100, 2)
                qty = entry.get("quantity")
                is_numeric = isinstance(qty, (int, float)) and not isinstance(qty, bool)
                if not is_numeric or qty <= 0 or abs(qty - expected) > max(1.0, expected * tolerance):
                    entry["quantity"] = expected
                    fixed += 1
                break
        return items, fixed

    @staticmethod
    def _looks_like_fake_precision(value) -> bool:
        if not isinstance(value, float) or isinstance(value, bool):
            return False
        s = repr(value)
        if "." not in s:
            return False
        frac = s.split(".", 1)[1].rstrip("0")
        if len(frac) < 8:
            return False
        prefix = frac[:8]
        ascending = "1234567890123456789012345678901234567890"
        descending = "9876543210987654321098765432109876543210"
        return prefix in ascending or prefix in descending

    def _clean_fake_precision(self, items: list) -> tuple:
        fixed = 0

        def _walk(obj):
            nonlocal fixed
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        _walk(v)
                    elif self._looks_like_fake_precision(v):
                        obj[k] = round(v, 2)
                        fixed += 1
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    if isinstance(v, (dict, list)):
                        _walk(v)
                    elif self._looks_like_fake_precision(v):
                        obj[i] = round(v, 2)
                        fixed += 1

        for item in items:
            if isinstance(item, dict):
                _walk(item)
        return items, fixed

    def _rank_items_by_implausibility(
        self, items: list,
        calories_per_serving_bounds: Optional[dict],
        weight_per_serving_bounds: Optional[dict],
        url_template_fields: Optional[list] = None,
    ) -> list:
        def _deviation(item: dict, bounds: Optional[dict]) -> float:
            if not bounds:
                return 0.0
            field, yield_field = bounds["field"], bounds["yield_field"]
            low, high = bounds["low"], bounds["high"]
            y, v = item.get(yield_field), item.get(field)
            if not (isinstance(y, (int, float)) and not isinstance(y, bool) and y > 0):
                return 0.0
            if not (isinstance(v, (int, float)) and not isinstance(v, bool)):
                return 0.0
            ratio = v / y
            if ratio < low and low > 0:
                return (low - ratio) / low
            if ratio > high and high > 0:
                return (ratio - high) / high
            return 0.0

        duplicated_indices: set = set()
        for field in (url_template_fields or []):
            seen: dict = {}
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                val = item.get(field)
                if val in (None, ""):
                    continue
                seen.setdefault(val, []).append(idx)
            for idx_list in seen.values():
                if len(idx_list) > 1:
                    duplicated_indices.update(idx_list)

        scored = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                scored.append((idx, 1.0))
                continue
            score = (
                _deviation(item, calories_per_serving_bounds)
                + _deviation(item, weight_per_serving_bounds)
            )
            if idx in duplicated_indices:
                score = max(score, 1.0)
            scored.append((idx, score))
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored

    def _build_avoid_note(self, rejected_values: dict, rejected_reasons: Optional[list] = None) -> str:
        parts = []
        if rejected_values:
            lines = [
                f"  - {field}: {', '.join(sorted(vals))}"
                for field, vals in rejected_values.items()
            ]
            parts.append(
                "PREVIOUSLY REJECTED VALUES — these exact values were already "
                "tried and are INVALID (not in the allowed list above); do NOT "
                "reuse them:\n" + "\n".join(lines)
            )
        if rejected_reasons:
            nl_contradictions = set()
            other_reasons = []
            for r in (rejected_reasons or []):
                if "contradicts actual ingredients" in r.lower():
                    try:
                        label = r.split("'")[1]
                        kw = r.split("keyword match: '")[1].split("'")[0]
                        nl_contradictions.add((label, kw))
                    except (IndexError, ValueError):
                        other_reasons.append(r)
                else:
                    other_reasons.append(r)

            if nl_contradictions:
                def _directive(label: str, kw: str) -> str:
                    norm = label.upper().replace("-", "_").replace(" ", "_")
                    if norm.endswith("_FREE"):
                        category = norm.rsplit("_FREE", 1)[0].split("_")[-1].lower()
                        return (
                            f"  - NEVER assign '{label}' to a recipe that contains "
                            f"'{kw}' (or any similar {category} ingredient)"
                        )
                    return (
                        f"  - NEVER assign '{label}' to a recipe that contains '{kw}'"
                    )
                directives = "\n".join(
                    _directive(label, kw) for label, kw in sorted(nl_contradictions)
                )
                parts.append(
                    "DIETARY LABEL VIOLATIONS — previous items were REJECTED "
                    "because their dietary/allergen labels contradicted actual "
                    "ingredients. These are HARD RULES:\n" + directives
                )
            if other_reasons:
                recent = list(dict.fromkeys(other_reasons[-10:]))[-5:]
                parts.append(
                    "PREVIOUSLY REJECTED ITEMS — recent attempts were rejected "
                    "for these structural reasons; make sure this attempt "
                    "doesn't repeat them:\n"
                    + "\n".join(f"  - {r}" for r in recent)
                )
        if not parts:
            return ""
        return "\n\n" + "\n\n".join(parts)

    def _fill_missing_categorical_defaults(
        self, items: list, categorical_enums: dict,
    ) -> tuple:
        if not categorical_enums:
            return items, {}

        filled_counts: dict = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            for field, counts in categorical_enums.items():
                if field in item:
                    continue
                if not counts:
                    continue
                default_value = next(iter(counts.keys()))
                item[field] = default_value
                filled_counts[field] = filled_counts.get(field, 0) + 1

        return items, filled_counts

    def _readable_descriptions(self, items: list) -> list:
        out = []
        for item in items:
            if not isinstance(item, dict):
                continue
            parts = []
            for key in ("name", "title", "dishType", "cookingMethod", "cuisine"):
                val = item.get(key)
                if val:
                    parts.append(f"{key}={val}")
            ing = item.get("ingredients")
            if isinstance(ing, list):
                foods = [i.get("food") or i.get("text") for i in ing[:4] if isinstance(i, dict)]
                foods = [f for f in foods if f]
                if foods:
                    parts.append("ingredients=" + ",".join(str(f) for f in foods))
            out.append(" | ".join(parts) if parts else str(item)[:80])
        return out

    def _pick_backend(self, settings) -> Optional[str]:
        override = getattr(settings, "GENERATION_BACKEND", "auto")
        if override in ("ollama", "anthropic", "huggingface", "groq", "nvidia"):
            return override
        if self.llm.anthropic_available:
            return "anthropic"
        if self.llm.nvidia_available:
            return "nvidia"
        if self.llm.groq_available:
            return "groq"
        if self.llm.huggingface_available:
            return "huggingface"
        return "ollama"

    def _abbreviate(self, obj, depth=0):
        if isinstance(obj, dict):
            if depth >= 1 and len(obj) > 6:
                items = list(obj.items())[:6]
                result = {k: self._abbreviate(v, depth + 1) for k, v in items}
                result[f"…{len(obj) - 6} more entries"] = "…"
                return result
            return {k: self._abbreviate(v, depth + 1)
                    for k, v in obj.items() if depth < 2}
        if isinstance(obj, list):
            return ([self._abbreviate(x, depth + 1) for x in obj[:2]]
                    + ([f"…{len(obj) - 2} more"] if len(obj) > 2 else []))
        return obj

    async def _gen_csv(self, task, features, ctx, user_ctx, retry_note):
        from config import settings
        qty = task.get("quantity") or 10
        return self.llm.chat([
            self._sys("Output ONLY valid CSV with a header row. No explanation."),
            self._usr(
                f"Task: {task.get('summary', '')}  Generate {qty} rows.\n"
                f"Features: {json.dumps(features, default=str)[:settings.MAX_LLM_CONTEXT_CHARS]}\n"
                f"Reference:\n{ctx[:settings.MAX_LLM_CONTEXT_CHARS]}\n{retry_note}"
            ),
        ], temperature=0.6)

    async def _gen_text(self, task, features, ctx, user_ctx, retry_note):
        from config import settings
        return self.llm.chat([
            self._sys("You are a professional writer. Produce high-quality, accurate output."),
            self._usr(
                f"Instructions: {user_ctx}\nTask: {task.get('summary', '')}\n"
                f"Patterns: {features.get('key_patterns', [])}\n"
                f"Reference:\n{ctx[:settings.MAX_LLM_CONTEXT_CHARS]}\n{retry_note}"
            ),
        ], temperature=0.7)

    async def _gen_image(self, task, features, ctx, user_ctx):
        from config import settings
        return self.llm.chat([
            self._sys(
                "Generate complete, runnable matplotlib Python code. "
                "Save to output_image.png. Output ONLY the Python code."
            ),
            self._usr(
                f"Task: {task.get('summary', '')}\nInstructions: {user_ctx}\n"
                f"Features: {json.dumps(features, default=str)[:settings.MAX_LLM_CONTEXT_CHARS]}\n"
                f"Data context:\n{ctx[:settings.MAX_LLM_CONTEXT_CHARS]}"
            ),
        ], temperature=0.3)
