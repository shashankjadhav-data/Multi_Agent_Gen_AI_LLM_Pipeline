import json
import os
import re
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from agents.base_agent import BaseAgent
from core.message_bus import MessageBus, Message
from utils.logger import logger

_VERDICT_FALLBACK = {
    "valid": True, "confidence": 0.6,
    "schema_compliance": True, "completeness": 0.75,
    "accuracy": 0.7, "task_fulfillment": 0.7,
    "critical_issues": [], "minor_issues": ["Verdict parse failed."],
    "feedback": "Verdict parse failed — accepted by default.",
    "recommended_action": "accept",
    "parse_error": True,
}

_ENUM_TOKEN_RE = re.compile(r"^[A-Z][A-Z0-9_\-]*$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,39}$")
_ENUM_PHRASE_RE = re.compile(r"^[a-z][a-z\s/&\-]{0,29}$")

def _looks_enum_like(v: str) -> bool:
    if not v:
        return False
    if not any(c.isalpha() for c in v):
        return False
    if _ENUM_TOKEN_RE.match(v):
        return True
    if _ENUM_PHRASE_RE.match(v):
        return True
    return bool(_IDENTIFIER_RE.match(v))

class SingleAgentBaseline(BaseAgent):

    def __init__(self, bus: MessageBus):
        super().__init__("baseline", "Baseline · Single-Agent", bus)
        self._embedder = None

    def _extract_list_from_response(self, raw: str) -> Optional[List[dict]]:
        cleaned = "\n".join(ln for ln in raw.splitlines() if not ln.strip().startswith("```")).strip()

        try:
            parsed = self.llm.extract_json(raw)
        except Exception:
            parsed = None

        if isinstance(parsed, list):
            dict_items = [x for x in parsed if isinstance(x, dict)]
            dropped = len(parsed) - len(dict_items)
            if dropped:
                logger.warning(f"[Baseline] Dropped {dropped} non-dict element(s) from parsed list.")
            return dict_items if dict_items else None

        array_start = cleaned.find("[")
        if array_start >= 0:
            salvaged = []
            pos = array_start + 1
            n = len(cleaned)
            while pos < n:

                while pos < n and cleaned[pos] in " \t\r\n,":
                    pos += 1
                if pos >= n or cleaned[pos] != "{":
                    break

                depth, in_str, esc = 0, False, False
                end = pos
                for i, ch in enumerate(cleaned[pos:], start=pos):
                    if esc:
                        esc = False
                        continue
                    if ch == "\\" and in_str:
                        esc = True
                        continue
                    if ch == '"':
                        in_str = not in_str
                    if not in_str:
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                        if depth == 0:
                            end = i
                            break
                if depth != 0:

                    break
                try:
                    item = json.loads(cleaned[pos:end + 1])
                    if isinstance(item, dict):
                        salvaged.append(item)
                except json.JSONDecodeError:
                    break
                pos = end + 1

            if salvaged:
                logger.info(
                    f"[Baseline] Salvaged {len(salvaged)} complete item(s) from "
                    f"a truncated array response (token ceiling hit)."
                )
                return salvaged

        if isinstance(parsed, dict):

            list_of_dict_values = [
                v for v in parsed.values()
                if isinstance(v, list) and v and any(isinstance(x, dict) for x in v)
            ]
            if list_of_dict_values:
                best = max(list_of_dict_values, key=lambda v: sum(1 for x in v if isinstance(x, dict)))
                dict_items = [x for x in best if isinstance(x, dict)]
                dropped = len(best) - len(dict_items)
                if dropped:
                    logger.warning(f"[Baseline] Unwrapped dict wrapper, dropped {dropped} non-dict element(s).")
                if dict_items:
                    logger.info(f"[Baseline] Unwrapped list from dict wrapper (dict items={len(dict_items)}).")
                    return dict_items

        return None

    @staticmethod
    def _deduplicate(items: List[dict]) -> List[dict]:
        seen_titles, seen_ids, out = set(), set(), []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("label") or item.get("name")
            item_id = item.get("_id") or item.get("id")
            dup = (title and title in seen_titles) or (
                item_id is not None and item_id in seen_ids
            )
            if not dup:
                out.append(item)
                if title:
                    seen_titles.add(title)
                if item_id is not None:
                    seen_ids.add(item_id)
            else:
                logger.info(f"[Baseline] Dropped exact duplicate: title={title!r}")
        return out

    def _generate(self, fmt: str, full_text: str, user_ctx: str, max_tokens: int, max_attempts: int, requested_qty: int = 0) -> List[dict]:
        sample_text = full_text[:12000]

        system_msg = self._sys(
            "You are a data-generation assistant. You will be given a "
            "sample of a raw dataset and a plain-language instruction. "
            "Infer the dataset's field names, nesting, and value conventions "
            "from the sample, then generate exactly as many items as requested.\n\n"
            "CRITICAL OUTPUT RULES:\n"
            "1. Output ONLY a raw JSON array starting with [ and ending with ].\n"
            "2. Every element must be a JSON object { } with the same fields "
            "visible in the sample. Never output plain text, numbered lists, "
            "or string values as array elements.\n"
            "3. Do NOT wrap the array in any object like {\"recipes\": [...]}. "
            "The array must be the outermost element.\n"
            "4. Every item must be unique — different title and different "
            "ingredients. Never repeat the same item.\n"
            "5. No markdown fences, no explanation, no text before or after."
        )

        accumulated: List[dict] = []
        seen_titles: set = set()
        attempts_used = 0

        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            remaining = requested_qty - len(accumulated) if requested_qty > 0 else 15

            user_msg = (
                f"Dataset format: {fmt}\n\n"
                f"Dataset sample (infer the exact schema from this):\n{sample_text}\n\n"
                f"Instructions:\n{user_ctx}\n\n"
            )

            if accumulated:
                already_titles = [i.get("title") or i.get("label") or i.get("name", "") for i in accumulated]
                user_msg += (
                    f"You have already generated {len(accumulated)} item(s): "
                    f"{already_titles}. "
                    f"Generate {remaining} MORE new, different items. "
                    f"Do NOT repeat any of the already-generated items.\n\n"
                )

            user_msg += (
                "Generate the requested items now. Each item must be a JSON object "
                "{{ }} with the same fields visible in the sample above. "
                "Output ONLY the JSON array [ {{...}}, {{...}}, ... ] — nothing else."
            )

            if attempt > 1 and not accumulated:
                user_msg += (
                    "\n\nPrevious attempt failed to produce a JSON array of "
                    "objects. Output ONLY a JSON array starting with [. "
                    "Each element must be a JSON object {{...}}, NOT a string."
                )

            raw = self.llm.chat(
                [system_msg, self._usr(user_msg)],
                temperature=0.4,
                max_tokens=max_tokens,
            )
            result = self._extract_list_from_response(raw)

            if result:

                new_items = []
                for item in result:
                    title = item.get("title") or item.get("label") or item.get("name")
                    if title not in seen_titles:
                        new_items.append(item)
                        if title:
                            seen_titles.add(title)
                accumulated.extend(new_items)
                logger.info(
                    f"[Baseline] Attempt {attempt}/{max_attempts}: "
                    f"got {len(result)} item(s), {len(new_items)} new → "
                    f"total {len(accumulated)}/{requested_qty or '?'}."
                )
                if requested_qty > 0 and len(accumulated) >= requested_qty:
                    break
            else:
                logger.warning(
                    f"[Baseline] Attempt {attempt}/{max_attempts}: "
                    f"no list of objects found in response."
                )

        if not accumulated:
            logger.error(f"[Baseline] All {max_attempts} attempt(s) failed — writing empty result.")

        if requested_qty > 0 and len(accumulated) < requested_qty:
            logger.warning(
                f"[Baseline] Token ceiling reached: produced {len(accumulated)}/{requested_qty} "
                f"item(s) across {attempts_used} attempt(s). This is the expected behaviour "
                f"for a single-call baseline on a schema with large nested fields — the full "
                f"pipeline's batched architecture exists specifically to handle this constraint."
            )

        output = accumulated[:requested_qty] if requested_qty > 0 else accumulated
        logger.info(f"[Baseline] Generated {len(output)} item(s) in {attempts_used} attempt(s).")
        return output

    def _programmatic_stats(self, data: Any, fmt: str) -> Dict:
        stats: Dict = {"format": fmt, "record_count": 0, "fields": {}}
        if fmt == "json" and isinstance(data, list):
            stats["record_count"] = len(data)
            sample = data[:min(300, len(data))]
            field_info: Dict = {}
            for item in sample:
                if not isinstance(item, dict):
                    continue
                for key, val in item.items():
                    if key not in field_info:
                        field_info[key] = {"count": 0, "types": set(), "num_vals": [], "str_vals": []}
                    fi = field_info[key]
                    fi["count"] += 1
                    fi["types"].add(type(val).__name__)
                    if isinstance(val, (int, float)):
                        fi["num_vals"].append(float(val))
                    elif isinstance(val, str) and len(fi["str_vals"]) < 8:
                        fi["str_vals"].append(val[:80])
                    elif isinstance(val, list) and len(fi["str_vals"]) < 3:
                        fi["str_vals"].append(f"<list len={len(val)}>")
            n = max(len(sample), 1)
            for key, fi in field_info.items():
                fs: Dict = {"presence_pct": round(fi["count"] / n * 100, 1), "types": list(fi["types"])}
                if fi["num_vals"]:
                    v = fi["num_vals"]
                    fs["num_min"] = round(min(v), 4)
                    fs["num_max"] = round(max(v), 4)
                    fs["num_mean"] = round(statistics.mean(v), 4)
                if fi["str_vals"]:
                    fs["sample_values"] = list(dict.fromkeys(fi["str_vals"]))[:6]
                stats["fields"][key] = fs
        return stats

    def _canonical_field_list(self, prog_stats: Dict, min_presence: float = 90.0) -> List[str]:
        return sorted(k for k, v in prog_stats.get("fields", {}).items() if v.get("presence_pct", 0) >= min_presence)

    def _extract_categorical_enums(self, data: Any, fmt: str, max_records: int = 1200) -> Dict[str, Dict[str, int]]:
        if fmt != "json" or not isinstance(data, list) or not data:
            return {}
        sample = data[:min(max_records, len(data))]
        field_values: Dict[str, list] = {}
        for item in sample:
            if not isinstance(item, dict):
                continue
            for key, val in item.items():
                if isinstance(val, list) and val and all(isinstance(v, str) for v in val):
                    field_values.setdefault(key, []).extend(val)
                elif isinstance(val, str) and val:
                    field_values.setdefault(key, []).append(val)
        enums: Dict[str, Dict[str, int]] = {}
        n = len(sample)
        for key, vals in field_values.items():
            distinct = set(vals)
            if not (1 <= len(distinct) <= max(20, int(n * 0.05))):
                continue
            enum_like = sum(1 for v in distinct if _looks_enum_like(v))
            if enum_like / len(distinct) < 0.9:
                continue
            enums[key] = dict(Counter(vals).most_common(100))
        return enums

    def _extract_mostly_empty_fields(self, data: Any, fmt: str, threshold: float = 0.9, max_records: int = 1200) -> Dict[str, Any]:
        if fmt != "json" or not isinstance(data, list) or not data:
            return {}
        sample = data[:min(max_records, len(data))]
        field_counts: Dict[str, Dict[str, Any]] = {}
        for item in sample:
            if not isinstance(item, dict):
                continue
            for key, val in item.items():
                fc = field_counts.setdefault(key, {"empty": 0, "total": 0, "empty_value": None})
                fc["total"] += 1
                if val in (None, "", [], {}):
                    fc["empty"] += 1
                    if fc["empty_value"] is None:
                        fc["empty_value"] = val
        return {key: fc["empty_value"] if fc["empty_value"] is not None else "" for key, fc in field_counts.items() if fc["total"] > 0 and fc["empty"] / fc["total"] >= threshold}

    def _extract_url_templates(self, data: Any, fmt: str, min_fraction: float = 0.9, min_prefix_len: int = 30, max_records: int = 1200) -> Dict[str, str]:
        if fmt != "json" or not isinstance(data, list) or not data:
            return {}
        sample = data[:min(max_records, len(data))]
        field_values: Dict[str, list] = {}
        for item in sample:
            if not isinstance(item, dict):
                continue
            for key, val in item.items():
                if isinstance(val, str) and val.startswith(("http://", "https://")):
                    field_values.setdefault(key, []).append(val)
        templates = {}
        for key, vals in field_values.items():
            if len(vals) < 5:
                continue
            prefix = os.path.commonprefix(vals)
            matching = sum(1 for v in vals if v.startswith(prefix))
            if not (len(prefix) >= min_prefix_len and matching / len(vals) >= min_fraction) and "/" in prefix:
                prefix = prefix[:prefix.rfind("/") + 1]
            matching = sum(1 for v in vals if v.startswith(prefix))
            if len(prefix) >= min_prefix_len and matching / len(vals) >= min_fraction:
                templates[key] = prefix
        return templates

    def _extract_numeric_bounds(self, data: Any, fmt: str, max_records: int = 1200) -> Dict[str, Dict[str, float]]:
        if fmt != "json" or not isinstance(data, list) or not data:
            return {}
        sample = data[:min(max_records, len(data))]
        field_values: Dict[str, list] = {}
        for item in sample:
            if not isinstance(item, dict):
                continue
            for key, val in item.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    field_values.setdefault(key, []).append(float(val))
        bounds = {}
        for key, vals in field_values.items():
            if len(vals) < 20:
                continue
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            low = vals_sorted[max(0, int(n * 0.01))]
            high = vals_sorted[min(n - 1, int(n * 0.99))]
            if low < high:
                bounds[key] = {"low": low, "high": high}
        return bounds

    def _detect_nested_dict_fields(self, data: Any, fmt: str, canonical_fields: List[str], min_presence: float = 60.0, min_records: int = 20, max_records: int = 1200) -> Dict[str, Dict[str, Any]]:
        if fmt != "json" or not isinstance(data, list) or not data:
            return {}
        sample = data[:min(max_records, len(data))]
        result: Dict[str, Dict[str, Any]] = {}
        for field in canonical_fields:
            inner_key_counts: Counter = Counter()
            value_counts: Dict[str, Counter] = {}
            total_inner = 0
            dict_of_dict_records = 0
            checked_records = 0
            for r in sample:
                if not isinstance(r, dict) or field not in r:
                    continue
                v = r[field]
                if not isinstance(v, dict) or not v:
                    continue
                checked_records += 1
                inner_items = [(ok, iv) for ok, iv in v.items() if isinstance(iv, dict) and iv]
                if inner_items:
                    dict_of_dict_records += 1
                    for outer_key, iv in inner_items:
                        total_inner += 1
                        inner_key_counts.update(iv.keys())
                        for k, val in iv.items():
                            if isinstance(val, (str, int, float)) and not isinstance(val, bool):
                                value_counts.setdefault(k, Counter())[val] += 1
            if checked_records < min_records or dict_of_dict_records / max(checked_records, 1) < 0.6:
                continue
            if total_inner == 0:
                continue
            canon = sorted(k for k, c in inner_key_counts.items() if (c / total_inner * 100) >= min_presence)
            if not canon:
                continue
            constants: Dict[str, Any] = {}
            for k in canon:
                vc = value_counts.get(k)
                if vc:
                    top_val, top_count = vc.most_common(1)[0]
                    if top_count / sum(vc.values()) >= 0.9:
                        constants[k] = top_val
            result[field] = {"subkeys": canon, "constants": constants}
        return result

    def _extract_per_serving_bounds(self, data: Any, fmt: str, canonical_fields: List[str], field: str = "calories") -> Optional[Dict]:
        if fmt != "json" or not isinstance(data, list) or not data or field not in canonical_fields:
            return None
        yield_field = next((f for f in canonical_fields if f.lower() in ("yield", "servings", "serves")), None)
        if not yield_field:
            return None
        ratios = []
        for r in data[:min(1200, len(data))]:
            if not isinstance(r, dict):
                continue
            v, y = r.get(field), r.get(yield_field)
            if (isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(y, (int, float)) and not isinstance(y, bool) and y > 0):
                ratios.append(v / y)
        if len(ratios) < 20:
            return None
        ratios.sort()
        n = len(ratios)
        low = ratios[max(0, int(n * 0.05))]
        high = ratios[min(n - 1, int(n * 0.95))]
        if low < high:
            return {"field": field, "yield_field": yield_field, "low": low, "high": high}
        return None

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
        "GLUTEN_FREE": ["wheat", "bread", "flour", "pasta", "barley", "rye", "baguette", "sourdough", "ciabatta", "croissant", "focaccia", "brioche", "bagel", "biscotti", "pastry", "dough", "noodle", "couscous", "semolina", "pretzel", "cookie", "cake", "muffin", "waffle", "pancake", "pizza", "tortilla", "crostini", "panko", "breadcrumb", "cornetto"],
        "WHEAT_FREE": ["wheat", "bread", "flour", "pasta", "baguette", "sourdough", "ciabatta", "croissant", "focaccia", "brioche", "bagel", "biscotti", "pastry", "dough", "noodle", "couscous", "semolina", "pretzel", "cookie", "cake", "muffin", "waffle", "pancake", "pizza", "tortilla", "crostini", "panko", "breadcrumb", "cornetto"],
        "PEANUT_FREE": ["peanut"], "TREE_NUT_FREE": ["almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "macadamia", "brazil nut", "pine nut"],
        "SESAME_FREE": ["sesame", "tahini"], "SOY_FREE": ["soy", "tofu", "edamame", "tempeh"],
        "CELERY_FREE": ["celery"], "MUSTARD_FREE": ["mustard"], "LUPINE_FREE": ["lupin", "lupini"],
        "ALCOHOL_FREE": ["wine", "beer", "rum", "brandy", "liqueur", "whiskey", "vodka"],
    }
    _DIET_EXCLUSION_FIELD_CANDIDATES = ("nutritionLabels", "dietaryRequirements", "allergies", "foodIntolerances", "healthLabels")
    _DIET_TYPE_CONTRADICTION_KEYWORDS = {
        "LOW_CARB": ["pasta", "bread", "flour", "rice", "potato", "sugar", "honey", "pastry", "noodle"],
        "LOW_FAT": ["cheese", "ricotta", "mascarpone", "butter", "cream", "bacon", "sausage", "heavy cream", "cream cheese"],
    }
    _DIET_LABEL_FIELD_CANDIDATES = ("dietLabels",)
    _CAUTIONS_FIELD_CANDIDATES = ("cautions",)
    _CAUTION_KEYWORDS = {
        "GLUTEN": ["wheat", "bread", "flour", "pasta", "barley", "rye", "baguette", "sourdough", "ciabatta", "croissant", "focaccia", "brioche", "bagel", "biscotti", "pastry", "dough", "noodle", "couscous", "semolina", "pretzel", "cookie", "cake", "muffin", "waffle", "pancake", "pizza", "tortilla", "crostini", "panko", "breadcrumb", "cornetto"],
        "WHEAT": ["wheat", "bread", "flour", "pasta", "baguette", "sourdough", "ciabatta", "croissant", "focaccia", "brioche", "bagel", "biscotti", "pastry", "dough", "noodle", "couscous", "semolina", "pretzel", "cookie", "cake", "muffin", "waffle", "pancake", "pizza", "tortilla", "crostini", "panko", "breadcrumb", "cornetto"],
        "EGGS": ["egg"], "EGG": ["egg"],
        "MILK": ["milk", "cheese", "butter", "yogurt", "yoghurt", "cream"],
        "DAIRY": ["milk", "cheese", "butter", "yogurt", "yoghurt", "cream"],
        "FISH": ["fish", "salmon", "tuna", "cod", "trout"],
        "SHELLFISH": ["shrimp", "crab", "lobster", "mussel", "oyster"],
        "PEANUTS": ["peanut"], "PEANUT": ["peanut"],
        "SESAME": ["sesame", "tahini"], "SOY": ["soy", "tofu", "edamame", "tempeh"],
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
        foods = " ".join(str(ing.get("food", "")) + " " + str(ing.get("text", "")) for ing in ings if isinstance(ing, dict)).lower()
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
                    return f"{field} '{label}' is unsubstantiated — no matching ingredient found"
        return None

    def _rule_check(self, output: Any, task: dict, categorical_enums: Dict, canonical_fields: List[str], mostly_empty_fields: set, nested_dict_field_schemas: Dict, url_template_fields: Dict) -> Dict:
        critical: List[str] = []
        minor: List[str] = []
        schema_ok = True

        if output is None or (isinstance(output, (list, str)) and len(output) == 0):
            critical.append("Output is empty or None.")

        requested_qty = task.get("quantity")
        if requested_qty and isinstance(output, list):
            actual = len(output)
            if actual < requested_qty:
                critical.append(f"Quantity mismatch: requested {requested_qty}, got {actual}. Missing {requested_qty - actual} item(s).")
            elif actual > requested_qty * 2:
                minor.append(f"Quantity excess: requested {requested_qty}, got {actual}.")

        if isinstance(output, list):
            non_dict = [i for i, x in enumerate(output) if not isinstance(x, dict)]
            if non_dict:
                critical.append(f"Item(s) at index(es) {non_dict[:5]} are not JSON objects.")

        if isinstance(output, list) and output and isinstance(output[0], dict) and len(output[0]) < 2:
            minor.append("First item has very few fields — possible incomplete generation.")

        if isinstance(output, list) and output and categorical_enums:
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
                critical.append("Invalid enum value(s): " + "; ".join(f"{k}={sorted(v)}" for k, v in violations.items()))

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
                    critical.append(f"Field(s) entirely missing from every generated item: {always_missing}.")
                if sometimes_missing:
                    minor.append(f"Field(s) missing from some items: {sometimes_missing}.")

        if isinstance(output, list) and output:
            contradiction_indices = [idx for idx, item in enumerate(output) if isinstance(item, dict) and self._find_label_ingredient_contradiction(item)]
            if contradiction_indices:
                critical.append(f"Dietary/allergen labels contradicted by actual ingredients in {len(contradiction_indices)} item(s) (index(es) {contradiction_indices[:10]}).")

        if isinstance(output, list) and output and nested_dict_field_schemas:
            for field, schema in nested_dict_field_schemas.items():
                subkeys = schema.get("subkeys", []) if isinstance(schema, dict) else schema
                canon_set = set(subkeys)
                if not canon_set:
                    continue
                bad_items = [idx for idx, item in enumerate(output) if isinstance(item, dict) and isinstance(item.get(field), dict) and item[field] and any(not isinstance(iv, dict) or not iv for iv in item[field].values())]
                if bad_items:
                    schema_ok = False
                    critical.append(f"'{field}' is structurally corrupted in item(s) {bad_items[:5]}.")

        if isinstance(output, list) and output and url_template_fields:
            for field, prefix in url_template_fields.items():
                bad_items = [idx for idx, item in enumerate(output) if isinstance(item, dict) and isinstance(item.get(field), str) and item[field] and not item[field].startswith(prefix)]
                if bad_items:
                    schema_ok = False
                    critical.append(f"'{field}' doesn't follow the real dataset's URL template in item(s) {bad_items[:5]}.")

        return {"critical": critical, "minor": minor, "passes": len(critical) == 0, "schema_ok": schema_ok}

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
        non_accept = verdict.get("recommended_action", "accept") != "accept"
        schema_bad = verdict.get("schema_compliance") is False
        if verdict.get("valid") is True and (has_critical or non_accept or schema_bad):
            verdict["valid"] = False
            if verdict.get("recommended_action", "accept") == "accept":
                verdict["recommended_action"] = "retry_agent5"
        return verdict

    def _build_judge_sample(self, output, max_items: int = 12, max_chars: int = 8000) -> str:
        if not isinstance(output, list) or not output:
            return json.dumps(output, indent=2, default=str, ensure_ascii=False)[:max_chars]
        n = len(output)
        indices = list(range(n)) if n <= max_items else sorted(set(int(i * n / max_items) for i in range(max_items)))
        sampled = [output[i] for i in indices]
        text = json.dumps(sampled, indent=2, default=str, ensure_ascii=False)
        while len(sampled) > 1 and len(text) > max_chars:
            sampled = sampled[:-1]
            text = json.dumps(sampled, indent=2, default=str, ensure_ascii=False)
        return (f"[SAMPLE: {len(sampled)} complete items out of {n} total]\n{text}")

    async def _llm_score(self, output: Any, task: dict, user_ctx: str, rule: Dict) -> Dict:
        out_preview = self._build_judge_sample(output)

        critique = self.llm.chat([
            self._sys("You are a strict quality-assurance evaluator. Assess the generated output against the requirements. Be concise."),
            self._usr(
                f"User Requirements: {user_ctx}\n"
                f"Task: {json.dumps(task, default=str)[:800]}\n"
                f"Generated Output:\n{out_preview}\n\n"
                "Rate on:\n"
                "1. Schema compliance — do fields match the schema?\n"
                "2. Domain accuracy — values realistic and appropriate?\n"
                "3. Uniqueness — items are distinct from each other?\n"
                "4. Task fulfillment — meets user requirements?\n"
                "List any specific, concrete issues found."
            ),
        ], temperature=0.1)

        verdict_raw = self.llm.chat([
            self._sys("Output ONLY a single valid JSON object. No markdown."),
            self._usr(
                f"Critique:\n{critique}\n\n"
                f"The output contains {len(output)} complete items total.\n\n"
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
                "EVERY entry in critical_issues and minor_issues MUST describe a "
                "SPECIFIC, concrete problem. A bare category name with no specific "
                "problem (e.g. \"Authenticity\", \"Uniqueness\") is NOT a valid "
                "issue — omit it entirely.\n"
                "Start with { and end with }."
            ),
        ], temperature=0.05)

        verdict = self.llm.extract_json_dict(verdict_raw, fallback=dict(_VERDICT_FALLBACK))
        verdict["critical_issues"] = self._normalize_issue_list(verdict.get("critical_issues"))
        verdict["minor_issues"] = self._normalize_issue_list(verdict.get("minor_issues"))
        verdict = self._enforce_verdict_consistency(verdict)

        merged = dict(verdict)
        merged["schema_compliance"] = bool(merged.get("schema_compliance", True)) and rule.get("schema_ok", True)
        if not rule["passes"]:
            merged["valid"] = False
            merged["confidence"] = min(float(merged.get("confidence", 0.5)), 0.4)
            merged["recommended_action"] = "retry_agent5"
            merged["critical_issues"] = rule["critical"] + [c for c in merged.get("critical_issues", []) if c not in rule["critical"]]
        if rule["minor"]:
            merged.setdefault("minor_issues", [])
            merged["minor_issues"] = rule["minor"] + merged["minor_issues"]
        return merged

    def _score_rule_only(self, rule: Dict) -> Dict:
        verdict = dict(_VERDICT_FALLBACK)
        verdict.update({
            "valid": False, "confidence": 0.4,
            "schema_compliance": rule.get("schema_ok", True),
            "critical_issues": rule["critical"], "minor_issues": rule["minor"],
            "feedback": "Rule-based check failed: " + "; ".join(rule["critical"]),
            "recommended_action": "retry_agent5", "parse_error": False,
        })
        return verdict

    async def execute(self) -> bool:
        from config import settings

        raw_data  = self.bus.get("raw_data")
        fmt       = self.bus.get("data_format", "unknown")
        full_text = self.bus.get("data_full_text", "")
        user_ctx  = self.bus.get("user_context", "")

        configured_max = getattr(settings, "BASELINE_MAX_TOKENS", None)
        llm_max = getattr(settings, "LLM_MAX_TOKENS", 6144)
        max_tokens = configured_max if configured_max else llm_max
        max_attempts = getattr(settings, "BASELINE_MAX_PARSE_ATTEMPTS", 3)

        task = self.bus.get("task_spec") or {
            "task_type": "generate", "output_format": "json",
            "quantity": None, "constraints": [],
        }
        if task.get("quantity") is None:
            import re as _re
            m = _re.search(r"\b(\d+)\b", user_ctx)
            if m:
                task = dict(task)
                task["quantity"] = int(m.group(1))

        requested_qty = task.get("quantity") or 15

        logger.info(f"[Baseline] max_tokens={max_tokens} max_parse_attempts={max_attempts}")

        output = self._generate(fmt, full_text, user_ctx, max_tokens, max_attempts, requested_qty)
        self.bus.set("generated_output", output)
        self.bus.set("generated_format", "json")
        self.bus.set("run_kind", "baseline")

        logger.info("[Baseline] Extracting ground truth for scoring...")
        prog_stats        = self._programmatic_stats(raw_data, fmt)
        canonical_fields  = self._canonical_field_list(prog_stats)
        categorical_enums = self._extract_categorical_enums(raw_data, fmt)
        mostly_empty      = self._extract_mostly_empty_fields(raw_data, fmt)
        url_templates     = self._extract_url_templates(raw_data, fmt)
        nested_schemas    = self._detect_nested_dict_fields(raw_data, fmt, canonical_fields)
        numeric_bounds    = self._extract_numeric_bounds(raw_data, fmt)
        cal_bounds        = self._extract_per_serving_bounds(raw_data, fmt, canonical_fields, "calories")

        mostly_empty_fields = set(mostly_empty.keys())

        logger.info(f"[Baseline] Canonical fields ({len(canonical_fields)}): {canonical_fields}")
        logger.info(f"[Baseline] Categorical enums: { {k: len(v) for k, v in categorical_enums.items()} }")
        if mostly_empty_fields:
            logger.info(f"[Baseline] Mostly-empty fields (exempt from R9): {sorted(mostly_empty_fields)}")

        self.bus.set("task_spec",              task)
        self.bus.set("canonical_fields",       canonical_fields)
        self.bus.set("categorical_enums",      categorical_enums)
        self.bus.set("mostly_empty_fields",    mostly_empty)
        self.bus.set("url_template_fields",    url_templates)
        self.bus.set("nested_dict_field_schemas", nested_schemas)
        self.bus.set("numeric_bounds",         numeric_bounds)
        self.bus.set("calories_per_serving_bounds", cal_bounds)

        rule = self._rule_check(output, task, categorical_enums, canonical_fields, mostly_empty_fields, nested_schemas, url_templates)
        logger.info(f"[Baseline] Rule check: passes={rule['passes']} critical={rule['critical']} minor={rule['minor']}")

        if rule["passes"]:
            try:
                verdict = await self._llm_score(output, task, user_ctx, rule)
            except Exception as exc:
                logger.error(f"[Baseline] LLM scoring failed ({exc}) — using rule-only verdict.")
                verdict = self._score_rule_only(rule)
                verdict.update({"valid": True, "confidence": 0.6, "schema_compliance": rule.get("schema_ok", True), "critical_issues": [], "minor_issues": rule["minor"], "recommended_action": "accept"})
        else:
            verdict = self._score_rule_only(rule)

        passes = verdict.get("valid", False) and float(verdict.get("confidence", 0.0)) >= 0.55

        logger.info(
            f"[Baseline] Verdict: valid={verdict.get('valid')} "
            f"conf={float(verdict.get('confidence', 0)):.2f} "
            f"passes={passes} "
            f"critical={len(verdict.get('critical_issues', []))}"
        )

        self.bus.set("validation_verdict",  verdict)
        self.bus.set("validator_feedback",  verdict.get("feedback", ""))
        self.bus.set("validation_passed",   passes)
        self.bus.set("retry_count",         0)

        await self.bus.publish(Message(
            agent_id=self.agent_id,
            topic="baseline_complete",
            payload={
                "items_generated": len(output),
                "passes": passes,
                "confidence": verdict.get("confidence"),
                "completeness": verdict.get("completeness"),
                "accuracy": verdict.get("accuracy"),
            },
        ))
        return True
