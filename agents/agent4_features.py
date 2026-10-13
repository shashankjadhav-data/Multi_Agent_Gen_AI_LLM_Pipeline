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

_FEATURES_FALLBACK = {
    "domain": "general",
    "key_patterns": ["Follow the input schema exactly."],
    "field_constraints": {},
    "value_ranges": {},
    "categorical_values": {},
    "generation_guidelines": [
        "Follow the input schema exactly.",
        "Generate realistic and diverse values.",
        "Ensure each item is unique.",
    ],
    "example_field_values": {},
    "avoid_patterns": ["Duplicating existing records."],
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

class Agent4FeatureExtractor(BaseAgent):

    def __init__(self, bus: MessageBus):
        super().__init__("agent4", "Agent 4 · Feature Extractor (RAG)", bus)
        self._embedder  = None
        self._index     = None
        self._indexed   : List[str] = []

    def _programmatic_stats(self, data: Any, fmt: str) -> Dict:
        stats: Dict = {"format": fmt, "record_count": 0, "fields": {}}

        if fmt == "json" and isinstance(data, list):
            stats["record_count"] = len(data)
            sample = data[: min(300, len(data))]
            field_info: Dict = {}

            for item in sample:
                if not isinstance(item, dict):
                    continue
                for key, val in item.items():
                    if key not in field_info:
                        field_info[key] = {
                            "count": 0,
                            "types": set(),
                            "num_vals": [],
                            "str_vals": [],
                        }
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
                fs: Dict = {
                    "presence_pct": round(fi["count"] / n * 100, 1),
                    "types": list(fi["types"]),
                }
                if fi["num_vals"]:
                    v = fi["num_vals"]
                    fs["num_min"]  = round(min(v), 4)
                    fs["num_max"]  = round(max(v), 4)
                    fs["num_mean"] = round(statistics.mean(v), 4)
                if fi["str_vals"]:
                    uniq = list(dict.fromkeys(fi["str_vals"]))[:6]
                    fs["sample_values"] = uniq
                stats["fields"][key] = fs

        elif fmt in ("csv", "excel"):
            stats["record_count"] = len(data)
            for col in data.columns:
                fs = {"presence_pct": round(data[col].notna().mean() * 100, 1)}
                if data[col].dtype in ("int64", "float64"):
                    fs["num_min"]  = float(data[col].min())
                    fs["num_max"]  = float(data[col].max())
                    fs["num_mean"] = float(data[col].mean())
                else:
                    fs["sample_values"] = data[col].dropna().unique()[:6].tolist()
                stats["fields"][col] = fs

        return stats

    def _canonical_field_list(self, prog_stats: Dict, min_presence: float = 90.0) -> List[str]:
        return sorted(
            k for k, v in prog_stats.get("fields", {}).items()
            if v.get("presence_pct", 0) >= min_presence
        )

    def _extract_categorical_enums(
        self, data: Any, fmt: str, max_records: int = 1200
    ) -> Dict[str, Dict[str, int]]:
        if fmt != "json" or not isinstance(data, list) or not data:
            return {}

        sample = data[: min(max_records, len(data))]
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

    def _extract_mostly_empty_fields(
        self, data: Any, fmt: str, threshold: float = 0.9, max_records: int = 1200
    ) -> Dict[str, Any]:
        if fmt != "json" or not isinstance(data, list) or not data:
            return {}
        sample = data[: min(max_records, len(data))]
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

        result = {}
        for key, fc in field_counts.items():
            if fc["total"] == 0:
                continue
            fraction = fc["empty"] / fc["total"]
            if fraction >= threshold:
                result[key] = fc["empty_value"] if fc["empty_value"] is not None else ""
        return result

    def _extract_url_templates(
        self, data: Any, fmt: str, min_fraction: float = 0.9, min_prefix_len: int = 30,
        max_records: int = 1200,
    ) -> Dict[str, str]:
        if fmt != "json" or not isinstance(data, list) or not data:
            return {}
        sample = data[: min(max_records, len(data))]
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
            matching_untrimmed = sum(1 for v in vals if v.startswith(prefix))
            if not (
                len(prefix) >= min_prefix_len
                and matching_untrimmed / len(vals) >= min_fraction
            ) and "/" in prefix:
                prefix = prefix[: prefix.rfind("/") + 1]
            matching = sum(1 for v in vals if v.startswith(prefix))
            if len(prefix) >= min_prefix_len and matching / len(vals) >= min_fraction:
                templates[key] = prefix
        return templates

    def _extract_numeric_bounds(
        self, data: Any, fmt: str, low_pct: float = 1, high_pct: float = 99,
        max_records: int = 1200,
    ) -> Dict[str, Dict[str, float]]:
        if fmt != "json" or not isinstance(data, list) or not data:
            return {}
        sample = data[: min(max_records, len(data))]
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
            low = vals_sorted[max(0, int(n * low_pct / 100))]
            high = vals_sorted[min(n - 1, int(n * high_pct / 100))]
            if low < high:
                bounds[key] = {"low": low, "high": high}
        return bounds

    def _extract_ingredient_conventions(
        self, data: Any, fmt: str, array_field: str = "ingredients", max_records: int = 1200
    ) -> Dict[str, Any]:
        if fmt != "json" or not isinstance(data, list) or not data:
            return {}

        measures: List[str] = []
        mirror_count, total = 0, 0
        for item in data[: min(max_records, len(data))]:
            if not isinstance(item, dict):
                continue
            for ing in item.get(array_field, []) or []:
                if not isinstance(ing, dict):
                    continue
                m = ing.get("measure")
                if isinstance(m, str) and m:
                    measures.append(m)
                q, w = ing.get("quantity"), ing.get("weight")
                try:
                    qf, wf = float(q), float(w)
                    total += 1
                    if abs(qf - wf) < 0.01:
                        mirror_count += 1
                except (TypeError, ValueError):
                    pass

        result: Dict[str, Any] = {}
        if measures:
            top_measure, top_count = Counter(measures).most_common(1)[0]
            if top_count / len(measures) >= 0.95:
                result["dominant_measure"] = top_measure
        if total >= 20 and mirror_count / total >= 0.95:
            result["weight_mirrors_quantity"] = True
        return result

    @staticmethod
    def _normalize_food_name(name: str) -> str:
        name = (name or "").lower().strip()
        idx = name.find("(")
        if idx > 0:
            name = name[:idx].strip()
        if name.endswith("s") and len(name) > 3:
            name = name[:-1]
        return name

    def _extract_ingredient_weight_lookup(
        self, data: Any, fmt: str, array_field: str = "ingredients",
        min_samples: int = 3, max_records: int = 1200,
    ) -> Dict[str, float]:
        if fmt != "json" or not isinstance(data, list) or not data:
            return {}

        food_weights: Dict[str, list] = {}
        for item in data[: min(max_records, len(data))]:
            if not isinstance(item, dict):
                continue
            for ing in item.get(array_field, []) or []:
                if not isinstance(ing, dict):
                    continue
                food = self._normalize_food_name(ing.get("food") or ing.get("text") or "")
                if not food:
                    continue
                try:
                    w = float(ing.get("weight"))
                    if w > 0:
                        food_weights.setdefault(food, []).append(w)
                except (TypeError, ValueError):
                    pass

        return {
            food: statistics.median(weights)
            for food, weights in food_weights.items()
            if len(weights) >= min_samples
        }

    def _extract_ingredient_energy_ratio_lookup(
        self, data: Any, fmt: str, array_field: str = "ingredients",
        min_samples: int = 3, max_records: int = 1200,
    ) -> tuple:
        if fmt != "json" or not isinstance(data, list) or not data:
            return None, {}
        sample = data[: min(max_records, len(data))]

        key_counts: Counter = Counter()
        for item in sample[:200]:
            if not isinstance(item, dict):
                continue
            for ing in item.get(array_field, []) or []:
                if isinstance(ing, dict):
                    key_counts.update(
                        k for k in ing.keys()
                        if "energy" in k.lower() or "kcal" in k.lower()
                    )
        if not key_counts:
            return None, {}
        energy_field = key_counts.most_common(1)[0][0]

        ratios: Dict[str, list] = {}
        for item in sample:
            if not isinstance(item, dict):
                continue
            for ing in item.get(array_field, []) or []:
                if not isinstance(ing, dict):
                    continue
                food = self._normalize_food_name(ing.get("food") or ing.get("text") or "")
                if not food:
                    continue
                try:
                    w = float(ing.get("weight"))
                    e = float(ing.get(energy_field))
                    if w > 0 and e > 0:
                        ratios.setdefault(food, []).append(e / w)
                except (TypeError, ValueError):
                    pass

        lookup = {
            food: statistics.median(rs)
            for food, rs in ratios.items()
            if len(rs) >= min_samples
        }
        return energy_field, lookup

    @staticmethod
    def _normalize_nutrient_name(name: str) -> str:
        n = (name or "").strip().lower()
        n = re.sub(r"\s*\([^)]*\)\s*$", "", n).strip()
        if n.endswith("s") and len(n) > 3:
            n = n[:-1]
        return n

    def _extract_nutrient_label_lookup(
        self, data: Any, fmt: str, array_field: str = "nutrients",
        min_samples: int = 3, min_fraction: float = 0.9, max_records: int = 1200,
    ) -> Dict[str, str]:
        if fmt != "json" or not isinstance(data, list) or not data:
            return {}
        sample = data[: min(max_records, len(data))]

        name_labels: Dict[str, Counter] = {}
        for item in sample:
            if not isinstance(item, dict):
                continue
            for n in item.get(array_field, []) or []:
                if not isinstance(n, dict):
                    continue
                name, label = n.get("name"), n.get("label")
                if not (isinstance(name, str) and name and isinstance(label, str) and label):
                    continue
                key = name.strip().lower()
                name_labels.setdefault(key, Counter())[label] += 1

        lookup: Dict[str, str] = {}
        simplified_best: Dict[str, tuple] = {}
        for name, counts in name_labels.items():
            total = sum(counts.values())
            if total < min_samples:
                continue
            top_label, top_count = counts.most_common(1)[0]
            if top_count / total < min_fraction:
                continue
            lookup[name] = top_label

            simplified = self._normalize_nutrient_name(name)
            if simplified and simplified != name:
                prev = simplified_best.get(simplified)
                if prev is None or total > prev[1]:
                    simplified_best[simplified] = (top_label, total)

        for simplified, (label, _count) in simplified_best.items():
            lookup.setdefault(simplified, label)

        return lookup

    def _extract_ingredient_canonical_fields(
        self, data: Any, fmt: str, array_field: str = "ingredients",
        min_presence: float = 70.0, max_records: int = 1200,
    ) -> List[str]:
        if fmt != "json" or not isinstance(data, list) or not data:
            return []
        sample = data[: min(max_records, len(data))]
        key_counts: Counter = Counter()
        total_ings = 0
        for item in sample:
            if not isinstance(item, dict):
                continue
            for ing in item.get(array_field, []) or []:
                if not isinstance(ing, dict):
                    continue
                total_ings += 1
                key_counts.update(ing.keys())
        if total_ings == 0:
            return []
        return sorted(
            k for k, c in key_counts.items()
            if (c / total_ings * 100) >= min_presence
        )

    def _extract_ingredient_field_types(
        self, data: Any, fmt: str, array_field: str = "ingredients", max_records: int = 1200,
    ) -> Dict[str, str]:
        if fmt != "json" or not isinstance(data, list) or not data:
            return {}
        sample = data[: min(max_records, len(data))]
        counts: Dict[str, Counter] = {}
        for item in sample:
            if not isinstance(item, dict):
                continue
            for ing in item.get(array_field, []) or []:
                if not isinstance(ing, dict):
                    continue
                for k, v in ing.items():
                    if isinstance(v, str):
                        counts.setdefault(k, Counter())["str"] += 1
                    elif isinstance(v, (int, float)) and not isinstance(v, bool):
                        counts.setdefault(k, Counter())["number"] += 1
        return {
            k: c.most_common(1)[0][0]
            for k, c in counts.items()
            if sum(c.values()) >= 5
        }

    def _detect_total_weight_field(
        self, data: Any, canonical_fields: List[str], fmt: str,
        array_field: str = "ingredients", min_samples: int = 10, max_records: int = 200,
    ) -> Optional[str]:
        if fmt != "json" or not isinstance(data, list) or not data:
            return None
        candidates = [f for f in canonical_fields if "weight" in f.lower()]
        if not candidates:
            return None
        sample = data[: min(max_records, len(data))]
        for field in candidates:
            diffs = []
            for r in sample:
                if not isinstance(r, dict):
                    continue
                tw = r.get(field)
                if not isinstance(tw, (int, float)) or isinstance(tw, bool) or tw <= 0:
                    continue
                ings = r.get(array_field, []) or []
                weights = [
                    ing.get("weight") for ing in ings
                    if isinstance(ing, dict) and isinstance(ing.get("weight"), (int, float))
                    and not isinstance(ing.get("weight"), bool)
                ]
                if weights:
                    diffs.append(abs(tw - sum(weights)) / tw)
            if len(diffs) >= min_samples and (sum(diffs) / len(diffs)) < 0.15:
                return field
        return None

    def _detect_nested_dict_fields(
        self, data: Any, fmt: str, canonical_fields: List[str],
        min_presence: float = 60.0, min_records: int = 20, max_records: int = 1200,
        min_key_samples: int = 5,
    ) -> Dict[str, Dict[str, Any]]:
        if fmt != "json" or not isinstance(data, list) or not data:
            return {}
        sample = data[: min(max_records, len(data))]
        result: Dict[str, Dict[str, Any]] = {}
        for field in canonical_fields:
            inner_key_counts: Counter = Counter()
            value_counts: Dict[str, Counter] = {}
            per_key_value_counts: Dict[str, Dict[str, Counter]] = {}
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
                                per_key_value_counts.setdefault(outer_key, {}).setdefault(
                                    k, Counter()
                                )[val] += 1
            if checked_records < min_records or dict_of_dict_records / max(checked_records, 1) < 0.6:
                continue
            if total_inner == 0:
                continue
            canon = sorted(
                k for k, c in inner_key_counts.items()
                if (c / total_inner * 100) >= min_presence
            )
            if not canon:
                continue
            constants: Dict[str, Any] = {}
            for k in canon:
                vc = value_counts.get(k)
                if vc:
                    top_val, top_count = vc.most_common(1)[0]
                    if top_count / sum(vc.values()) >= 0.9:
                        constants[k] = top_val
            per_key_constants: Dict[str, Dict[str, Any]] = {}
            per_key_numeric_fallback: Dict[str, Dict[str, float]] = {}
            for outer_key, subkey_counts in per_key_value_counts.items():
                for k, vc in subkey_counts.items():
                    if k in constants:
                        continue
                    n = sum(vc.values())
                    if n < min_key_samples:
                        continue
                    top_val, top_count = vc.most_common(1)[0]
                    if top_count / n >= 0.9:
                        per_key_constants.setdefault(outer_key, {})[k] = top_val
                    else:
                        numeric_vals = [
                            v for v, cnt in vc.items()
                            for _ in range(cnt)
                            if isinstance(v, (int, float)) and not isinstance(v, bool)
                        ]
                        if len(numeric_vals) >= min_key_samples:
                            numeric_vals.sort()
                            mid = len(numeric_vals) // 2
                            median = (
                                numeric_vals[mid] if len(numeric_vals) % 2
                                else (numeric_vals[mid - 1] + numeric_vals[mid]) / 2
                            )
                            per_key_numeric_fallback.setdefault(outer_key, {})[k] = median
            result[field] = {
                "subkeys": canon, "constants": constants,
                "per_key_constants": per_key_constants,
                "per_key_numeric_fallback": per_key_numeric_fallback,
            }
        return result

    def _extract_per_serving_bounds(
        self, data: Any, fmt: str, canonical_fields: List[str], field: str = "calories",
        yield_field_candidates: tuple = ("yield", "servings", "serves"),
        low_pct: float = 5, high_pct: float = 95, min_samples: int = 20, max_records: int = 1200,
    ) -> Optional[dict]:
        if fmt != "json" or not isinstance(data, list) or not data or field not in canonical_fields:
            return None
        yield_field = next(
            (f for f in canonical_fields if f.lower() in yield_field_candidates), None
        )
        if not yield_field:
            return None
        sample = data[: min(max_records, len(data))]
        ratios = []
        for r in sample:
            if not isinstance(r, dict):
                continue
            v, y = r.get(field), r.get(yield_field)
            if (
                isinstance(v, (int, float)) and not isinstance(v, bool)
                and isinstance(y, (int, float)) and not isinstance(y, bool) and y > 0
            ):
                ratios.append(v / y)
        if len(ratios) < min_samples:
            return None
        ratios.sort()
        n = len(ratios)
        low = ratios[max(0, int(n * low_pct / 100))]
        high = ratios[min(n - 1, int(n * high_pct / 100))]
        if low < high:
            return {"field": field, "yield_field": yield_field, "low": low, "high": high}
        return None

    def _get_embedder(self):
        if self._embedder is None:
            import os
            from sentence_transformers import SentenceTransformer
            from config import settings
            logger.info("[A4] Loading sentence-transformer…")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            try:
                self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
            except Exception as exc:
                logger.warning(
                    f"[A4] Offline load failed ({exc}) — model likely not "
                    "cached yet; retrying with network access."
                )
                os.environ.pop("HF_HUB_OFFLINE", None)
                self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self._embedder

    def _build_index(self, chunks: List[str]) -> None:
        import faiss
        from config import settings
        logger.info(f"[A4] Encoding {len(chunks)} chunks into FAISS index…")
        emb  = self._get_embedder()
        vecs = emb.encode(
            chunks,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= np.maximum(norms, 1e-10)
        idx  = faiss.IndexFlatIP(vecs.shape[1])
        idx.add(vecs)
        self._index   = idx
        self._indexed = list(chunks)
        logger.info(f"[A4] FAISS ready: {idx.ntotal} vectors dim={vecs.shape[1]}")

    def _retrieve(self, query: str, chunks: List[str], k: int = 8) -> List[str]:
        if self._index is None or self._indexed != chunks:
            self._build_index(chunks)
        emb = self._get_embedder()
        qv  = emb.encode([query], convert_to_numpy=True).astype(np.float32)
        qv /= np.maximum(np.linalg.norm(qv), 1e-10)
        k   = min(k, len(chunks))
        _, idxs = self._index.search(qv, k)
        return [chunks[i] for i in idxs[0] if 0 <= i < len(chunks)]

    async def execute(self) -> bool:
        from config import settings

        raw_data  = self.bus.get("raw_data")
        fmt       = self.bus.get("data_format", "unknown")
        rag_chunks = self.bus.get("data_rag_chunks", [])
        examples  = self.bus.get("data_examples", [])
        task      = self.bus.get("task_spec", {})
        user_ctx  = self.bus.get("user_context", "")

        prog_stats = self._programmatic_stats(raw_data, fmt)
        logger.info(
            f"[A4] Programmatic stats: {prog_stats['record_count']} records, "
            f"{len(prog_stats['fields'])} fields."
        )

        canonical_fields  = self._canonical_field_list(prog_stats)
        categorical_enums = self._extract_categorical_enums(raw_data, fmt)
        logger.info(
            f"[A4] Canonical fields (programmatic): {len(canonical_fields)} "
            f"-> {canonical_fields}"
        )
        logger.info(
            f"[A4] Categorical enums (programmatic): "
            f"{ {k: len(v) for k, v in categorical_enums.items()} }"
        )

        mostly_empty_fields = self._extract_mostly_empty_fields(raw_data, fmt)
        url_template_fields = self._extract_url_templates(raw_data, fmt)
        numeric_bounds = self._extract_numeric_bounds(raw_data, fmt)
        ingredient_conventions = self._extract_ingredient_conventions(raw_data, fmt)
        ingredient_weight_lookup = self._extract_ingredient_weight_lookup(raw_data, fmt)
        ingredient_energy_field, ingredient_energy_lookup = self._extract_ingredient_energy_ratio_lookup(raw_data, fmt)
        nutrient_label_lookup = self._extract_nutrient_label_lookup(raw_data, fmt)
        ingredient_canonical_fields = self._extract_ingredient_canonical_fields(raw_data, fmt)
        ingredient_field_types = self._extract_ingredient_field_types(raw_data, fmt)
        total_weight_field = self._detect_total_weight_field(raw_data, canonical_fields, fmt)
        nested_dict_field_schemas = self._detect_nested_dict_fields(raw_data, fmt, canonical_fields)
        calories_per_serving_bounds = self._extract_per_serving_bounds(
            raw_data, fmt, canonical_fields, field="calories"
        )
        weight_per_serving_bounds = (
            self._extract_per_serving_bounds(
                raw_data, fmt, canonical_fields, field=total_weight_field
            )
            if total_weight_field else None
        )
        if mostly_empty_fields:
            logger.info(f"[A4] Mostly-empty fields (programmatic): {list(mostly_empty_fields.keys())}")
        if url_template_fields:
            logger.info(f"[A4] URL template fields (programmatic): {list(url_template_fields.keys())}")
        if numeric_bounds:
            logger.info(f"[A4] Numeric sanity bounds (programmatic): {numeric_bounds}")
        if ingredient_conventions:
            logger.info(f"[A4] Ingredient sub-field conventions (programmatic): {ingredient_conventions}")
        if ingredient_weight_lookup:
            logger.info(
                f"[A4] Ingredient weight lookup (programmatic): "
                f"{len(ingredient_weight_lookup)} known foods, e.g. "
                f"{dict(list(ingredient_weight_lookup.items())[:5])}"
            )
        if ingredient_energy_lookup:
            logger.info(
                f"[A4] Ingredient energy lookup (programmatic): field="
                f"'{ingredient_energy_field}' {len(ingredient_energy_lookup)} known "
                f"foods (kcal/gram), e.g. "
                f"{dict(list(ingredient_energy_lookup.items())[:5])}"
            )
        if nutrient_label_lookup:
            logger.info(
                f"[A4] Nutrient name->label lookup (programmatic): "
                f"{len(nutrient_label_lookup)} known nutrients, e.g. "
                f"{dict(list(nutrient_label_lookup.items())[:5])}"
            )
        if ingredient_canonical_fields:
            logger.info(
                f"[A4] Ingredient canonical sub-fields (programmatic): "
                f"{ingredient_canonical_fields}"
            )
        if ingredient_field_types:
            logger.info(
                f"[A4] Ingredient sub-field types (programmatic): {ingredient_field_types}"
            )
        if total_weight_field:
            logger.info(
                f"[A4] Total-weight field detected (programmatic): '{total_weight_field}' "
                f"correlates with sum(ingredient weights)"
            )
        if nested_dict_field_schemas:
            const_summary = {
                f: s.get("constants") for f, s in nested_dict_field_schemas.items() if s.get("constants")
            }
            logger.info(
                f"[A4] Nested dict-of-dicts field schemas (programmatic): "
                f"{ {f: s['subkeys'] for f, s in nested_dict_field_schemas.items()} }"
                + (f"  constant sub-values: {const_summary}" if const_summary else "")
            )
        if calories_per_serving_bounds:
            logger.info(
                f"[A4] Per-serving bounds (programmatic): {calories_per_serving_bounds}"
            )
        if weight_per_serving_bounds:
            logger.info(
                f"[A4] Per-serving bounds (programmatic): {weight_per_serving_bounds}"
            )

        query = " ".join(filter(None, [
            user_ctx,
            task.get("summary", ""),
            task.get("action", ""),
            " ".join(task.get("constraints", [])),
        ]))

        rag_relevant = self._retrieve(query, rag_chunks, k=settings.TOP_K_CHUNKS)
        logger.info(f"[A4] FAISS retrieved {len(rag_relevant)} relevant chunks.")

        important_fields = {
            k: v for k, v in prog_stats["fields"].items()
            if v.get("presence_pct", 0) >= 50
        }
        stats_summary = json.dumps(
            {
                "record_count": prog_stats["record_count"],
                "top_fields": important_fields,
            },
            indent=2, default=str
        )[:settings.MAX_LLM_CONTEXT_CHARS]

        def _abbreviate(obj, depth=0):
            if isinstance(obj, dict):
                return {k: _abbreviate(v, depth + 1) for k, v in obj.items()
                        if depth < 3}
            if isinstance(obj, list):
                sample = obj[:3]
                return [_abbreviate(x, depth + 1) for x in sample] + (
                    [f"... ({len(obj)} items total)"] if len(obj) > 3 else []
                )
            return obj

        abbrev_examples = [_abbreviate(e) for e in examples[:settings.MAX_EXAMPLES_IN_PROMPT]]
        examples_text = json.dumps(abbrev_examples, indent=2, default=str, ensure_ascii=False)
        examples_text = examples_text[:settings.MAX_LLM_CONTEXT_CHARS]

        feats_raw = self.llm.chat([
            self._sys(
                "You are a data analyst. You will receive dataset statistics and "
                "abbreviated examples. Your task is to ANALYSE the patterns and "
                "produce a JSON OBJECT describing those patterns.\n"
                "DO NOT reproduce the examples. DO NOT output a list of records.\n"
                "Output ONLY a single JSON object {{ ... }} with your analysis.\n"
                "No markdown fences. No explanation."
            ),
            self._usr(
                f"Task: {task.get('summary', user_ctx)}\n\n"
                f"=== DATASET STATISTICS (programmatic) ===\n{stats_summary}\n\n"
                f"=== 5 ABBREVIATED EXAMPLES (for reference only) ===\n{examples_text}\n\n"
                "Produce a single JSON analysis object with EXACTLY these keys:\n"
                "{\n"
                '  "domain": "<domain name>",\n'
                '  "key_patterns": ["<pattern observed in the data>"],\n'
                '  "field_constraints": {"<fieldName>": "<observed rule/constraint>"},\n'
                '  "value_ranges": {"<numericField>": {"min": 0, "max": 0, "typical": 0}},\n'
                '  "categorical_values": {"<field>": ["<common value 1>", "<common value 2>"]},\n'
                '  "generation_guidelines": ["<what to do when generating new records>"],\n'
                '  "example_field_values": {"<field>": "<realistic example value>"},\n'
                '  "avoid_patterns": ["<what to avoid in new records>"]\n'
                "}\n\n"
                "CRITICAL: Start your response with { and end with }. "
                "Do NOT output the example records. Output ONLY your analysis."
            ),
        ], temperature=0.1)

        features = self.llm.extract_json_dict(feats_raw, fallback=dict(_FEATURES_FALLBACK))

        features["_prog_stats"] = {
            "record_count": prog_stats["record_count"],
            "field_count": len(prog_stats["fields"]),
            "field_names": list(prog_stats["fields"].keys())[:30],
        }

        llm_categorical = features.get("categorical_values") or {}
        merged_categorical = dict(llm_categorical)
        for field, counts in categorical_enums.items():
            merged_categorical[field] = list(counts.keys())
        features["categorical_values"] = merged_categorical

        if "value_ranges" not in features or not features["value_ranges"]:
            features["value_ranges"] = {
                k: {
                    "min": v["num_min"],
                    "max": v["num_max"],
                    "typical": v["num_mean"],
                }
                for k, v in prog_stats["fields"].items()
                if "num_min" in v
            }

        logger.info(
            f"[A4] domain='{features.get('domain', '?')}' "
            f"patterns={len(features.get('key_patterns', []))} "
            f"guidelines={len(features.get('generation_guidelines', []))}"
        )

        self.bus.set("extracted_features", features)
        self.bus.set("prog_stats",         prog_stats)
        self.bus.set("rag_context",        "\n\n---\n\n".join(rag_relevant[:5]))
        self.bus.set("canonical_fields",   canonical_fields)
        self.bus.set("categorical_enums",  categorical_enums)
        self.bus.set("mostly_empty_fields", mostly_empty_fields)
        self.bus.set("url_template_fields", url_template_fields)
        self.bus.set("numeric_bounds",      numeric_bounds)
        self.bus.set("ingredient_conventions", ingredient_conventions)
        self.bus.set("ingredient_weight_lookup", ingredient_weight_lookup)
        self.bus.set("ingredient_energy_field", ingredient_energy_field)
        self.bus.set("ingredient_energy_lookup", ingredient_energy_lookup)
        self.bus.set("nutrient_label_lookup", nutrient_label_lookup)
        self.bus.set("ingredient_canonical_fields", ingredient_canonical_fields)
        self.bus.set("ingredient_field_types", ingredient_field_types)
        self.bus.set("total_weight_field", total_weight_field)
        self.bus.set("nested_dict_field_schemas", nested_dict_field_schemas)
        self.bus.set("calories_per_serving_bounds", calories_per_serving_bounds)
        self.bus.set("weight_per_serving_bounds", weight_per_serving_bounds)

        await self.bus.publish(Message(
            agent_id=self.agent_id, topic="features_extracted", payload=features
        ))
        return True
