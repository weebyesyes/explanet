# exo_infer.py
# --------------------------------------------
# Inference utilities for the NASA Exoplanet Classifier bundle (v4)
# - Validation (schema/sanity)
# - Prediction (two-stage, seed/fold avg, meta blender, Platt)
# - Optional metrics/curves if labels exist in the upload
#
# Usage (quick):
#   eng = InferenceEngine("/path/to/exo_bundle_v4")
#   ok, report = validate_df(df)         # gate before scoring
#   out = eng.predict_df(df, top_n=50)   # or threshold=0.5
#   out["preds"].head()
#   out["metrics"], out["curves"]
# --------------------------------------------

from __future__ import annotations
import os, io, json, sys, math, zipfile
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any, Set

import numpy as np
import pandas as pd
import joblib
import re

# ML libs (soft optional: if Cat/XGB not present, those families are skipped)
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, average_precision_score,
    f1_score, recall_score, brier_score_loss, confusion_matrix
)
from sklearn.linear_model import LogisticRegression


# =====================
# Shared constants (must match training)
# =====================
PLANET, CAND, FP = "Planet", "Candidate", "False Positive"
LABEL_ORDER = [FP, CAND, PLANET]
LABEL2ID = {lab:i for i,lab in enumerate(LABEL_ORDER)}
ID2LABEL = {i:lab for lab,i in LABEL2ID.items()}

KEPLER_MAP = {"CONFIRMED": PLANET, "CANDIDATE": CAND, "FALSE POSITIVE": FP}
TOI_MAP    = {"CP": PLANET, "KP": PLANET, "PC": CAND, "APC": CAND, "FP": FP, "FA": FP}
K2_MAP     = {"CONFIRMED": PLANET, "CANDIDATE": CAND, "FALSE POSITIVE": FP}

INTERNAL = {
    "id": "id",
    "star_id": "star_id",
    "mission": "mission",
    "label": "label",
    "period_days": "period_days",
    "duration_hours": "duration_hours",
    "depth_ppm": "depth_ppm",
    "planet_radius_re": "planet_radius_re",
    "stellar_teff_k": "stellar_teff_k",
    "stellar_logg_cgs": "stellar_logg_cgs",
    "stellar_radius_rsun": "stellar_radius_rsun",
    "mag_kepler": "mag_kepler",
    "mag_tess": "mag_tess",
    "insolation_earth": "insolation_earth",
    "eqt_k": "eqt_k",
    "snr_like": "snr_like",
}

# Candidate columns that look leaky in raw sources (we always strip them)
LEAKY_SUBSTRINGS = ["disposition","pdisposition","tfopwg","fpflag","score","disp_","_flag"]

# Validator notes (kept for UI parity; inference no longer defaults)
MISSION_FALLBACK_NOTE = "Unrecognized/empty mission type, defaulting to TESS"
KNOWN_MISSIONS = {"Kepler", "TESS", "K2"}
REQUIRED = [INTERNAL["id"], INTERNAL["mission"]]

NUM_HINTS = [
    INTERNAL["period_days"], INTERNAL["duration_hours"], INTERNAL["depth_ppm"],
    INTERNAL["planet_radius_re"], INTERNAL["stellar_teff_k"], INTERNAL["stellar_logg_cgs"],
    INTERNAL["stellar_radius_rsun"], INTERNAL["insolation_earth"], INTERNAL["eqt_k"],
    INTERNAL["snr_like"]
]

def _normalize(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# =====================
# Helper: robust CSV loader (also supports simple ZIP->first CSV)
# =====================
def load_any_csv(path_or_bytes) -> pd.DataFrame:
    if isinstance(path_or_bytes, (str, os.PathLike)) and str(path_or_bytes).lower().endswith(".zip"):
        with zipfile.ZipFile(path_or_bytes) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            with z.open(name) as f:
                return pd.read_csv(f, low_memory=False)
    if isinstance(path_or_bytes, (bytes, bytearray, io.BytesIO)):
        return pd.read_csv(io.BytesIO(path_or_bytes), low_memory=False)
    return pd.read_csv(path_or_bytes, low_memory=False)


# =====================
# VALIDATOR
# Returns (ok_for_scoring: bool, report: dict)
# - status: "Ready" | "Needs fixes" | "Cannot score"
# - issues: list of {severity, code, msg, ...}
# =====================
def _mission_status(raw: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (stringified, missing_mask, invalid_mask) for mission column."""
    mission_str = raw.astype(str).str.strip()
    lower = mission_str.str.lower()
    missing_mask = (
        raw.isna()
        | (mission_str == "")
        | lower.isin({"nan", "none", "null", "unknown"})
    )
    invalid_mask = ~(mission_str.isin(KNOWN_MISSIONS) | missing_mask)
    return mission_str, missing_mask, invalid_mask

def validate_df(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    issues = []
    # 0) trivial
    if df is None or len(df) == 0:
        return False, {
            "status": "Cannot score", "row_count": 0,
            "issues": [{"severity":"error","code":"empty","msg":"Empty file or no rows."}],
            "detected_columns": []
        }

    # 1) trim whitespace, standardize column names
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    cols = set(df.columns)

    # 2) required columns
    missing_required = [c for c in REQUIRED if c not in cols]
    if missing_required:
        issues.append({"severity":"error","code":"missing_required",
                       "msg": f"Missing required columns: {missing_required}"})

    # 3) mission values (if present) — validator remains *lenient* (warns)
    if INTERNAL["mission"] in cols:
        raw_mission = df[INTERNAL["mission"]]
        mission_str, missing_mask, invalid_mask = _mission_status(raw_mission)
        if invalid_mask.any():
            sample = mission_str[invalid_mask].value_counts().index.tolist()[:5]
            issues.append({
                "severity": "warn",
                "code": "mission_unrecognized",
                "msg": (
                    "Mission values found that are not part of the three missions we were "
                    f"trained on ({sorted(KNOWN_MISSIONS)}). Treating them as blank. "
                    f"Examples: {sample}."
                )
            })
        missing_like = missing_mask | invalid_mask
        if missing_like.any():
            missing_count = int(missing_like.sum())
            issues.append({
                "severity": "warn",
                "code": "mission_missing",
                "msg": (
                    "Mission is empty or unrecognized for "
                    f"{missing_count} rows. Defaulting those rows to the TESS pipeline for scoring (may impact predictions)."
                )
            })

    # 4) duplicates in id
    if INTERNAL["id"] in cols:
        dup = df[INTERNAL["id"]].astype(str).duplicated(keep=False)
        if dup.any():
            issues.append({"severity":"warn","code":"duplicates",
                           "msg": f"Duplicate id values in {int(dup.sum())} rows."})

    # 5) leaky columns
    leaky = [c for c in df.columns if any(s in c.lower() for s in LEAKY_SUBSTRINGS)]
    if leaky:
        issues.append({"severity":"warn","code":"leaky_columns",
                       "msg": f"Detected potential label-leak columns: {leaky}. They will be ignored."})

    # 6) numeric sanity (warnings only)
    num_cols_present = [c for c in NUM_HINTS if c in cols]
    out_of_range_msgs = []

    def _warn_range(col, cond, msg):
        if col in cols:
            bad = df[col].apply(pd.to_numeric, errors="coerce")
            m = cond(bad.fillna(0))
            if m.any():
                idx = m[m].index[:5].tolist()
                out_of_range_msgs.append(f"{col}: {msg} (examples rows {idx})")

    _warn_range(INTERNAL["duration_hours"], lambda s: s > 1000, "> 1000 hours")
    _warn_range(INTERNAL["depth_ppm"], lambda s: s > 2e5, "> 200,000 ppm")
    _warn_range(INTERNAL["planet_radius_re"], lambda s: s > 50, "> 50 R_earth")
    _warn_range(INTERNAL["stellar_teff_k"], lambda s: (s < 1500) | (s > 20000), "<1500 or >20000 K")
    if out_of_range_msgs:
        issues.append({"severity":"warn","code":"out_of_range",
                       "msg": "; ".join(out_of_range_msgs)})

    # 7) missing %
    missing_pct = {}
    for c in num_cols_present:
        s = pd.to_numeric(df[c], errors="coerce")
        miss = float((s.isna() | ~np.isfinite(s)).mean() * 100.0)
        if miss > 50:
            issues.append({"severity":"warn","code":"high_missing",
                           "msg": f"High missing rate in {c}: {miss:.1f}%."})
        missing_pct[c] = miss

    # final status
    if any(i["severity"] == "error" for i in issues):
        status = "Cannot score"
        ok = False
    elif issues:
        status = "Needs fixes"
        ok = True
    else:
        status = "Ready"
        ok = True

    return ok, {
        "status": status,
        "row_count": int(len(df)),
        "detected_columns": list(df.columns),
        "missing_pct": missing_pct,
        "issues": issues,
        "tips": [
            "Mission must be one of: Kepler, TESS, K2.",
            "Depth must be in ppm; 1% transit = 10,000 ppm.",
            "ID should be unique per signal."
        ]
    }


# =====================
# Preprocessor class (same name as training) so preprocess.pkl loads cleanly
# (Contains the original .transform logic, trimmed comments; we will rely on the
# pickled instance’s learned medians/clip/z params.)
# =====================
@dataclass
class HarmonizeSpec:
    src: Dict[str, str]
    label_map: Dict[str, str]

def _find_col(df, candidates):
    if df is None or df.shape[1] == 0:
        return None
    if isinstance(candidates, str):
        candidates = [candidates]
    normmap = {_normalize(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize(cand)
        if key in normmap:
            return normmap[key]
    for c in df.columns:  # loose contains
        if any(_normalize(cand) in _normalize(c) for cand in candidates):
            return c
    return None

class ExoPreprocessor:
    RAD_EARTH_PER_SUN = 109.076
    def __init__(self, seed=42, k2_trandep_unit="auto", feature_switches=None):
        self.seed = seed
        self.k2_trandep_unit = k2_trandep_unit
        self.fs = feature_switches or {}
        self.mission_medians: Dict[str,pd.Series] = {}
        self.numeric_features_: List[str] = []
        self.cat_features_: List[str] = [INTERNAL["mission"]]
        self.mission_clip_: Dict[str, Dict[str, Tuple[float,float]]] = {}
        self.mission_meanstd_: Dict[str, Dict[str, Tuple[float,float]]] = {}

    def _fix_k2_depth_units(self, s: pd.Series) -> pd.Series:
        # “auto” heuristic (matches training code path)
        med = float(np.nanmedian(s.dropna())) if s.notna().any() else np.nan
        if self.k2_trandep_unit == "ppm": return s
        if self.k2_trandep_unit == "percent": return s * 10000.0
        if not np.isfinite(med): return s
        return s*10000.0 if med < 10 else s

    def _harmonize_one(self, df: pd.DataFrame, spec: HarmonizeSpec, mission: str) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out[INTERNAL["mission"]] = mission
        # ids
        id_col = _find_col(df, spec.src.get("id")) if spec.src.get("id") else None
        out[INTERNAL["id"]] = (df[id_col] if id_col is not None else df.index).astype(str)
        star_col = _find_col(df, spec.src.get("star_id")) if spec.src.get("star_id") else None
        out[INTERNAL["star_id"]] = (df[star_col] if star_col is not None else out[INTERNAL["id"]]).astype(str)
        # label
        lab_col = _find_col(df, spec.src.get("label_candidates", []))
        if lab_col:
            out[INTERNAL["label"]] = df[lab_col].astype(str).str.strip().str.upper().map(spec.label_map)
        else:
            out[INTERNAL["label"]] = np.nan
        # physics
        resolved = {}
        for k in [
            "period_days","duration_hours","depth_ppm","planet_radius_re",
            "stellar_teff_k","stellar_logg_cgs","stellar_radius_rsun",
            "mag_kepler","mag_tess","insolation_earth","eqt_k","snr_like",
        ]:
            src_name = spec.src.get(k)
            use_col = _find_col(df, src_name) if src_name else None
            resolved[k] = use_col
            out[INTERNAL[k]] = pd.to_numeric(df[use_col], errors="coerce") if use_col else np.nan

        # pass-through error-widths if present (training supports these in jitter calc)
        for e in ["period_errw","duration_errw","depth_errw","prad_errw",
                  "teff_errw","logg_errw","srad_errw","insol_errw","eqt_errw"]:
            c = _find_col(df, e)
            out[e] = pd.to_numeric(df[c], errors="coerce") if c else np.nan

        # K2 unit fix
        if mission == "K2" and not pd.isna(out[INTERNAL["depth_ppm"]]).all():
            out[INTERNAL["depth_ppm"]] = self._fix_k2_depth_units(out[INTERNAL["depth_ppm"]])

        # duration days->hours heuristic
        dur = out[INTERNAL["duration_hours"]]
        per = out[INTERNAL["period_days"]]
        mask_days = (dur > 48) & (per.notna()) & (dur < 24 * per * 5)
        if mask_days.any():
            out.loc[mask_days, INTERNAL["duration_hours"]] = dur[mask_days] * 24.0
        return out

    def transform(self, parts: Dict[str, Tuple[pd.DataFrame, HarmonizeSpec]]) -> pd.DataFrame:
        # We rely on the pickled instance's learned medians/clip/z; here we only harmonize and
        # recompute engineered features to align with training (no jitter at inference).
        dfs = []
        for mission, (df, spec) in parts.items():
            h = self._harmonize_one(df, spec, mission)

            # ensure all trained numeric features are present
            for c in self.numeric_features_:
                if c not in h.columns:
                    h[c] = np.nan

            # impute + missing flags using training medians
            for c in self.numeric_features_:
                miss = h[c].isna()
                h[c + "__missing"] = miss.astype(np.int8)
                med = self.mission_medians.get(mission, pd.Series(dtype=float)).get(c, np.nan)
                if np.isnan(med):
                    all_meds = [m.get(c, np.nan) for m in self.mission_medians.values()]
                    med = np.nanmedian(all_meds) if len(all_meds) else 0.0
                h.loc[miss, c] = med

            # basic features
            pdays  = h[INTERNAL["period_days"]].clip(lower=1e-6)
            dhours = h[INTERNAL["duration_hours"]].clip(lower=1e-6)
            dppm   = h[INTERNAL["depth_ppm"]].clip(lower=1e-9)

            h["log_period"]     = np.log1p(pdays)
            h["log_duration_h"] = np.log1p(dhours)
            h["log_depth_ppm"]  = np.log1p(dppm)
            h["duty_cycle"]     = (dhours / (pdays * 24.0)).clip(0, 1)
            h["depth_per_hour"] = (dppm / dhours)

            # physics ratios (match training)
            Rp_re   = h[INTERNAL["planet_radius_re"]].clip(lower=1e-6)
            Rs_rsun = h[INTERNAL["stellar_radius_rsun"]].clip(lower=1e-6)
            Rs_re   = Rs_rsun * self.RAD_EARTH_PER_SUN
            expected_ppm = 1e6 * (Rp_re / Rs_re)**2
            h["depth_over_radii_model"] = (dppm / expected_ppm).replace([np.inf, -np.inf], np.nan).fillna(1.0)

            if "depth_errw" in h.columns:
                h["snr_proxy"] = (dppm / (h["depth_errw"].replace(0, np.nan))).fillna(0)
            else:
                h["snr_proxy"] = dppm / np.sqrt(h[INTERNAL["duration_hours"]].clip(lower=1e-6))

            if INTERNAL["eqt_k"] in h.columns and INTERNAL["stellar_teff_k"] in h.columns:
                h["eqt_over_teff"] = (h[INTERNAL["eqt_k"]] / h[INTERNAL["stellar_teff_k"]]).replace([np.inf, -np.inf], np.nan).fillna(0)
            else:
                h["eqt_over_teff"] = 0.0

            h["log_depth_over_radii_model"] = np.log1p(h["depth_over_radii_model"].clip(lower=1e-9))
            h["log_snr_proxy"]              = np.log1p(h["snr_proxy"].clip(lower=1e-9))

            dfs.append(h)

        merged = pd.concat(dfs, axis=0, ignore_index=True)

        # star-context features
        g = merged.groupby(INTERNAL["star_id"], sort=False)
        merged["star_multiplicity"] = g[INTERNAL["id"]].transform("count")
        for feat, asc in [(INTERNAL["depth_ppm"], False),
                          (INTERNAL["snr_like"], False),
                          (INTERNAL["period_days"], True)]:
            if feat in merged.columns:
                merged[f"{feat}__rank_in_star"] = g[feat].rank(method="average", ascending=asc)
        if INTERNAL["depth_ppm"] in merged.columns:
            merged["star_depth_mean"] = g[INTERNAL["depth_ppm"]].transform("mean")
            merged["star_depth_std"]  = g[INTERNAL["depth_ppm"]].transform("std").fillna(0)
            merged["star_depth_cv"]   = (merged["star_depth_std"] / (merged["star_depth_mean"].replace(0, np.nan))).fillna(0)

        # per-mission winsorize + z using stored params
        if self.fs.get("mission_standardize", False):
            missions_here = [m for m in pd.Series(merged[INTERNAL["mission"]]).dropna().unique()
                             if m in self.mission_clip_]
            for mission in missions_here:
                mask = (merged[INTERNAL["mission"]] == mission)
                clip_map    = self.mission_clip_.get(mission, {})
                meanstd_map = self.mission_meanstd_.get(mission, {})
                for c in self.numeric_features_:
                    lo, hi = clip_map.get(c, (None, None))
                    if lo is not None:
                        merged.loc[mask, c] = np.clip(merged.loc[mask, c].values, lo, hi)
                    mu, sd = meanstd_map.get(c, (None, None))
                    if mu is not None and sd not in (None, 0):
                        merged.loc[mask, f"{c}__z_miss"] = (merged.loc[mask, c].values - mu) / sd

        # One-hot mission (no defaults added here)
        mission_ohe = pd.get_dummies(merged[INTERNAL["mission"]], prefix="mission", dummy_na=False)
        merged = pd.concat([merged, mission_ohe], axis=1)
        # DO NOT auto-create mission_* columns here; training features list will gate selection
        return merged

# Make unpickling robust if preprocess.pkl was dumped from a notebook (__main__)
import types
if "__main__" not in sys.modules:
    sys.modules["__main__"] = types.ModuleType("__main__")
setattr(sys.modules["__main__"], "ExoPreprocessor", ExoPreprocessor)
setattr(sys.modules["__main__"], "HarmonizeSpec", HarmonizeSpec)


# =====================
# Inference Engine
# =====================
class InferenceEngine:
    """
    Loads a trained bundle directory and exposes:
      - predict_df(df, threshold=..., top_n=...)
      - recommend_threshold (global/per-mission)
      - compute_metrics (if labels provided)
    Expects df with at least: id, mission; optional: label and numeric features.
    STRICT: mission column must be present and ONLY contain Kepler, TESS, or K2.
    """

    def __init__(self, bundle_dir: str):
        self.dir = bundle_dir
        # Core artifacts
        self.pre: ExoPreprocessor = joblib.load(os.path.join(bundle_dir, "preprocess.pkl"))
        self.features: List[str]  = json.load(open(os.path.join(bundle_dir, "feature_list.json")))
        self.labmap: Dict[str, Any] = json.load(open(os.path.join(bundle_dir, "label_mapping.json")))
        self.thr: Dict[str, Any] = json.load(open(os.path.join(bundle_dir, "thresholds.json")))
        self.platt = None
        platt_path = os.path.join(bundle_dir, "calibrator_platt.pkl")
        if os.path.exists(platt_path):
            self.platt = joblib.load(platt_path)
        self.two_stage = bool(self.thr.get("two_stage", False))
        self.use_meta  = bool(self.thr.get("use_meta", False))
        self.bw        = self.thr.get("blend_weights", {"lgbm": 1.0, "cat": 0.0, "xgb": 0.0})

        # Discover base models: {fam: {"A": {seed:[paths]}, "B": {seed:[paths]}}}
        self.registry = {fam: {"A": {}, "B": {}} for fam in ["lgb", "cat", "xgb"]}
        for fname in os.listdir(bundle_dir):
            path = os.path.join(bundle_dir, fname)
            if not (os.path.isfile(path) and fname.startswith("model_")):
                continue
            m = re.search(r"^model_(lgb|cat|xgb)_(A|B)_seed(\d+)_fold(\d+)\.", fname)
            if not m:  # tolerate any extra files
                continue
            fam = m.group(1); stage = m.group(2); seed = int(m.group(3))
            self.registry[fam][stage].setdefault(seed, []).append(path)

        # deterministic order used by meta blender
        self.family_order = ["lgb", "cat", "xgb"]
        self.seed_order = sorted(
            set(s for fam in self.family_order for s in self.registry[fam]["A"].keys())
        )

        # Load meta blender (if present)
        self.meta = None
        meta_path_lr = os.path.join(bundle_dir, "meta_blender.pkl")
        if os.path.exists(meta_path_lr):
            self.meta = joblib.load(meta_path_lr)

        # Cached identities
        self.planet_idx = self.labmap["to_id"][PLANET]
        self.class_names = [self.labmap["to_label"][str(i)] for i in range(3)]

    # ---------- Harmonize incoming df into model features (STRICT MISSIONS) ----------
    def _harmonize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Accepts a single merged dataframe with 'mission' column.
        STRICT: require mission ∈ {Kepler, TESS, K2} for *every* row.
        No guessing, no defaulting, no demotion. If violated → raise ValueError.
        """
        if INTERNAL["mission"] not in df.columns:
            raise ValueError("Missing required column 'mission'. Provide Kepler, TESS, or K2 for each row.")

        # Normalize exact casing only; do not infer/guess from ID and do not default.
        raw = df[INTERNAL["mission"]].astype(str).str.strip()
        norm = raw.copy()
        low  = raw.str.lower()

        # Map lowercase to canonical tokens, but only if it's exactly those names
        norm[low == "kepler"] = "Kepler"
        norm[low == "tess"]   = "TESS"
        norm[low == "k2"]     = "K2"

        # Validate: missing-like?
        missing_like = (
            df[INTERNAL["mission"]].isna()
            | (norm == "")
            | low.isin({"nan","none","null","unknown"})
        )
        if missing_like.any():
            bad_rows = missing_like[missing_like].index[:10].tolist()
            raise ValueError(
                f"'mission' has empty/unknown values in {int(missing_like.sum())} rows "
                f"(e.g., indices {bad_rows}). No defaults will be applied."
            )

        # Validate: unrecognized tokens?
        bad = ~norm.isin(KNOWN_MISSIONS)
        if bad.any():
            bad_vals = norm[bad].value_counts().index.tolist()[:5]
            raise ValueError(
                "Unrecognized mission values detected. Allowed: Kepler, TESS, K2. "
                f"Examples of bad values: {bad_vals}"
            )

        # Now split by normalized mission strictly
        df2 = df.copy()
        df2[INTERNAL["mission"]] = norm

        SELF_SRC = {
            "id": INTERNAL["id"], "star_id": INTERNAL.get("star_id", INTERNAL["id"]),
            "label_candidates": [INTERNAL["label"]],
            "period_days": INTERNAL["period_days"], "duration_hours": INTERNAL["duration_hours"],
            "depth_ppm": INTERNAL["depth_ppm"], "planet_radius_re": INTERNAL["planet_radius_re"],
            "stellar_teff_k": INTERNAL["stellar_teff_k"], "stellar_logg_cgs": INTERNAL["stellar_logg_cgs"],
            "stellar_radius_rsun": INTERNAL["stellar_radius_rsun"],
            "mag_kepler": INTERNAL["mag_kepler"], "mag_tess": INTERNAL["mag_tess"],
            "insolation_earth": INTERNAL["insolation_earth"], "eqt_k": INTERNAL["eqt_k"],
            "snr_like": INTERNAL["snr_like"],
        }

        parts = {}
        for mission in KNOWN_MISSIONS:
            mask = (df2[INTERNAL["mission"]] == mission)
            sub = df2[mask]
            if len(sub):
                parts[mission] = (sub.copy(), HarmonizeSpec(
                    src=SELF_SRC,
                    label_map={
                        PLANET: PLANET, CAND: CAND, FP: FP,
                        PLANET.upper(): PLANET, CAND.upper(): CAND, FP.upper(): FP
                    }
                ))

        if not parts:
            # Should never happen given checks above, but keep an explicit guard.
            raise ValueError("Zero rows after mission filtering. Ensure 'mission' is Kepler, TESS, or K2.")

        data = self.pre.transform(parts)

        # Absolutely no defaulting or demotion to NaN/other after transform.
        # Keep canonical mission tokens as produced above.
        if INTERNAL["mission"] in data.columns:
            s = data[INTERNAL["mission"]].astype(str).str.strip()
            # Tighten casing once more; anything not canonical triggers error
            s = s.replace({"kepler":"Kepler", "tess":"TESS", "k2":"K2"})
            if (~s.isin(KNOWN_MISSIONS)).any():
                bad_vals = s[~s.isin(KNOWN_MISSIONS)].value_counts().index.tolist()[:5]
                raise ValueError(f"Post-transform, mission contained invalid values: {bad_vals}")
            data[INTERNAL["mission"]] = s
        else:
            raise ValueError("Preprocessor output lost the 'mission' column; cannot proceed.")

        # Drop leaky cols (training removed them; ensure same here)
        leaky = [c for c in data.columns if any(s in c.lower() for s in LEAKY_SUBSTRINGS)]
        keep = [c for c in data.columns if (c not in leaky) or (c == INTERNAL["label"])]
        return data[keep]

    # ---------- Family prediction helpers ----------
    def _predict_one_path(self, fam: str, stage: str, path: str, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        if fam == "lgb":
            booster = lgb.Booster(model_file=path)
            P = booster.predict(X, num_iteration=booster.best_iteration)
            return np.asarray(P).reshape(-1)
        elif fam == "cat" and CATBOOST_AVAILABLE:
            clf = CatBoostClassifier()
            clf.load_model(path)
            P = np.array(clf.predict_proba(X))[:, 1]
            return P.reshape(-1)
        elif fam == "xgb" and XGB_AVAILABLE:
            dm = xgb.DMatrix(X)
            bst = xgb.Booster(model_file=path)
            P = bst.predict(dm)
            return np.asarray(P).reshape(-1)
        else:
            return np.full(n, np.nan, dtype=float)

    def _family_seed_tri_probs(self, fam: str, X: pd.DataFrame) -> dict[int, np.ndarray]:
        if fam == "cat" and not CATBOOST_AVAILABLE:
            return {}
        if fam == "xgb" and not XGB_AVAILABLE:
            return {}

        out: dict[int, np.ndarray] = {}
        seeds = sorted(self.registry[fam]["A"].keys())
        for seed in seeds:
            pathsA = sorted(self.registry[fam]["A"].get(seed, []))
            if self.two_stage:
                pathsB = sorted(self.registry[fam]["B"].get(seed, []))
                k = min(len(pathsA), len(pathsB))
                if k == 0:
                    continue
                tri_list = []
                for pa, pb in zip(pathsA[:k], pathsB[:k]):
                    pA = self._predict_one_path(fam, "A", pa, X)  # (n,)
                    pB = self._predict_one_path(fam, "B", pb, X)  # (n,)
                    P = np.zeros((len(X), 3), dtype=float)
                    P[:, self.planet_idx] = pA
                    P[:, LABEL2ID[CAND]]  = (1.0 - pA) * pB
                    P[:, LABEL2ID[FP]]    = (1.0 - pA) * (1.0 - pB)
                    tri_list.append(P)
                out[seed] = np.mean(tri_list, axis=0)
            else:
                tri_list = []
                for pa in pathsA:
                    raw = self._predict_one_path(fam, "A", pa, X)  # (n,)
                    if raw.ndim == 1:
                        P = np.zeros((len(X), 3), dtype=float)
                        P[:, self.planet_idx] = raw
                        P[:, LABEL2ID[FP]]    = 1.0 - raw
                    else:
                        P = raw  # already (n,3)
                    tri_list.append(P)
                if tri_list:
                    out[seed] = np.mean(tri_list, axis=0)
        return out

    # ---------- Public: run predictions ----------
    def predict_df(
        self,
        df_in: pd.DataFrame,
        threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        use_per_mission: bool = False
    ) -> Dict[str, Any]:
        """
        Returns dict with:
          - preds: DataFrame [id, mission, P(planet), PredictedClass, Confidence, Planet>=Thr]
          - threshold_used, per_mission_thresholds
          - metrics (if labels exist), curves (if labels exist)
        STRICT: raises ValueError if 'mission' is missing/invalid.
        """
        # Validator remains lenient for UI, but inference will enforce strict missions.
        ok, vrep = validate_df(df_in)
        if not ok and vrep["status"] == "Cannot score":
            return {"error": "Cannot score", "validator": vrep}

        # Strip leaky columns (safety)
        df = df_in.copy()
        leak_cols = [c for c in df.columns if any(s in c.lower() for s in LEAKY_SUBSTRINGS)]
        df = df.drop(columns=leak_cols, errors="ignore")

        # Harmonize → model matrix (STRICT mission checks happen inside)
        data = self._harmonize_df(df)
        X = data[self.features].copy()
        n = len(X)

        # Family probs (for blend) and per-seed probs (for meta)
        fam_probs_blend: Dict[str, Optional[np.ndarray]] = {"lgb": None, "cat": None, "xgb": None}
        per_seed: Dict[str, Dict[int, np.ndarray]] = {}
        for fam in self.family_order:
            seed_map = self._family_seed_tri_probs(fam, X)  # {seed: (n,3)}
            if seed_map:
                per_seed[fam] = seed_map
                fam_probs_blend[fam] = np.mean(list(seed_map.values()), axis=0)

        enabled_fams_for_blend = [f for f,v in fam_probs_blend.items() if v is not None]
        if not enabled_fams_for_blend:
            raise RuntimeError("No base models available to score (missing model files or unsupported families).")

        # Meta (expecting specific concatenation shape), else weighted blend
        probs = None
        can_meta = (
            self.meta is not None
            and all(f in per_seed and len(per_seed[f]) >= 3 for f in self.family_order)
        )
        if can_meta:
            blocks = []
            for fam in self.family_order:
                for seed in self.seed_order[:3]:
                    Ptri = per_seed[fam][seed]
                    blocks.append(Ptri)
            Z = np.concatenate(blocks, axis=1)  # (n, 27) if 3 fams × 3 seeds × 3 classes
            expected = getattr(self.meta, "n_features_in_", Z.shape[1])
            if Z.shape[1] == expected:
                probs = self.meta.predict_proba(Z)

        if probs is None:
            probs = np.zeros((n, 3), dtype=float)
            W = self.bw  # e.g., {"lgbm":0.5,"cat":0.5,"xgb":0.0}
            for fam in enabled_fams_for_blend:
                if fam == "lgb": probs += W.get("lgbm", 0.0) * fam_probs_blend[fam]
                if fam == "cat": probs += W.get("cat",  0.0) * fam_probs_blend[fam]
                if fam == "xgb": probs += W.get("xgb",  0.0) * fam_probs_blend[fam]

        # P(planet) + optional Platt calibration
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        p_planet = np.nan_to_num(probs[:, self.planet_idx], nan=0.0, posinf=1.0, neginf=0.0)
        if self.platt is not None:
            p_planet = self.platt.predict_proba(p_planet.reshape(-1,1))[:,1]
        p_planet = np.clip(np.nan_to_num(p_planet, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        # Threshold selection
        per_mission_thr = self.thr.get("per_mission", {}) if use_per_mission else {}
        if top_n is not None and top_n >= 1:
            thr_used = _threshold_for_topn(p_planet, top_n)
        elif threshold is not None:
            thr_used = float(threshold)
        else:
            thr_used = float(self.thr.get("planet_recommended", 0.5))

        # Final table
        y_idx = np.argmax(probs, axis=1)
        y_name = [self.class_names[i] for i in y_idx]
        def conf_tag(p): return "High" if p>=0.8 else ("Medium" if p>=0.6 else "Low")

        missions = data[INTERNAL["mission"]].astype(str)
        preds = pd.DataFrame({
            INTERNAL["id"]: data[INTERNAL["id"]].values,
            INTERNAL["mission"]: missions.values,
            "P(planet)": p_planet,
            "PredictedClass": y_name,
            "Confidence": [conf_tag(p) for p in p_planet],
        })

        if per_mission_thr:
            preds["Planet>=Thr"] = [
                int(p >= per_mission_thr.get(m, thr_used))
                for m, p in zip(preds[INTERNAL["mission"]].values, preds["P(planet)"].values)
            ]
        else:
            preds["Planet>=Thr"] = (preds["P(planet)"].values >= thr_used).astype(int)

        # Metrics/curves if label column exists and is recognized
        metrics, curves = None, None
        if INTERNAL["label"] in data.columns:
            lab = data[INTERNAL["label"]].astype(str)
            have = lab.isin([PLANET, CAND, FP])
            if have.any():
                y_true_tri = lab.map(LABEL2ID).to_numpy()
                y_true_bin = (y_true_tri == LABEL2ID[PLANET]).astype(int)
                metrics, curves = _compute_metrics(y_true_bin, y_true_tri, probs, p_planet, thr_used)

        return {
            "preds": preds,
            "threshold_used": thr_used,
            "per_mission_thresholds": per_mission_thr,
            "metrics": metrics,
            "curves": curves
        }


# =====================
# Metrics, curves, helpers
# =====================
def _threshold_for_topn(p: np.ndarray, n_pos: int) -> float:
    """ Pick a threshold so that ~n_pos items are >= thr (ties break arbitrarily). """
    n = len(p)
    if n_pos <= 0:
        return 1.0
    if n_pos >= n:
        return 0.0
    kth = np.partition(p, -n_pos)[-n_pos]
    return float(kth)

def _compute_metrics(y_true_bin, y_true_tri, probs3, p_planet, thr):
    # Binary at threshold
    y_pred_bin = (p_planet >= thr).astype(int)

    # Headline
    pr_auc = float(average_precision_score(y_true_bin, p_planet))
    fpr, tpr, _ = roc_curve(y_true_bin, p_planet)
    roc_auc = float(auc(fpr, tpr))
    f1 = float(f1_score(y_true_bin, y_pred_bin))
    rec = float(recall_score(y_true_bin, y_pred_bin))
    brier = float(brier_score_loss(y_true_bin, p_planet))
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0,1]).tolist()

    # Tri-class accuracy via argmax
    tri_pred = probs3.argmax(axis=1)
    tri_acc  = float((tri_pred == y_true_tri).mean())

    # Curves
    prec, rec_curve, thr_curve = precision_recall_curve(y_true_bin, p_planet)
    cal_curve = _calibration_curve(y_true_bin, p_planet, n_bins=10)

    metrics = {
        "PR_AUC": pr_auc, "ROC_AUC": roc_auc, "F1": f1, "Recall": rec,
        "Brier": brier, "TriClassAcc": tri_acc, "ConfusionBinary": cm
    }
    curves = {
        "pr": {"precision": prec.tolist(), "recall": rec_curve.tolist(), "thresholds": thr_curve.tolist()},
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "calibration": cal_curve
    }
    return metrics, curves

def _calibration_curve(y_true, y_prob, n_bins=10):
    """ Returns points for reliability plot (mean predicted vs empirical). """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    out = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins-1 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            out.append({"bin": i, "p_mean": float((lo+hi)/2), "empirical": None, "count": 0})
        else:
            pmean = float(y_prob[mask].mean())
            emp = float(y_true[mask].mean())
            out.append({"bin": i, "p_mean": pmean, "empirical": emp, "count": int(mask.sum())})
    return out


# =====================
# Visualization helpers
# =====================

MISSIONS = ["Kepler", "TESS", "K2"]


def _err_width(df: pd.DataFrame, pos: str, neg: str) -> pd.Series:
    if pos not in df.columns and neg not in df.columns:
        return pd.Series(np.nan, index=df.index)
    pos_vals = pd.to_numeric(df[pos], errors="coerce") if pos in df.columns else pd.Series(np.nan, index=df.index)
    neg_vals = pd.to_numeric(df[neg], errors="coerce") if neg in df.columns else pd.Series(np.nan, index=df.index)
    return (pos_vals.abs() + neg_vals.abs()) / 2.0


def _prepare_training_kepler(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out[INTERNAL["mission"]] = "Kepler"
    out[INTERNAL["id"]] = df.get("kepoi_name", df.index).astype(str)
    out[INTERNAL["star_id"]] = df.get("kepid", out[INTERNAL["id"]]).astype(str)

    disp = df.get("koi_disposition", pd.Series("", index=df.index)).astype(str).str.upper().str.strip()
    pdisp = df.get("koi_pdisposition", pd.Series("", index=df.index)).astype(str).str.upper().str.strip()
    label = disp
    label = label.mask(~label.isin(KEPLER_MAP.keys()) & pdisp.isin(KEPLER_MAP.keys()), pdisp)
    out[INTERNAL["label"]] = label.map(KEPLER_MAP)

    def col(name: str):
        return pd.to_numeric(df.get(name), errors="coerce")

    out[INTERNAL["period_days"]] = col("koi_period")
    out[INTERNAL["duration_hours"]] = col("koi_duration")
    out[INTERNAL["depth_ppm"]] = col("koi_depth")
    out[INTERNAL["planet_radius_re"]] = col("koi_prad")
    out[INTERNAL["stellar_teff_k"]] = col("koi_steff")
    out[INTERNAL["stellar_logg_cgs"]] = col("koi_slogg")
    out[INTERNAL["stellar_radius_rsun"]] = col("koi_srad")
    out[INTERNAL["mag_kepler"]] = col("koi_kepmag")
    out[INTERNAL["mag_tess"]] = np.nan
    out[INTERNAL["insolation_earth"]] = col("koi_insol")
    out[INTERNAL["eqt_k"]] = col("koi_teq")
    out[INTERNAL["snr_like"]] = col("koi_model_snr")

    out["period_errw"] = _err_width(df, "koi_period_err1", "koi_period_err2")
    out["duration_errw"] = _err_width(df, "koi_duration_err1", "koi_duration_err2")
    out["depth_errw"] = _err_width(df, "koi_depth_err1", "koi_depth_err2")
    out["prad_errw"] = _err_width(df, "koi_prad_err1", "koi_prad_err2")
    out["teff_errw"] = _err_width(df, "koi_steff_err1", "koi_steff_err2")
    out["logg_errw"] = _err_width(df, "koi_slogg_err1", "koi_slogg_err2")
    out["srad_errw"] = _err_width(df, "koi_srad_err1", "koi_srad_err2")
    out["insol_errw"] = _err_width(df, "koi_insol_err1", "koi_insol_err2")
    out["eqt_errw"] = _err_width(df, "koi_teq_err1", "koi_teq_err2")

    return out


def _prepare_training_tess(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out[INTERNAL["mission"]] = "TESS"
    out[INTERNAL["id"]] = df.get("toi", df.index).astype(str)
    out[INTERNAL["star_id"]] = df.get("tid", out[INTERNAL["id"]]).astype(str)

    disp = df.get("tfopwg_disp", pd.Series("", index=df.index)).astype(str).str.upper().str.strip()
    out[INTERNAL["label"]] = disp.map(TOI_MAP)

    def col(name: str):
        return pd.to_numeric(df.get(name), errors="coerce")

    out[INTERNAL["period_days"]] = col("pl_orbper")
    out[INTERNAL["duration_hours"]] = col("pl_trandurh")
    out[INTERNAL["depth_ppm"]] = col("pl_trandep")
    out[INTERNAL["planet_radius_re"]] = col("pl_rade")
    out[INTERNAL["stellar_teff_k"]] = col("st_teff")
    out[INTERNAL["stellar_logg_cgs"]] = col("st_logg")
    out[INTERNAL["stellar_radius_rsun"]] = col("st_rad")
    out[INTERNAL["mag_kepler"]] = np.nan
    out[INTERNAL["mag_tess"]] = col("st_tmag")
    out[INTERNAL["insolation_earth"]] = col("pl_insol")
    out[INTERNAL["eqt_k"]] = col("pl_eqt")
    out[INTERNAL["snr_like"]] = np.nan

    out["period_errw"] = _err_width(df, "pl_orbpererr1", "pl_orbpererr2")
    out["duration_errw"] = _err_width(df, "pl_trandurherr1", "pl_trandurherr2")
    out["depth_errw"] = _err_width(df, "pl_trandeperr1", "pl_trandeperr2")
    out["prad_errw"] = _err_width(df, "pl_radeerr1", "pl_radeerr2")
    out["teff_errw"] = _err_width(df, "st_tefferr1", "st_tefferr2")
    out["logg_errw"] = _err_width(df, "st_loggerr1", "st_loggerr2")
    out["srad_errw"] = _err_width(df, "st_raderr1", "st_raderr2")
    out["insol_errw"] = _err_width(df, "pl_insolerr1", "pl_insolerr2")
    out["eqt_errw"] = _err_width(df, "pl_eqterr1", "pl_eqterr2")

    return out


def _prepare_training_k2(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out[INTERNAL["mission"]] = "K2"
    out[INTERNAL["id"]] = df.get("pl_name", df.index).astype(str)
    out[INTERNAL["star_id"]] = df.get("hostname", out[INTERNAL["id"]]).astype(str)

    disp = df.get("disposition", pd.Series("", index=df.index)).astype(str).str.upper().str.strip()
    out[INTERNAL["label"]] = disp.map(K2_MAP)

    def col(name: str):
        return pd.to_numeric(df.get(name), errors="coerce")

    out[INTERNAL["period_days"]] = col("pl_orbper")
    out[INTERNAL["duration_hours"]] = np.nan  # not available
    out[INTERNAL["depth_ppm"]] = np.nan
    out[INTERNAL["planet_radius_re"]] = col("pl_rade")
    out[INTERNAL["stellar_teff_k"]] = col("st_teff")
    out[INTERNAL["stellar_logg_cgs"]] = col("st_logg")
    out[INTERNAL["stellar_radius_rsun"]] = col("st_rad")
    out[INTERNAL["mag_kepler"]] = col("sy_vmag")
    out[INTERNAL["mag_tess"]] = np.nan
    out[INTERNAL["insolation_earth"]] = col("pl_insol")
    out[INTERNAL["eqt_k"]] = col("pl_eqt")
    out[INTERNAL["snr_like"]] = np.nan

    out["period_errw"] = _err_width(df, "pl_orbpererr1", "pl_orbpererr2")
    out["duration_errw"] = np.nan
    out["depth_errw"] = np.nan
    out["prad_errw"] = _err_width(df, "pl_radeerr1", "pl_radeerr2")
    out["teff_errw"] = _err_width(df, "st_tefferr1", "st_tefferr2")
    out["logg_errw"] = _err_width(df, "st_loggerr1", "st_loggerr2")
    out["srad_errw"] = _err_width(df, "st_raderr1", "st_raderr2")
    out["insol_errw"] = _err_width(df, "pl_insolerr1", "pl_insolerr2")
    out["eqt_errw"] = _err_width(df, "pl_eqterr1", "pl_eqterr2")

    return out


def _load_training_reference(engine: InferenceEngine, training_dir: str) -> pd.DataFrame:
    parts = []
    loaders = {
        "Kepler": ("cumulative.csv", _prepare_training_kepler),
        "TESS": ("toi.csv", _prepare_training_tess),
        "K2": ("k2_planets_candidates.csv", _prepare_training_k2),
    }
    for mission, (fname, builder) in loaders.items():
        path = os.path.join(training_dir, fname)
        if not os.path.exists(path):
            continue
        raw = pd.read_csv(path, low_memory=False)
        df_std = builder(raw)
        df_std = df_std.dropna(subset=[INTERNAL["id"]])
        try:
            harmonized = engine._harmonize_df(df_std)
            parts.append(harmonized)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: unable to harmonize training file {fname}: {exc}", file=sys.stderr)
    if parts:
        return pd.concat(parts, axis=0, ignore_index=True)
    return pd.DataFrame()


def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return float(f)


def _format_range(lo: float, hi: float) -> str:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "—"

    def fmt(x: float) -> str:
        ax = abs(x)
        if ax >= 10000 or (0 < ax < 1e-3):
            return f"{x:.2e}"
        if ax >= 100:
            return f"{x:.0f}"
        if ax >= 10:
            return f"{x:.1f}"
        return f"{x:.2f}"

    return f"{fmt(lo)}–{fmt(hi)}"


FEATURE_SPECS = [
    {"feature": INTERNAL["period_days"], "label": "Orbital period (days)", "log": True},
    {"feature": INTERNAL["duration_hours"], "label": "Transit duration (hours)", "log": True},
    {"feature": INTERNAL["depth_ppm"], "label": "Transit depth (ppm)", "log": True},
    {"feature": INTERNAL["planet_radius_re"], "label": "Planet radius (R⊕)", "log": False},
    {"feature": INTERNAL["stellar_teff_k"], "label": "Stellar Teff (K)", "log": False},
    {"feature": INTERNAL["stellar_radius_rsun"], "label": "Stellar radius (R☉)", "log": False},
    {"feature": INTERNAL["snr_like"], "label": "Pipeline SNR", "log": True},
    {"feature": "duty_cycle", "label": "Duty cycle", "log": False},
    {"feature": "depth_per_hour", "label": "Depth per hour", "log": True},
    {"feature": "depth_over_radii_model", "label": "Depth / radii model", "log": True},
    {"feature": "snr_proxy", "label": "Proxy SNR", "log": True},
    {"feature": "eqt_over_teff", "label": "Equilibrium/Teff", "log": False},
]

OUTLIER_FEATURES = [
    (INTERNAL["depth_ppm"], "ppm"),
    (INTERNAL["planet_radius_re"], "R⊕"),
    (INTERNAL["stellar_teff_k"], "K"),
    ("depth_over_radii_model", "ratio"),
    ("snr_proxy", ""),
    ("duty_cycle", "fraction"),
]

SCATTER_SPECS = [
    {
        "id": "period_depth",
        "label": "Period vs depth",
        "x": INTERNAL["period_days"],
        "y": INTERNAL["depth_ppm"],
        "xLabel": "Period (days)",
        "yLabel": "Depth (ppm)",
        "logX": True,
        "logY": True,
    },
    {
        "id": "period_duration",
        "label": "Period vs duration",
        "x": INTERNAL["period_days"],
        "y": INTERNAL["duration_hours"],
        "xLabel": "Period (days)",
        "yLabel": "Duration (hours)",
        "logX": True,
        "logY": True,
    },
    {
        "id": "radius_depth",
        "label": "Radius vs depth",
        "x": INTERNAL["planet_radius_re"],
        "y": INTERNAL["depth_ppm"],
        "xLabel": "Planet radius (R⊕)",
        "yLabel": "Depth (ppm)",
        "logX": False,
        "logY": True,
    },
    {
        "id": "teff_eqt",
        "label": "Teff vs equilibrium",
        "x": INTERNAL["stellar_teff_k"],
        "y": INTERNAL["eqt_k"],
        "xLabel": "Stellar Teff (K)",
        "yLabel": "Equilibrium temp (K)",
        "logX": False,
        "logY": False,
    },
]


def _compute_histogram_stats(
    scoring: pd.DataFrame,
    training: pd.DataFrame,
    feature: str,
    label: str,
    log_scale: bool,
):
    scoring_series = pd.to_numeric(scoring.get(feature), errors="coerce")
    training_series = pd.to_numeric(training.get(feature), errors="coerce") if not training.empty else pd.Series(dtype=float)

    combined = pd.concat([scoring_series, training_series], ignore_index=True)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
    if log_scale:
        combined = combined[combined > 0]

    if combined.empty:
        edges = np.linspace(0.0, 1.0, 11)
    else:
        lo = combined.min()
        hi = combined.max()
        if log_scale:
            lo = max(lo, 1e-9)
            hi = max(hi, lo * 1.0001)
            edges = np.logspace(np.log10(lo), np.log10(hi), 11)
        else:
            if hi == lo:
                hi = lo + 1.0
            edges = np.linspace(lo, hi, 11)

    scoring_counts: Dict[str, np.ndarray] = {}
    training_counts: Dict[str, np.ndarray] = {}
    for mission in MISSIONS:
        s_mask = scoring[INTERNAL["mission"]] == mission
        s_vals = scoring_series[s_mask].replace([np.inf, -np.inf], np.nan).dropna()
        if log_scale:
            s_vals = s_vals[s_vals > 0]
        s_hist, _ = np.histogram(s_vals.to_numpy(), bins=edges)
        scoring_counts[mission] = s_hist

        if training.empty:
            training_counts[mission] = np.zeros(len(edges) - 1, dtype=int)
        else:
            t_mask = training[INTERNAL["mission"]] == mission
            t_vals = training_series[t_mask].replace([np.inf, -np.inf], np.nan).dropna()
            if log_scale:
                t_vals = t_vals[t_vals > 0]
            t_hist, _ = np.histogram(t_vals.to_numpy(), bins=edges)
            training_counts[mission] = t_hist

    bins_payload = []
    for idx in range(len(edges) - 1):
        bins_payload.append(
            {
                "bin": _format_range(edges[idx], edges[idx + 1]),
                "missions": {mission: int(scoring_counts[mission][idx]) for mission in MISSIONS},
            }
        )

    return {
        "config": {
            "feature": feature,
            "label": label,
            "logScale": log_scale,
            "bins": bins_payload,
        },
        "edges": edges,
        "scoring_counts": scoring_counts,
        "training_counts": training_counts,
    }


def _compute_quantile_payload(
    scoring: pd.DataFrame,
    training: pd.DataFrame,
    feature: str,
    log_scale: bool,
    edges: np.ndarray,
    scoring_counts: Dict[str, np.ndarray],
):
    items = []
    scoring_series = pd.to_numeric(scoring.get(feature), errors="coerce")
    training_series = pd.to_numeric(training.get(feature), errors="coerce") if not training.empty else pd.Series(dtype=float)

    for mission in MISSIONS:
        t_mask = training[INTERNAL["mission"]] == mission if not training.empty else pd.Series(dtype=bool)
        t_vals = training_series[t_mask].replace([np.inf, -np.inf], np.nan).dropna()
        if log_scale:
            t_vals = t_vals[t_vals > 0]

        if not t_vals.empty:
            train_quant = {
                "p10": _safe_float(t_vals.quantile(0.10)),
                "median": _safe_float(t_vals.quantile(0.50)),
                "p90": _safe_float(t_vals.quantile(0.90)),
                "p5": _safe_float(t_vals.quantile(0.05)),
                "p95": _safe_float(t_vals.quantile(0.95)),
                "iqr": _safe_float(t_vals.quantile(0.75) - t_vals.quantile(0.25)),
            }
        else:
            train_quant = {"p10": None, "median": None, "p90": None, "p5": None, "p95": None, "iqr": None}

        items.append(
            {
                "feature": feature,
                "mission": mission,
                "logScale": log_scale,
                "train": train_quant,
                "scoring": {
                    "edges": edges.tolist(),
                    "counts": scoring_counts[mission].astype(int).tolist(),
                },
            }
        )

    return items


def _compute_outliers(scoring: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    ids = scoring[INTERNAL["id"]].astype(str)
    for feature, units in OUTLIER_FEATURES:
        series = pd.to_numeric(scoring.get(feature), errors="coerce").replace([np.inf, -np.inf], np.nan)
        if series is None:
            continue
        for mission in MISSIONS:
            mask = scoring[INTERNAL["mission"]] == mission
            vals = series[mask].dropna()
            if len(vals) < 5:
                continue
            mean = vals.mean()
            std = vals.std(ddof=0)
            if std == 0 or np.isnan(std):
                continue
            zscores = (vals - mean) / std
            p1 = vals.quantile(0.01)
            p99 = vals.quantile(0.99)
            flagged = vals[(np.abs(zscores) > 3) | (vals < p1) | (vals > p99)]
            for idx, value in flagged.items():
                rows.append(
                    {
                        "id": ids.loc[idx],
                        "mission": mission,
                        "feature": feature,
                        "zScore": _safe_float(zscores.loc[idx]) or 0.0,
                        "value": _safe_float(value) or 0.0,
                        "units": units,
                    }
                )
    rows.sort(key=lambda item: abs(item.get("zScore", 0.0)), reverse=True)
    return rows[:20]


def _compute_duplicates(scoring: pd.DataFrame) -> dict[str, Any]:
    ids = scoring[INTERNAL["id"]].astype(str)
    id_counts = ids.value_counts()
    dup_ids = int((id_counts > 1).sum())
    dup_rows = int(id_counts[id_counts > 1].sum())
    dup_mission = (
        scoring.loc[ids.isin(id_counts[id_counts > 1].index), INTERNAL["mission"]]
        .value_counts()
        .reindex(MISSIONS, fill_value=0)
    )

    star_mult = scoring.get("star_multiplicity", pd.Series(dtype=float))
    star_counts = []
    if star_mult is not None and not star_mult.empty:
        counts = star_mult.value_counts().sort_index()
        for mult, count in counts.items():
            star_counts.append({"value": int(mult), "count": int(count)})

    return {
        "duplicateIds": dup_ids,
        "duplicateRows": dup_rows,
        "perMission": {mission: int(dup_mission.get(mission, 0)) for mission in MISSIONS},
        "starMultiplicity": star_counts,
    }


def _compute_score_histogram(preds: pd.DataFrame) -> list[dict[str, Any]]:
    scores = pd.to_numeric(preds.get("P(planet)"), errors="coerce").fillna(0.0)
    edges = np.linspace(0.0, 1.0, 11)
    mission_counts = {}
    for mission in MISSIONS:
        mask = preds[INTERNAL["mission"]] == mission
        hist, _ = np.histogram(scores[mask], bins=edges)
        mission_counts[mission] = hist

    payload = []
    for idx in range(len(edges) - 1):
        entry = {
            "lower": float(edges[idx]),
            "upper": float(edges[idx + 1]),
            "total": int(sum(mission_counts[mission][idx] for mission in MISSIONS)),
            "missions": {mission: int(mission_counts[mission][idx]) for mission in MISSIONS},
        }
        payload.append(entry)
    return payload


def _compute_threshold_snapshots(
    preds: pd.DataFrame,
    scores: np.ndarray,
    labels: Optional[np.ndarray],
    pr_auc: Optional[float],
    roc_auc: Optional[float],
    brier_score: Optional[float],
) -> list[dict[str, Any]]:
    thresholds = np.linspace(0, 1, 21)
    snapshots = []
    for thr in thresholds:
        matrix = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        f1 = None
        recall = None
        if labels is not None:
            preds_bin = (scores >= thr).astype(int)
            tp = int(((labels == 1) & (preds_bin == 1)).sum())
            fp = int(((labels == 0) & (preds_bin == 1)).sum())
            fn = int(((labels == 1) & (preds_bin == 0)).sum())
            tn = int(((labels == 0) & (preds_bin == 0)).sum())
            matrix = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recall = rec
            if prec + rec > 0:
                f1 = 2 * prec * rec / (prec + rec)
            else:
                f1 = 0.0

        snapshots.append(
            {
                "threshold": float(thr),
                "prAuc": pr_auc,
                "rocAuc": roc_auc,
                "brier": brier_score,
                "f1": f1,
                "recall": recall,
                "matrix": matrix,
            }
        )
    return snapshots


def _load_feature_importance(bundle_dir: str) -> list[dict[str, Any]]:
    path = os.path.join(bundle_dir, "metrics.json")
    if not os.path.exists(path):
        return []
    try:
        data = json.load(open(path))
    except Exception:  # pragma: no cover - defensive
        return []
    items = data.get("feature_importance_A_top20") or []
    out = []
    for item in items:
        feature = item.get("feature")
        gain = _safe_float(item.get("avg_gain"))
        if feature is None or gain is None:
            continue
        out.append({"feature": feature, "gain": gain})
    out.sort(key=lambda x: x["gain"], reverse=True)
    return out[:20]


def _sanitize(obj: Any):
    if isinstance(obj, pd.DataFrame):
        return _sanitize(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _sanitize(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return _safe_float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def build_visualization_payload(
    bundle_dir: str,
    csv_path: str,
    training_dir: str,
) -> Dict[str, Any]:
    eng = InferenceEngine(bundle_dir)
    raw = load_any_csv(csv_path)
    scoring = eng._harmonize_df(raw)
    training = _load_training_reference(eng, training_dir)

    pred_out = eng.predict_df(raw, use_per_mission=True)
    preds_df = pred_out.get("preds")
    if isinstance(preds_df, pd.DataFrame):
        preds = preds_df.copy()
    else:
        preds = pd.DataFrame(preds_df or [])

    scores = pd.to_numeric(preds.get("P(planet)"), errors="coerce").fillna(0.0).to_numpy()
    missions_series = scoring[INTERNAL["mission"]].astype(str)

    mission_mix = (
    missions_series.value_counts()
    .reindex(MISSIONS, fill_value=0)
    .rename_axis("mission")
    .reset_index(name="count")
    )
    mission_mix_payload = mission_mix.to_dict("records")

    missing_cols = [c for c in scoring.columns if c.endswith("__missing")]
    missing_payload = []
    for col in missing_cols:
        for mission in MISSIONS:
            mask = missions_series == mission
            if mask.any():
                rate = scoring.loc[mask, col].astype(float).mean()
            else:
                rate = 0.0
            missing_payload.append({"feature": col, "mission": mission, "missingRate": float(rate or 0.0)})

    histogram_results = []
    quantile_payload = []
    drift_payload = []
    for spec in FEATURE_SPECS:
        feature = spec["feature"]
        if feature not in scoring.columns:
            continue
        result = _compute_histogram_stats(scoring, training, feature, spec["label"], spec["log"])
        histogram_results.append(result["config"])
        quantile_payload.extend(
            _compute_quantile_payload(
                scoring,
                training,
                feature,
                spec["log"],
                result["edges"],
                result["scoring_counts"],
            )
        )

        for mission in MISSIONS:
            train_counts = result["training_counts"].get(mission)
            score_counts = result["scoring_counts"].get(mission)
            if train_counts is None or score_counts is None:
                continue
            train_vals = pd.to_numeric(training.get(feature), errors="coerce") if not training.empty else pd.Series(dtype=float)
            train_vals = train_vals[training[INTERNAL["mission"]] == mission] if not training.empty else pd.Series(dtype=float)
            train_vals = train_vals.replace([np.inf, -np.inf], np.nan).dropna()
            if spec["log"]:
                train_vals = train_vals[train_vals > 0]

            score_vals = pd.to_numeric(scoring.get(feature), errors="coerce")
            score_vals = score_vals[missions_series == mission].replace([np.inf, -np.inf], np.nan).dropna()
            if spec["log"]:
                score_vals = score_vals[score_vals > 0]

            if not train_vals.empty:
                train_median = train_vals.median()
                train_q25 = train_vals.quantile(0.25)
                train_q75 = train_vals.quantile(0.75)
                train_iqr = train_q75 - train_q25
                train_p5 = train_vals.quantile(0.05)
                train_p95 = train_vals.quantile(0.95)
                train_p = train_counts.astype(float)
                score_p = score_counts.astype(float)
                train_prob = train_p / train_p.sum() if train_p.sum() > 0 else np.ones_like(train_p) / len(train_p)
                score_prob = score_p / score_p.sum() if score_p.sum() > 0 else np.ones_like(score_p) / len(score_p)
                mask = (train_prob > 0) & (score_prob > 0)
                if mask.any():
                    kl = float(np.sum(score_prob[mask] * np.log((score_prob[mask] + 1e-12) / (train_prob[mask] + 1e-12))))
                else:
                    kl = 0.0
                if train_iqr == 0 or np.isnan(train_iqr):
                    delta = None
                else:
                    delta = _safe_float((score_vals.median() - train_median) / train_iqr)
                if train_p95 is not None and train_p5 is not None:
                    coverage = float(((score_vals >= train_p5) & (score_vals <= train_p95)).mean()) if len(score_vals) else 0.0
                else:
                    coverage = 0.0
            else:
                delta = None
                kl = 0.0
                coverage = 0.0

            drift_payload.append(
                {
                    "feature": feature,
                    "mission": mission,
                    "delta": delta,
                    "kl": kl,
                    "coverage": coverage,
                }
            )

    scatter_payload = []
    ids = scoring[INTERNAL["id"]].astype(str)
    for spec in SCATTER_SPECS:
        x_vals = pd.to_numeric(scoring.get(spec["x"]), errors="coerce")
        y_vals = pd.to_numeric(scoring.get(spec["y"]), errors="coerce")
        points = []
        for idx in scoring.index:
            x = x_vals.at[idx]
            y = y_vals.at[idx]
            if np.isnan(x) or np.isnan(y):
                continue
            if spec.get("logX") and x <= 0:
                continue
            if spec.get("logY") and y <= 0:
                continue
            px = float(np.log10(x)) if spec.get("logX") else float(x)
            py = float(np.log10(y)) if spec.get("logY") else float(y)
            points.append(
                {
                    "id": f"{spec['id']}-{idx}",
                    "mission": missions_series.at[idx],
                    "x": px,
                    "y": py,
                }
            )
        if len(points) > 600:
            points = points[:600]
        scatter_payload.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "xLabel": spec["xLabel"],
                "yLabel": spec["yLabel"],
                "logX": spec.get("logX", False),
                "logY": spec.get("logY", False),
                "points": points,
            }
        )

    outliers_payload = _compute_outliers(scoring)
    duplicates_payload = _compute_duplicates(scoring)
    score_hist_payload = _compute_score_histogram(preds)

    curves = pred_out.get("curves") or {}
    pr_curve = curves.get("pr", {})
    roc_curve = curves.get("roc", {})
    cal_curve = curves.get("calibration", [])

    pr_points = []
    if pr_curve:
        precision = pr_curve.get("precision", [])
        recall_vals = pr_curve.get("recall", [])
        pr_points = [
            {"x": _safe_float(rec), "y": _safe_float(prec)}
            for prec, rec in zip(precision, recall_vals)
            if _safe_float(rec) is not None and _safe_float(prec) is not None
        ]
    roc_points = []
    if roc_curve:
        fpr = roc_curve.get("fpr", [])
        tpr = roc_curve.get("tpr", [])
        roc_points = [
            {"x": _safe_float(f), "y": _safe_float(t)}
            for f, t in zip(fpr, tpr)
            if _safe_float(f) is not None and _safe_float(t) is not None
        ]
    cal_points = []
    for entry in cal_curve:
        pm = entry.get("p_mean")
        emp = entry.get("empirical")
        if emp is None:
            continue
        cal_points.append({"x": _safe_float(pm), "y": _safe_float(emp)})

    labels_series = scoring.get(INTERNAL["label"])
    label_binary = None
    if labels_series is not None:
        mapped = labels_series.astype(str).map({PLANET: 1, CAND: 0, FP: 0})
        if mapped.notna().any():
            label_binary = mapped.fillna(0).astype(int).to_numpy()

    metrics = pred_out.get("metrics") or {}
    snapshots = _compute_threshold_snapshots(
        preds,
        scores,
        label_binary,
        _safe_float(metrics.get("PR_AUC")),
        _safe_float(metrics.get("ROC_AUC")),
        _safe_float(metrics.get("Brier")),
    )

    threshold_used = pred_out.get("threshold_used")
    per_mission_thr = pred_out.get("per_mission_thresholds") or {}

    mission_acceptance_payload = []
    for mission in MISSIONS:
        thr = per_mission_thr.get(mission, threshold_used)
        thr = thr if isinstance(thr, (int, float)) else threshold_used
        mission_scores = scores[preds[INTERNAL["mission"]] == mission]
        accepted = int((mission_scores >= (thr or 0.5)).sum())
        total = int((preds[INTERNAL["mission"]] == mission).sum())
        mission_acceptance_payload.append(
            {
                "mission": mission,
                "accepted": accepted,
                "rejected": max(total - accepted, 0),
            }
        )

    candidates_payload = []
    if not preds.empty:
        merged = preds.merge(
            scoring[[INTERNAL["id"], INTERNAL["period_days"], INTERNAL["duration_hours"], INTERNAL["depth_ppm"], "snr_proxy", "depth_over_radii_model"]],
            on=INTERNAL["id"],
            how="left",
        )

        merged = merged.sort_values("P(planet)", ascending=False).head(100)
        for _, row in merged.iterrows():
            candidates_payload.append(
                {
                    "id": row[INTERNAL["id"]],
                    "mission": row[INTERNAL["mission"]],
                    "score": _safe_float(row.get("P(planet)")),
                    "predictedClass": row.get("PredictedClass"),
                    "confidence": row.get("Confidence"),
                    "period_days": _safe_float(row.get(INTERNAL["period_days"])),
                    "duration_hours": _safe_float(row.get(INTERNAL["duration_hours"])),
                    "depth_ppm": _safe_float(row.get(INTERNAL["depth_ppm"])),
                    "snr_proxy": _safe_float(row.get("snr_proxy")),
                    "depth_over_radii_model": _safe_float(row.get("depth_over_radii_model")),
                }
            )

    feature_importance_payload = _load_feature_importance(bundle_dir)

    payload = {
        "missionMix": mission_mix_payload,
        "missingness": missing_payload,
        "duplicates": duplicates_payload,
        "histograms": histogram_results,
        "scatter": scatter_payload,
        "outliers": outliers_payload,
        "quantiles": quantile_payload,
        "drift": drift_payload,
        "scoreHistogram": score_hist_payload,
        "prCurve": pr_points,
        "rocCurve": roc_points,
        "calibration": cal_points,
        "missionAcceptance": mission_acceptance_payload,
        "thresholdSnapshots": snapshots,
        "candidates": candidates_payload,
        "featureImportance": feature_importance_payload,
        "shapExample": None,
        "thresholdUsed": threshold_used,
        "perMissionThresholds": per_mission_thr,
    }

    return _sanitize(payload)


# =====================
# Convenience: quick file->predict wrapper
# =====================
def predict_file(path_or_bytes, engine: InferenceEngine, **kwargs) -> Dict[str, Any]:
    df = load_any_csv(path_or_bytes)
    ok, report = validate_df(df)
    try:
        out = engine.predict_df(df, **kwargs)
        out["validator"] = report
        return out
    except Exception as e:
        return {"error": str(e), "validator": report}


# =====================
# Example (you can delete this block in production)
# =====================
if __name__ == "__main__":
    # Quick manual test:
    #   python exo_infer.py /kaggle/working/exo_bundle_v4 /kaggle/working/exo_bundle_v4/demo.csv
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle_dir", nargs="?")
    ap.add_argument("csv_path", nargs="?")
    ap.add_argument("--top_n", type=int, default=None)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--mode", choices=["validate", "predict", "visualize"], default=None,
                    help="Optional CLI helper for the web app")
    ap.add_argument("--per_mission", action="store_true", default=True,
                    help="Use per-mission thresholds from thresholds.json")

    args = ap.parse_args()


    if args.mode == "validate":
        if not args.csv_path:
            raise SystemExit("--mode validate requires csv_path")
        df = load_any_csv(args.csv_path)
        ok, rep = validate_df(df)
        print(json.dumps({"ok": ok, "report": rep}))
    elif args.mode == "predict":
        if not args.bundle_dir or not args.csv_path:
            raise SystemExit("--mode predict requires bundle_dir and csv_path")
        eng = InferenceEngine(args.bundle_dir)
        df = load_any_csv(args.csv_path)
        try:
            out = eng.predict_df(df, threshold=args.thr, top_n=args.top_n, use_per_mission=args.per_mission)
            payload = {
                "threshold_used": out.get("threshold_used"),
                "per_mission_thresholds": out.get("per_mission_thresholds"),
                "metrics": out.get("metrics"),
                "curves": out.get("curves"),
                "preds": out.get("preds").to_dict(orient="records") if isinstance(out.get("preds"), pd.DataFrame) else out.get("preds"),
            }
            def _json_sanitize(obj):
                if isinstance(obj, float):
                    if not np.isfinite(obj):
                        return None
                    return float(obj)
                if isinstance(obj, dict):
                    return {k: _json_sanitize(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_json_sanitize(v) for v in obj]
                return obj

            payload = _json_sanitize(payload)
            print(json.dumps(payload, allow_nan=False))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
    elif args.mode == "visualize":
        if not args.bundle_dir or not args.csv_path:
            raise SystemExit("--mode visualize requires bundle_dir and csv_path")
        training_dir = os.path.join(os.path.dirname(args.bundle_dir), "public", "training-reference")
        if not os.path.exists(training_dir):
            training_dir = os.path.join(os.path.dirname(__file__), "public", "training-reference")
        payload = build_visualization_payload(args.bundle_dir, args.csv_path, training_dir)
        print(json.dumps(payload, allow_nan=False))
    else:
        if not args.bundle_dir or not args.csv_path:
            raise SystemExit("Usage: python exo_infer.py BUNDLE_DIR CSV_PATH [--top_n N | --thr T]")
        eng = InferenceEngine(args.bundle_dir)
        df = load_any_csv(args.csv_path)
        ok, rep = validate_df(df)
        print("Validator:", rep["status"], "| rows:", rep["row_count"])
        try:
            out = eng.predict_df(df, threshold=args.thr, top_n=args.top_n, use_per_mission=args.per_mission)
            print("Used threshold:", out["threshold_used"])
            print(out["preds"].head().to_string(index=False))
            if out["metrics"]:
                print("Metrics:", json.dumps(out["metrics"], indent=2))
        except Exception as e:
            print("ERROR:", str(e))
