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
MISSION_FALLBACK_NOTE = "Unrecognized/empty mission type, defaulting to TESS"
REQUIRED = [INTERNAL["id"], INTERNAL["mission"]]
KNOWN_MISSIONS = {"Kepler","TESS","K2"}

NUM_HINTS = [
    INTERNAL["period_days"], INTERNAL["duration_hours"], INTERNAL["depth_ppm"],
    INTERNAL["planet_radius_re"], INTERNAL["stellar_teff_k"], INTERNAL["stellar_logg_cgs"],
    INTERNAL["stellar_radius_rsun"], INTERNAL["insolation_earth"], INTERNAL["eqt_k"],
    INTERNAL["snr_like"]
]

def _normalize(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _mission_status(raw: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (stringified, missing_mask, invalid_mask) for mission column."""
    mission_str = raw.astype(str).str.strip()
    lower = mission_str.str.lower()
    missing_mask = (
        raw.isna()
        | (mission_str == "")
        | lower.isin({"nan", "none", "null"})
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

    # 3) mission values (if present)
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
# (Contains the original .transform logic, trimmed comments)
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
        # “auto” heuristic
        med = float(np.nanmedian(s.dropna())) if s.notna().any() else np.nan
        if self.k2_trandep_unit == "ppm": return s
        if self.k2_trandep_unit == "percent": return s * 10000.0
        if not np.isfinite(med): return s
        return s*10000.0 if med < 10 else s

    def _harmonize_one(self, df: pd.DataFrame, spec: HarmonizeSpec, mission: str) -> pd.DataFrame:
        out = pd.DataFrame()
        out[INTERNAL["mission"]] = mission
        # ids
        id_col = _find_col(df, spec.src.get("id")) if spec.src.get("id") else None
        out[INTERNAL["id"]] = (df[id_col] if id_col else df.index).astype(str)
        star_col = _find_col(df, spec.src.get("star_id")) if spec.src.get("star_id") else None
        out[INTERNAL["star_id"]] = (df[star_col] if star_col else out[INTERNAL["id"]]).astype(str)
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
            
        # pass through error-widths if present
        ERRW = [
            "period_errw","duration_errw","depth_errw","prad_errw",
            "teff_errw","logg_errw","srad_errw","insol_errw","eqt_errw"
        ]
        for e in ERRW:
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
        # (At inference we rely on stored self.mission_medians/clip/z from training)
        dfs = []
        for mission, (df, spec) in parts.items():
            h = self._harmonize_one(df, spec, mission)
            
            # make sure all trained numeric features exist (avoid KeyError)
            for c in self.numeric_features_:
                if c not in h.columns:
                    h[c] = np.nan


            # Impute + missing flags using training medians
            for c in self.numeric_features_:
                miss = h[c].isna()
                h[c + "__missing"] = miss.astype(np.int8)
                med = self.mission_medians.get(mission, pd.Series(dtype=float)).get(c, np.nan)
                if np.isnan(med):
                    # fallback to across-mission median if needed
                    all_meds = [m.get(c, np.nan) for m in self.mission_medians.values()]
                    med = np.nanmedian(all_meds) if len(all_meds) else 0.0
                h.loc[miss, c] = med

            # Basics
            pdays  = h[INTERNAL["period_days"]].clip(lower=1e-6)
            dhours = h[INTERNAL["duration_hours"]].clip(lower=1e-6)
            dppm   = h[INTERNAL["depth_ppm"]].clip(lower=1e-9)
            h["log_period"]     = np.log1p(pdays)
            h["log_duration_h"] = np.log1p(dhours)
            h["log_depth_ppm"]  = np.log1p(dppm)
            h["duty_cycle"]     = (dhours / (pdays * 24.0)).clip(0, 1)
            h["depth_per_hour"] = (dppm / dhours)
            
            # ---- physics ratios (match training) ----
            Rp_re  = h[INTERNAL["planet_radius_re"]].clip(lower=1e-6)
            Rs_rsun= h[INTERNAL["stellar_radius_rsun"]].clip(lower=1e-6)
            dppm   = h[INTERNAL["depth_ppm"]].clip(lower=1e-9)

            Rs_re  = Rs_rsun * self.RAD_EARTH_PER_SUN
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

        # Per-mission winsorize + z, using stored params
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

        # One-hot mission
        mission_ohe = pd.get_dummies(merged[INTERNAL["mission"]], prefix="mission", dummy_na=False)
        merged = pd.concat([merged, mission_ohe], axis=1)
        for mcol in ["mission_Kepler","mission_TESS","mission_K2"]:
            if mcol not in merged.columns:
                merged[mcol] = 0
                
        
        return merged

# Make unpickling robust if preprocess.pkl was dumped from a notebook (__main__)
# We register our ExoPreprocessor class under __main__ so joblib can resolve it.
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
        self._mission_default_ids: Set[str] = set()

        # Discover base models (family -> stage -> seed -> [fold paths])
        self.registry = {fam: {"A": {}, "B": {}} for fam in ["lgb", "cat", "xgb"]}

        for fname in os.listdir(bundle_dir):
            path = os.path.join(bundle_dir, fname)
            if not (os.path.isfile(path) and fname.startswith("model_")):
                continue
            # expected: model_<fam>_<stage>_seed<k>_fold<j>.<ext>
            m = re.search(r"^model_(lgb|cat|xgb)_(A|B)_seed(\d+)_fold(\d+)\.", fname)
            if not m:
                continue
            fam = m.group(1)
            stage = m.group(2)
            seed = int(m.group(3))
            self.registry[fam][stage].setdefault(seed, []).append(path)

        # deterministic order used by meta
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

    # ---------- Harmonize incoming df into model features ----------
    def _harmonize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Accepts a single merged dataframe with 'mission' column in {Kepler, TESS, K2}.
        We split by mission and feed each split through the preprocessor using
        an identity spec (i.e., our "internal" column names).
        """
        # Identity map: tells the harmonizer to read columns with our INTERNAL names if present
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

        # ---- Normalize/guess mission BEFORE splitting (REPLACE PREVIOUS BLOCK) ----
        self._mission_default_ids = set()
        parts = {}
        if INTERNAL["mission"] in df.columns:
            df = df.copy()
            raw = df[INTERNAL["mission"]].astype(str).str.strip()

            # strict normalize to exactly: Kepler / TESS / K2
            norm = raw.copy()
            low  = raw.str.lower()

            norm[low == "kepler"] = "Kepler"
            norm[low == "tess"]   = "TESS"
            norm[low == "k2"]     = "K2"
            
            missing_like = low.isin({"", "nan", "none", "unknown"})
            if missing_like.any():
                fallback_ids = df.loc[missing_like, INTERNAL["id"]].astype(str).tolist()
                self._mission_default_ids.update(fallback_ids)
                norm.loc[missing_like] = "TESS"

            # guess from id when still not recognized
            def _guess_from_id(x):
                s = str(x).upper()
                if s.startswith("EPIC") or "K2-" in s: return "K2"
                if "TOI" in s or "TIC" in s:          return "TESS"
                # common Kepler styles: KOI #######, pure 7–9 digit IDs
                if s.startswith("KOI") or (s.isdigit() and 7 <= len(s) <= 9):
                    return "Kepler"
                return "TESS"  # safe default

            bad = ~norm.isin(KNOWN_MISSIONS)
            if bad.any():
                norm.loc[bad] = df.loc[bad, INTERNAL["id"]].map(_guess_from_id)
                
            still_bad = ~norm.isin(KNOWN_MISSIONS)
            if still_bad.any():
                fallback_ids = df.loc[still_bad, INTERNAL["id"]].astype(str).tolist()
                self._mission_default_ids.update(fallback_ids)
                norm.loc[still_bad] = "TESS"

            df[INTERNAL["mission"]] = norm

            # now split by normalized mission (no Unknown bucket)
            for mission in KNOWN_MISSIONS:
                mask = norm == mission
                sub = df[mask]
                if len(sub):
                    parts[mission] = (sub.copy(), HarmonizeSpec(src=SELF_SRC, label_map={
                        PLANET: PLANET, CAND: CAND, FP: FP,
                        PLANET.upper(): PLANET, CAND.upper(): CAND, FP.upper(): FP
                    }))
        else:
            # fallback: assume TESS if mission column absent
            parts = {"TESS": (df.copy(), HarmonizeSpec(src=SELF_SRC, label_map={}))}
            if INTERNAL["id"] in df.columns:
                self._mission_default_ids.update(df[INTERNAL["id"]].astype(str).tolist())

        # if nothing matched (extreme edge case), default to TESS
        if not parts:
            parts = {"TESS": (df.copy(), HarmonizeSpec(src=SELF_SRC, label_map={}))}
            if INTERNAL["id"] in df.columns:
                self._mission_default_ids.update(df[INTERNAL["id"]].astype(str).tolist())


        data = self.pre.transform(parts)
        # Keep canonical missions; do NOT demote to NaN (we normalized before splitting)
        # keep canonical missions; we already normalized before splitting
        if INTERNAL["mission"] in data.columns:
            s = data[INTERNAL["mission"]].astype(str).str.strip()
            s = s.replace({"kepler":"Kepler", "tess":"TESS", "k2":"K2"})
            low = s.str.lower()
            missing_like = low.isin({"nan","none","","unknown"})
            if missing_like.any():
                if INTERNAL["id"] in data.columns:
                    self._mission_default_ids.update(data.loc[missing_like, INTERNAL["id"]].astype(str).tolist())
                s.loc[missing_like] = "TESS"
            data[INTERNAL["mission"]] = s
        else:
            # No mission column made it through preprocessing; fabricate a TESS default.
            mission_stub = pd.Series("TESS", index=data.index)
            if INTERNAL["id"] in data.columns:
                self._mission_default_ids.update(data[INTERNAL["id"]].astype(str).tolist())
            data[INTERNAL["mission"]] = mission_stub


        # Drop any leaky columns in case user sent them; training already removed them
        leaky = [c for c in data.columns if any(s in c.lower() for s in LEAKY_SUBSTRINGS)]
        keep = [c for c in data.columns if (c not in leaky) or (c == INTERNAL["label"])]
        return data[keep]

    # ---------- Family prediction helpers ----------
    def _predict_family(self, fam: str, stage: str, X: pd.DataFrame) -> List[np.ndarray]:
        outs = []
        entries = sorted([p for p in self.families.get(fam, {}).get(stage, [])])
        for p in entries:
            if fam == "lgb":
                booster = lgb.Booster(model_file=p)
                P = booster.predict(X, num_iteration=booster.best_iteration)
                if self.two_stage:
                    if stage == "A":
                        proba = np.vstack([np.zeros(len(X)), np.zeros(len(X)), P]).T
                    else:
                        proba = np.vstack([np.zeros(len(X)), P, np.zeros(len(X))]).T
                else:
                    proba = P
            elif fam == "cat" and CATBOOST_AVAILABLE:
                clf = CatBoostClassifier()
                clf.load_model(p)
                P = np.array(clf.predict_proba(X))
                if self.two_stage:
                    if stage == "A":
                        proba = np.vstack([np.zeros(len(X)), np.zeros(len(X)), P[:,1]]).T
                    else:
                        proba = np.vstack([np.zeros(len(X)), P[:,1], np.zeros(len(X))]).T
                else:
                    proba = P
            elif fam == "xgb" and XGB_AVAILABLE:
                dm = xgb.DMatrix(X)
                bst = xgb.Booster(model_file=p)
                P = bst.predict(dm)
                if self.two_stage:
                    if stage == "A":
                        proba = np.vstack([np.zeros(len(X)), np.zeros(len(X)), P]).T
                    else:
                        proba = np.vstack([np.zeros(len(X)), P, np.zeros(len(X))]).T
                else:
                    proba = P
            else:
                continue  # skip unavailable family
            outs.append(proba)
        return outs

    def _predict_one_path(self, fam: str, stage: str, path: str, X: pd.DataFrame) -> np.ndarray:
        """Return binary probability (n,) for the given model path."""
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
        """
        For one family, return {seed: tri_probs(n,3)} averaged over folds.
        Two-stage: combine A(planet) & B(candidate) into 3-way probs:
        Pplanet = pA
        Pcand   = (1-pA)*pB
        Pfp     = (1-pA)*(1-pB)
        One-stage: if a family emitted tri-prob directly, use it; else map binary to tri.
        """
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
                    P[:, self.planet_idx]        = pA
                    P[:, LABEL2ID[CAND]]         = (1.0 - pA) * pB
                    P[:, LABEL2ID[FP]]           = (1.0 - pA) * (1.0 - pB)
                    tri_list.append(P)
                out[seed] = np.mean(tri_list, axis=0)
            else:
                # one-stage: if models emit tri-prob we would use it; in this project they’re binary → map to tri
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
          - preds: DataFrame [id, mission, P(planet), PredictedClass, Confidence]
          - threshold_used, per_mission_thresholds
          - metrics (if labels exist), curves (if labels exist)
        """
        # Validate upstream (UI should gate; we re-check quickly)
        ok, vrep = validate_df(df_in)
        if not ok and vrep["status"] == "Cannot score":
            return {"error": "Cannot score", "validator": vrep}

        # Harmonize → model matrix
        df = df_in.copy()
        # Strip leaky columns (safe-guard)
        leak_cols = [c for c in df.columns if any(s in c.lower() for s in LEAKY_SUBSTRINGS)]
        df = df.drop(columns=leak_cols, errors="ignore")
        data = self._harmonize_df(df)
        X = data[self.features].copy()
        n = len(X)

        # --------- Family probs (for blend) AND per-seed probs (for meta) ----------
        fam_probs_blend: Dict[str, Optional[np.ndarray]] = {"lgb": None, "cat": None, "xgb": None}
        per_seed: Dict[str, Dict[int, np.ndarray]] = {}

        for fam in self.family_order:
            seed_map = self._family_seed_tri_probs(fam, X)  # {seed: (n,3)}
            if seed_map:
                per_seed[fam] = seed_map
                # blend = average across seeds
                fam_probs_blend[fam] = np.mean(list(seed_map.values()), axis=0)

        enabled_fams_for_blend = [f for f,v in fam_probs_blend.items() if v is not None]

        # --------- Try exact 27-feature meta (3 fams × 3 seeds × 3 classes) ----------
        can_meta = (
            self.meta is not None
            and all(f in per_seed and len(per_seed[f]) >= 3 for f in self.family_order)
        )

        if can_meta:
            blocks = []
            for fam in self.family_order:          # family-major
                for seed in self.seed_order[:3]:   # 3 seeds in order
                    Ptri = per_seed[fam][seed]     # (n,3) in [FP, CAND, PLANET] order
                    blocks.append(Ptri)
            Z = np.concatenate(blocks, axis=1)     # (n, 27)

            expected = getattr(self.meta, "n_features_in_", Z.shape[1])
            if Z.shape[1] == expected:
                probs = self.meta.predict_proba(Z)
            else:
                can_meta = False

        if not can_meta:
            # Plain weighted blend across families
            probs = np.zeros((n, 3), dtype=float)
            W = self.bw  # e.g. {"lgbm": 1.0, "cat": 0.0, "xgb": 0.0}
            for fam in enabled_fams_for_blend:
                if fam == "lgb": probs += W.get("lgbm", 0.0) * fam_probs_blend[fam]
                if fam == "cat": probs += W.get("cat",  0.0) * fam_probs_blend[fam]
                if fam == "xgb": probs += W.get("xgb",  0.0) * fam_probs_blend[fam]


        # P(planet) + optional Platt calibration
        # Ensure finite probs before using them
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

        # P(planet) + optional Platt calibration
        p_planet = np.nan_to_num(probs[:, self.planet_idx], nan=0.0, posinf=1.0, neginf=0.0)

        if self.platt is not None:
            p_planet = self.platt.predict_proba(p_planet.reshape(-1,1))[:,1]
            
        # Final clamp/sanitize
        p_planet = np.clip(np.nan_to_num(p_planet, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


        # Determine threshold
        per_mission_thr = self.thr.get("per_mission", {}) if use_per_mission else {}
        thr_used: float
        if top_n is not None and top_n >= 1:
            # pick threshold to roughly yield N positives globally
            thr_used = _threshold_for_topn(p_planet, top_n)
        elif threshold is not None:
            thr_used = float(threshold)
        else:
            thr_used = float(self.thr.get("planet_recommended", 0.5))

        # Class predictions for table
        y_idx = np.argmax(probs, axis=1)
        y_name = [self.class_names[i] for i in y_idx]
        def conf_tag(p): return "High" if p>=0.8 else ("Medium" if p>=0.6 else "Low")
        
        missions = data[INTERNAL["mission"]].astype(str) if INTERNAL["mission"] in data else pd.Series("TESS", index=data.index)
        missions = missions.str.strip()
        missions = missions.replace({"kepler": "Kepler", "tess": "TESS", "k2": "K2"})
        lower = missions.str.lower()
        missing_like = lower.isin({"nan", "none", "", "unknown"})
        if missing_like.any():
            missing_ids = data.loc[missing_like, INTERNAL["id"]].astype(str).tolist()
            self._mission_default_ids.update(missing_ids)
            missions.loc[missing_like] = "TESS"

        preds = pd.DataFrame({
            INTERNAL["id"]: data[INTERNAL["id"]].values,
            INTERNAL["mission"]: missions.values,
            "P(planet)": p_planet,
            "PredictedClass": y_name,
            "Confidence": [conf_tag(p) for p in p_planet],
        })
        
        # ADD after creating preds DataFrame
        preds["IsPlanet"] = (preds["P(planet)"] >= thr_used).astype(int)
        preds["BinaryClass"] = np.where(preds["IsPlanet"] == 1, "Planet", "Not Planet")
        preds["BinaryConfidence"] = np.where(preds["IsPlanet"] == 1,
                                            preds["P(planet)"],
                                            1.0 - preds["P(planet)"])

        
        # Make mission printable and remove any NaN in preds
        preds[INTERNAL["mission"]] = preds[INTERNAL["mission"]].fillna("TESS")
        if self._mission_default_ids:
            note_col = preds[INTERNAL["id"]].astype(str).map(
                lambda x: MISSION_FALLBACK_NOTE if x in self._mission_default_ids else None
            )
            if note_col.notna().any():
                preds["MissionNote"] = note_col
        preds = preds.replace({np.nan: None})

        # If per-mission thresholding requested, add a binary tag computed per-row
        if per_mission_thr:
            pos_mask = []
            for m, p in zip(preds[INTERNAL["mission"]].values, preds["P(planet)"].values):
                t = per_mission_thr.get(m, thr_used)
                pos_mask.append(int(p >= t))
            preds["Planet>=Thr"] = pos_mask
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
    kth = np.partition(p, -n_pos)[-n_pos]  # value so that at least n_pos are >= kth
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
# Convenience: quick file->predict wrapper
# =====================
def predict_file(path_or_bytes, engine: InferenceEngine, **kwargs) -> Dict[str, Any]:
    df = load_any_csv(path_or_bytes)
    ok, report = validate_df(df)
    if not ok and report["status"] == "Cannot score":
        return {"error": "Cannot score", "validator": report}
    out = engine.predict_df(df, **kwargs)
    out["validator"] = report
    return out

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
    ap.add_argument("--mode", choices=["validate", "predict"], default=None,
                    help="Optional CLI helper for the web app")
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
        out = eng.predict_df(df, threshold=args.thr, top_n=args.top_n)
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

    else:
        if not args.bundle_dir or not args.csv_path:
            raise SystemExit("Usage: python exo_infer.py BUNDLE_DIR CSV_PATH [--top_n N | --thr T]")
        eng = InferenceEngine(args.bundle_dir)
        df = load_any_csv(args.csv_path)
        ok, rep = validate_df(df)
        print("Validator:", rep["status"], "| rows:", rep["row_count"])
        out = eng.predict_df(df, threshold=args.thr, top_n=args.top_n)
        print("Used threshold:", out["threshold_used"])
        print(out["preds"].head().to_string(index=False))
        if out["metrics"]:
            print("Metrics:", json.dumps(out["metrics"], indent=2))
