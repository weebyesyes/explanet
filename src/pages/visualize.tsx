import { useMemo, useState, type ReactNode } from "react";
import styles from "../styles/pages/visualize.module.css";

type MissionKey = "Kepler" | "TESS" | "K2";

type MissionMixDatum = {
  mission: MissionKey;
  count: number;
};

type MissingnessDatum = {
  feature: string;
  mission: MissionKey;
  missingRate: number;
};

type HistogramDatum = {
  bin: string;
  missions: Record<MissionKey, number>;
};

type HistogramConfig = {
  feature: string;
  label: string;
  logScale?: boolean;
  bins: HistogramDatum[];
};

type ScatterDatum = {
  id: string;
  x: number;
  y: number;
  mission: MissionKey;
};

type ScatterConfig = {
  id: string;
  label: string;
  xLabel: string;
  yLabel: string;
  logX?: boolean;
  logY?: boolean;
  points: ScatterDatum[];
};

type OutlierRow = {
  id: string;
  mission: MissionKey;
  feature: string;
  zScore: number;
  value: number;
  units: string;
};

type QuantileRibbon = {
  feature: string;
  mission: MissionKey;
  logScale?: boolean;
  train: Array<{ value: number; p10: number; p90: number; median: number }>;
  scoring: Array<{ value: number; density: number }>;
};

type DriftSnapshot = {
  feature: string;
  mission: MissionKey;
  delta: number;
  kl: number;
  coverage: number;
};

type ThresholdSnapshot = {
  threshold: number;
  prAuc: number;
  rocAuc: number;
  f1: number;
  recall: number;
  brier: number;
  matrix: { tp: number; fp: number; fn: number; tn: number };
};

type CurvePoint = { x: number; y: number };

type MissionAcceptance = {
  mission: MissionKey;
  accepted: number;
  rejected: number;
};

type CandidateRow = {
  id: string;
  mission: MissionKey;
  score: number;
  predictedClass: string;
  confidence: string;
  period_days: number;
  duration_hours: number;
  depth_ppm: number;
  snr_proxy: number;
  depth_over_radii_model: number;
};

type FeatureImportanceRow = {
  feature: string;
  gain: number;
};

type ShapRow = {
  feature: string;
  contribution: number;
};

type VisualizeDataset = {
  missionMix: MissionMixDatum[];
  missingness: MissingnessDatum[];
  histograms: HistogramConfig[];
  scatter: ScatterConfig[];
  outliers: OutlierRow[];
  quantiles: QuantileRibbon[];
  drift: DriftSnapshot[];
  scoreHistogram: Array<{
    lower: number;
    upper: number;
    total: number;
    missions: Record<MissionKey, number>;
  }>;
  prCurve: CurvePoint[];
  rocCurve: CurvePoint[];
  calibration: CurvePoint[];
  missionAcceptance: MissionAcceptance[];
  thresholdSnapshots: ThresholdSnapshot[];
  candidates: CandidateRow[];
  featureImportance: FeatureImportanceRow[];
  shapExample: { id: string; mission: MissionKey; contributions: ShapRow[] };
};

const missionPalette: Record<MissionKey, string> = {
  Kepler: "#00dfd8",
  TESS: "#df6ac9",
  K2: "#7f8cff",
};

const sampleData: VisualizeDataset = {
  missionMix: [
    { mission: "Kepler", count: 412 },
    { mission: "TESS", count: 308 },
    { mission: "K2", count: 184 },
  ],
  missingness: [
    { feature: "period_days__missing", mission: "Kepler", missingRate: 0.01 },
    { feature: "period_days__missing", mission: "TESS", missingRate: 0.02 },
    { feature: "period_days__missing", mission: "K2", missingRate: 0.03 },
    { feature: "duration_hours__missing", mission: "Kepler", missingRate: 0.03 },
    { feature: "duration_hours__missing", mission: "TESS", missingRate: 0.04 },
    { feature: "duration_hours__missing", mission: "K2", missingRate: 0.05 },
    { feature: "depth_ppm__missing", mission: "Kepler", missingRate: 0.04 },
    { feature: "depth_ppm__missing", mission: "TESS", missingRate: 0.03 },
    { feature: "depth_ppm__missing", mission: "K2", missingRate: 0.06 },
    { feature: "planet_radius_re__missing", mission: "Kepler", missingRate: 0.06 },
    { feature: "planet_radius_re__missing", mission: "TESS", missingRate: 0.07 },
    { feature: "planet_radius_re__missing", mission: "K2", missingRate: 0.09 },
    { feature: "stellar_teff_k__missing", mission: "Kepler", missingRate: 0.08 },
    { feature: "stellar_teff_k__missing", mission: "TESS", missingRate: 0.12 },
    { feature: "stellar_teff_k__missing", mission: "K2", missingRate: 0.15 },
    { feature: "stellar_radius_rsun__missing", mission: "Kepler", missingRate: 0.05 },
    { feature: "stellar_radius_rsun__missing", mission: "TESS", missingRate: 0.06 },
    { feature: "stellar_radius_rsun__missing", mission: "K2", missingRate: 0.08 },
    { feature: "snr_like__missing", mission: "Kepler", missingRate: 0.02 },
    { feature: "snr_like__missing", mission: "TESS", missingRate: 0.02 },
    { feature: "snr_like__missing", mission: "K2", missingRate: 0.04 },
    { feature: "duty_cycle__missing", mission: "Kepler", missingRate: 0.01 },
    { feature: "duty_cycle__missing", mission: "TESS", missingRate: 0.03 },
    { feature: "duty_cycle__missing", mission: "K2", missingRate: 0.03 },
    { feature: "depth_per_hour__missing", mission: "Kepler", missingRate: 0.02 },
    { feature: "depth_per_hour__missing", mission: "TESS", missingRate: 0.04 },
    { feature: "depth_per_hour__missing", mission: "K2", missingRate: 0.05 },
    { feature: "depth_over_radii_model__missing", mission: "Kepler", missingRate: 0.02 },
    { feature: "depth_over_radii_model__missing", mission: "TESS", missingRate: 0.03 },
    { feature: "depth_over_radii_model__missing", mission: "K2", missingRate: 0.04 },
    { feature: "snr_proxy__missing", mission: "Kepler", missingRate: 0.03 },
    { feature: "snr_proxy__missing", mission: "TESS", missingRate: 0.04 },
    { feature: "snr_proxy__missing", mission: "K2", missingRate: 0.05 },
    { feature: "eqt_over_teff__missing", mission: "Kepler", missingRate: 0.01 },
    { feature: "eqt_over_teff__missing", mission: "TESS", missingRate: 0.02 },
    { feature: "eqt_over_teff__missing", mission: "K2", missingRate: 0.03 },
  ],
  histograms: [
    {
      feature: "period_days",
      label: "Orbital period (days)",
      logScale: true,
      bins: [
        { bin: "0.3-1", missions: { Kepler: 24, TESS: 30, K2: 12 } },
        { bin: "1-3", missions: { Kepler: 48, TESS: 52, K2: 23 } },
        { bin: "3-10", missions: { Kepler: 66, TESS: 48, K2: 35 } },
        { bin: "10-30", missions: { Kepler: 42, TESS: 36, K2: 28 } },
        { bin: "30-120", missions: { Kepler: 32, TESS: 18, K2: 18 } },
      ],
    },
    {
      feature: "duration_hours",
      label: "Transit duration (hours)",
      logScale: true,
      bins: [
        { bin: "0.5-1", missions: { Kepler: 18, TESS: 20, K2: 10 } },
        { bin: "1-2", missions: { Kepler: 36, TESS: 42, K2: 20 } },
        { bin: "2-4", missions: { Kepler: 58, TESS: 47, K2: 28 } },
        { bin: "4-8", missions: { Kepler: 40, TESS: 32, K2: 16 } },
        { bin: "8-16", missions: { Kepler: 14, TESS: 10, K2: 8 } },
      ],
    },
    {
      feature: "depth_ppm",
      label: "Transit depth (ppm)",
      logScale: true,
      bins: [
        { bin: "10-50", missions: { Kepler: 20, TESS: 38, K2: 18 } },
        { bin: "50-120", missions: { Kepler: 58, TESS: 70, K2: 32 } },
        { bin: "120-400", missions: { Kepler: 70, TESS: 54, K2: 40 } },
        { bin: "400-1200", missions: { Kepler: 46, TESS: 32, K2: 26 } },
        { bin: "1200-4000", missions: { Kepler: 20, TESS: 14, K2: 12 } },
      ],
    },
    {
      feature: "planet_radius_re",
      label: "Planet radius (R⊕)",
      bins: [
        { bin: "0-1", missions: { Kepler: 28, TESS: 32, K2: 20 } },
        { bin: "1-2", missions: { Kepler: 74, TESS: 80, K2: 42 } },
        { bin: "2-4", missions: { Kepler: 62, TESS: 50, K2: 34 } },
        { bin: "4-8", missions: { Kepler: 36, TESS: 28, K2: 22 } },
        { bin: "8-16", missions: { Kepler: 18, TESS: 12, K2: 12 } },
      ],
    },
    {
      feature: "stellar_teff_k",
      label: "Stellar Teff (K)",
      bins: [
        { bin: "3000-4000", missions: { Kepler: 20, TESS: 18, K2: 10 } },
        { bin: "4000-5000", missions: { Kepler: 52, TESS: 48, K2: 26 } },
        { bin: "5000-6000", missions: { Kepler: 96, TESS: 88, K2: 48 } },
        { bin: "6000-7000", missions: { Kepler: 54, TESS: 50, K2: 32 } },
        { bin: "7000-8000", missions: { Kepler: 18, TESS: 12, K2: 8 } },
      ],
    },
    {
      feature: "stellar_radius_rsun",
      label: "Stellar radius (R☉)",
      bins: [
        { bin: "0.2-0.6", missions: { Kepler: 32, TESS: 22, K2: 14 } },
        { bin: "0.6-1.0", missions: { Kepler: 86, TESS: 78, K2: 40 } },
        { bin: "1.0-1.4", missions: { Kepler: 62, TESS: 60, K2: 34 } },
        { bin: "1.4-2.0", missions: { Kepler: 38, TESS: 28, K2: 20 } },
        { bin: "2.0-3.5", missions: { Kepler: 18, TESS: 12, K2: 8 } },
      ],
    },
    {
      feature: "snr_like",
      label: "Pipeline SNR",
      logScale: true,
      bins: [
        { bin: "2-4", missions: { Kepler: 30, TESS: 26, K2: 18 } },
        { bin: "4-6", missions: { Kepler: 58, TESS: 52, K2: 28 } },
        { bin: "6-10", missions: { Kepler: 72, TESS: 68, K2: 38 } },
        { bin: "10-20", missions: { Kepler: 52, TESS: 40, K2: 26 } },
        { bin: "20-40", missions: { Kepler: 28, TESS: 18, K2: 12 } },
      ],
    },
    {
      feature: "duty_cycle",
      label: "Duty cycle",
      bins: [
        { bin: "0-0.2", missions: { Kepler: 24, TESS: 26, K2: 16 } },
        { bin: "0.2-0.4", missions: { Kepler: 46, TESS: 54, K2: 28 } },
        { bin: "0.4-0.6", missions: { Kepler: 72, TESS: 60, K2: 38 } },
        { bin: "0.6-0.8", missions: { Kepler: 54, TESS: 42, K2: 24 } },
        { bin: "0.8-1.0", missions: { Kepler: 32, TESS: 24, K2: 16 } },
      ],
    },
    {
      feature: "depth_per_hour",
      label: "Depth per hour",
      logScale: true,
      bins: [
        { bin: "10-30", missions: { Kepler: 32, TESS: 42, K2: 22 } },
        { bin: "30-90", missions: { Kepler: 68, TESS: 60, K2: 36 } },
        { bin: "90-270", missions: { Kepler: 72, TESS: 54, K2: 38 } },
        { bin: "270-800", missions: { Kepler: 48, TESS: 34, K2: 22 } },
        { bin: "800-2000", missions: { Kepler: 24, TESS: 18, K2: 12 } },
      ],
    },
    {
      feature: "depth_over_radii_model",
      label: "Depth / radii model",
      logScale: true,
      bins: [
        { bin: "0.1-0.3", missions: { Kepler: 38, TESS: 42, K2: 24 } },
        { bin: "0.3-0.6", missions: { Kepler: 76, TESS: 64, K2: 38 } },
        { bin: "0.6-1.2", missions: { Kepler: 70, TESS: 54, K2: 34 } },
        { bin: "1.2-2.5", missions: { Kepler: 44, TESS: 32, K2: 20 } },
        { bin: "2.5-5.0", missions: { Kepler: 30, TESS: 22, K2: 14 } },
      ],
    },
    {
      feature: "snr_proxy",
      label: "Proxy SNR",
      logScale: true,
      bins: [
        { bin: "1-3", missions: { Kepler: 24, TESS: 26, K2: 16 } },
        { bin: "3-6", missions: { Kepler: 56, TESS: 54, K2: 32 } },
        { bin: "6-12", missions: { Kepler: 82, TESS: 70, K2: 40 } },
        { bin: "12-24", missions: { Kepler: 54, TESS: 40, K2: 26 } },
        { bin: "24-48", missions: { Kepler: 32, TESS: 20, K2: 16 } },
      ],
    },
    {
      feature: "eqt_over_teff",
      label: "Equilibrium/Teff",
      bins: [
        { bin: "0-0.1", missions: { Kepler: 22, TESS: 28, K2: 14 } },
        { bin: "0.1-0.2", missions: { Kepler: 58, TESS: 62, K2: 30 } },
        { bin: "0.2-0.3", missions: { Kepler: 84, TESS: 70, K2: 42 } },
        { bin: "0.3-0.4", missions: { Kepler: 52, TESS: 40, K2: 26 } },
        { bin: "0.4-0.5", missions: { Kepler: 32, TESS: 24, K2: 18 } },
      ],
    },
  ],
  scatter: [
    {
      id: "period_depth",
      label: "Period vs depth",
      xLabel: "Period (days)",
      yLabel: "Depth (ppm)",
      logX: true,
      logY: true,
      points: Array.from({ length: 60 }).map((_, idx) => {
        const mission: MissionKey = idx % 3 === 0 ? "Kepler" : idx % 3 === 1 ? "TESS" : "K2";
        return {
          id: `pd-${idx}`,
          mission,
          x: Math.exp(Math.log(0.4) + Math.random() * Math.log(300)),
          y: Math.exp(Math.log(20) + Math.random() * Math.log(5000)),
        };
      }),
    },
    {
      id: "period_duration",
      label: "Period vs duration",
      xLabel: "Period (days)",
      yLabel: "Duration (hours)",
      logX: true,
      logY: true,
      points: Array.from({ length: 60 }).map((_, idx) => {
        const mission: MissionKey = idx % 3 === 0 ? "Kepler" : idx % 3 === 1 ? "TESS" : "K2";
        const period = Math.exp(Math.log(0.5) + Math.random() * Math.log(200));
        return {
          id: `pdur-${idx}`,
          mission,
          x: period,
          y: Math.pow(period, 0.3) * (1 + Math.random()),
        };
      }),
    },
    {
      id: "radius_depth",
      label: "Radius vs depth",
      xLabel: "Planet radius (R⊕)",
      yLabel: "Depth (ppm)",
      logY: true,
      points: Array.from({ length: 60 }).map((_, idx) => {
        const mission: MissionKey = idx % 3 === 0 ? "Kepler" : idx % 3 === 1 ? "TESS" : "K2";
        const radius = Math.random() * 10 + 0.5;
        return {
          id: `rad-${idx}`,
          mission,
          x: radius,
          y: Math.exp(Math.log(40) + Math.random() * Math.log(5000)) * (radius / 3),
        };
      }),
    },
    {
      id: "teff_eqt",
      label: "Teff vs equilibrium",
      xLabel: "Stellar Teff (K)",
      yLabel: "Eqt (K)",
      points: Array.from({ length: 60 }).map((_, idx) => {
        const mission: MissionKey = idx % 3 === 0 ? "Kepler" : idx % 3 === 1 ? "TESS" : "K2";
        const teff = 3200 + Math.random() * 3500;
        return {
          id: `teq-${idx}`,
          mission,
          x: teff,
          y: teff * (0.1 + Math.random() * 0.25),
        };
      }),
    },
  ],
  outliers: [
    { id: "KIC-1234567", mission: "Kepler", feature: "depth_ppm", zScore: 4.2, value: 18500, units: "ppm" },
    { id: "KIC-9876543", mission: "Kepler", feature: "planet_radius_re", zScore: 3.8, value: 18.2, units: "R⊕" },
    { id: "TIC-445566", mission: "TESS", feature: "stellar_teff_k", zScore: -3.4, value: 3050, units: "K" },
    { id: "EPIC-20123456", mission: "K2", feature: "depth_over_radii_model", zScore: 3.5, value: 4.8, units: "ratio" },
    { id: "TIC-778899", mission: "TESS", feature: "snr_proxy", zScore: 3.3, value: 48, units: "" },
    { id: "KIC-24681012", mission: "Kepler", feature: "duty_cycle", zScore: -3.1, value: 0.05, units: "fraction" },
    { id: "EPIC-20202020", mission: "K2", feature: "stellar_teff_k", zScore: 3.2, value: 7600, units: "K" },
    { id: "TIC-112233", mission: "TESS", feature: "depth_ppm", zScore: -3.4, value: 35, units: "ppm" },
    { id: "KIC-56473829", mission: "Kepler", feature: "snr_proxy", zScore: 3.0, value: 46, units: "" },
    { id: "EPIC-99887766", mission: "K2", feature: "duration_hours", zScore: 3.1, value: 26, units: "h" },
  ],
  quantiles: [
    {
      feature: "period_days",
      mission: "Kepler",
      logScale: true,
      train: [
        { value: 0.5, p10: 0.4, p90: 2.4, median: 1.1 },
        { value: 2, p10: 0.8, p90: 6.2, median: 3.1 },
        { value: 10, p10: 2.5, p90: 24, median: 10.2 },
        { value: 40, p10: 6.8, p90: 70, median: 32 },
      ],
      scoring: [
        { value: 0.5, density: 8 },
        { value: 2, density: 20 },
        { value: 10, density: 24 },
        { value: 40, density: 12 },
      ],
    },
    {
      feature: "period_days",
      mission: "TESS",
      logScale: true,
      train: [
        { value: 0.5, p10: 0.4, p90: 1.8, median: 0.8 },
        { value: 2, p10: 0.7, p90: 4.8, median: 2.2 },
        { value: 10, p10: 1.6, p90: 14, median: 6.8 },
        { value: 40, p10: 3.5, p90: 45, median: 18 },
      ],
      scoring: [
        { value: 0.5, density: 10 },
        { value: 2, density: 26 },
        { value: 10, density: 18 },
        { value: 40, density: 6 },
      ],
    },
    {
      feature: "period_days",
      mission: "K2",
      logScale: true,
      train: [
        { value: 0.5, p10: 0.3, p90: 2.0, median: 0.9 },
        { value: 2, p10: 0.6, p90: 5.2, median: 2.6 },
        { value: 10, p10: 1.8, p90: 16, median: 7.2 },
        { value: 40, p10: 3.2, p90: 48, median: 20 },
      ],
      scoring: [
        { value: 0.5, density: 6 },
        { value: 2, density: 16 },
        { value: 10, density: 14 },
        { value: 40, density: 8 },
      ],
    },
    {
      feature: "depth_ppm",
      mission: "Kepler",
      logScale: true,
      train: [
        { value: 30, p10: 20, p90: 160, median: 85 },
        { value: 120, p10: 60, p90: 420, median: 220 },
        { value: 400, p10: 180, p90: 1200, median: 540 },
        { value: 1200, p10: 320, p90: 3600, median: 1600 },
      ],
      scoring: [
        { value: 30, density: 14 },
        { value: 120, density: 28 },
        { value: 400, density: 22 },
        { value: 1200, density: 12 },
      ],
    },
  ],
  drift: [
    { feature: "period_days", mission: "Kepler", delta: -0.3, kl: 0.08, coverage: 0.92 },
    { feature: "period_days", mission: "TESS", delta: 0.4, kl: 0.11, coverage: 0.88 },
    { feature: "period_days", mission: "K2", delta: 0.1, kl: 0.06, coverage: 0.9 },
    { feature: "depth_ppm", mission: "Kepler", delta: -0.1, kl: 0.07, coverage: 0.94 },
    { feature: "depth_ppm", mission: "TESS", delta: 0.8, kl: 0.16, coverage: 0.83 },
    { feature: "depth_ppm", mission: "K2", delta: -0.2, kl: 0.05, coverage: 0.96 },
    { feature: "planet_radius_re", mission: "Kepler", delta: -0.05, kl: 0.04, coverage: 0.95 },
    { feature: "planet_radius_re", mission: "TESS", delta: 0.2, kl: 0.09, coverage: 0.9 },
    { feature: "planet_radius_re", mission: "K2", delta: 0.12, kl: 0.07, coverage: 0.91 },
    { feature: "stellar_teff_k", mission: "Kepler", delta: 0.1, kl: 0.03, coverage: 0.97 },
    { feature: "stellar_teff_k", mission: "TESS", delta: -0.5, kl: 0.05, coverage: 0.93 },
    { feature: "stellar_teff_k", mission: "K2", delta: 0.2, kl: 0.04, coverage: 0.95 },
  ],
  scoreHistogram: [
    { lower: 0.0, upper: 0.1, total: 42, missions: { Kepler: 12, TESS: 20, K2: 10 } },
    { lower: 0.1, upper: 0.2, total: 58, missions: { Kepler: 18, TESS: 28, K2: 12 } },
    { lower: 0.2, upper: 0.3, total: 68, missions: { Kepler: 22, TESS: 32, K2: 14 } },
    { lower: 0.3, upper: 0.4, total: 74, missions: { Kepler: 26, TESS: 34, K2: 14 } },
    { lower: 0.4, upper: 0.5, total: 82, missions: { Kepler: 28, TESS: 38, K2: 16 } },
    { lower: 0.5, upper: 0.6, total: 72, missions: { Kepler: 26, TESS: 30, K2: 16 } },
    { lower: 0.6, upper: 0.7, total: 54, missions: { Kepler: 20, TESS: 22, K2: 12 } },
    { lower: 0.7, upper: 0.8, total: 42, missions: { Kepler: 16, TESS: 18, K2: 8 } },
    { lower: 0.8, upper: 0.9, total: 32, missions: { Kepler: 12, TESS: 14, K2: 6 } },
    { lower: 0.9, upper: 1.0, total: 18, missions: { Kepler: 6, TESS: 8, K2: 4 } },
  ],
  prCurve: [
    { x: 0.0, y: 0.94 },
    { x: 0.1, y: 0.9 },
    { x: 0.2, y: 0.88 },
    { x: 0.3, y: 0.84 },
    { x: 0.4, y: 0.8 },
    { x: 0.5, y: 0.76 },
    { x: 0.6, y: 0.7 },
    { x: 0.7, y: 0.64 },
    { x: 0.8, y: 0.56 },
    { x: 0.9, y: 0.44 },
    { x: 1.0, y: 0.3 },
  ],
  rocCurve: [
    { x: 0.0, y: 0.0 },
    { x: 0.05, y: 0.45 },
    { x: 0.1, y: 0.65 },
    { x: 0.15, y: 0.78 },
    { x: 0.2, y: 0.85 },
    { x: 0.3, y: 0.9 },
    { x: 0.4, y: 0.93 },
    { x: 0.6, y: 0.97 },
    { x: 0.8, y: 0.99 },
    { x: 1.0, y: 1.0 },
  ],
  calibration: [
    { x: 0.05, y: 0.04 },
    { x: 0.15, y: 0.12 },
    { x: 0.25, y: 0.2 },
    { x: 0.35, y: 0.32 },
    { x: 0.45, y: 0.43 },
    { x: 0.55, y: 0.52 },
    { x: 0.65, y: 0.62 },
    { x: 0.75, y: 0.72 },
    { x: 0.85, y: 0.8 },
    { x: 0.95, y: 0.88 },
  ],
  missionAcceptance: [
    { mission: "Kepler", accepted: 86, rejected: 326 },
    { mission: "TESS", accepted: 94, rejected: 214 },
    { mission: "K2", accepted: 52, rejected: 132 },
  ],
  thresholdSnapshots: [
    {
      threshold: 0.3,
      prAuc: 0.86,
      rocAuc: 0.96,
      f1: 0.78,
      recall: 0.9,
      brier: 0.12,
      matrix: { tp: 128, fp: 44, fn: 18, tn: 428 },
    },
    {
      threshold: 0.5,
      prAuc: 0.87,
      rocAuc: 0.97,
      f1: 0.8,
      recall: 0.84,
      brier: 0.1,
      matrix: { tp: 120, fp: 32, fn: 26, tn: 440 },
    },
    {
      threshold: 0.7,
      prAuc: 0.85,
      rocAuc: 0.95,
      f1: 0.76,
      recall: 0.74,
      brier: 0.09,
      matrix: { tp: 106, fp: 20, fn: 40, tn: 452 },
    },
  ],
  candidates: [
    {
      id: "KIC-33445566",
      mission: "Kepler",
      score: 0.94,
      predictedClass: "PC",
      confidence: "High",
      period_days: 12.42,
      duration_hours: 4.2,
      depth_ppm: 640,
      snr_proxy: 18,
      depth_over_radii_model: 1.3,
    },
    {
      id: "TIC-22001122",
      mission: "TESS",
      score: 0.91,
      predictedClass: "PC",
      confidence: "High",
      period_days: 5.87,
      duration_hours: 3.6,
      depth_ppm: 520,
      snr_proxy: 16,
      depth_over_radii_model: 1.1,
    },
    {
      id: "EPIC-2455667",
      mission: "K2",
      score: 0.88,
      predictedClass: "PC",
      confidence: "Medium",
      period_days: 18.74,
      duration_hours: 5.1,
      depth_ppm: 710,
      snr_proxy: 14,
      depth_over_radii_model: 1.4,
    },
    {
      id: "TIC-33994422",
      mission: "TESS",
      score: 0.86,
      predictedClass: "PC",
      confidence: "Medium",
      period_days: 3.42,
      duration_hours: 2.4,
      depth_ppm: 430,
      snr_proxy: 11,
      depth_over_radii_model: 0.92,
    },
    {
      id: "KIC-77889900",
      mission: "Kepler",
      score: 0.84,
      predictedClass: "PC",
      confidence: "Medium",
      period_days: 22.12,
      duration_hours: 4.8,
      depth_ppm: 560,
      snr_proxy: 12,
      depth_over_radii_model: 1.05,
    },
  ],
  featureImportance: [
    { feature: "snr_proxy", gain: 0.24 },
    { feature: "depth_over_radii_model", gain: 0.18 },
    { feature: "depth_ppm", gain: 0.16 },
    { feature: "period_days", gain: 0.14 },
    { feature: "duration_hours", gain: 0.09 },
    { feature: "planet_radius_re", gain: 0.07 },
    { feature: "duty_cycle", gain: 0.05 },
    { feature: "stellar_teff_k", gain: 0.04 },
    { feature: "eqt_over_teff", gain: 0.03 },
  ],
  shapExample: {
    id: "KIC-33445566",
    mission: "Kepler",
    contributions: [
      { feature: "snr_proxy", contribution: 0.18 },
      { feature: "depth_over_radii_model", contribution: 0.12 },
      { feature: "depth_ppm", contribution: 0.1 },
      { feature: "period_days", contribution: 0.05 },
      { feature: "duty_cycle", contribution: -0.03 },
      { feature: "stellar_teff_k", contribution: -0.02 },
    ],
  },
};

function ChartCard({
  title,
  subtitle,
  children,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <h3>{title}</h3>
        {subtitle && <span>{subtitle}</span>}
      </div>
      {actions && <div className={styles.toolbar}>{actions}</div>}
      <div className={styles.cardBody}>{children}</div>
    </div>
  );
}

function DonutChart({ data }: { data: MissionMixDatum[] }) {
  const total = data.reduce((acc, item) => acc + item.count, 0);
  const radius = 72;
  const circumference = 2 * Math.PI * radius;
  let offset = 0;

  return (
    <svg viewBox="0 0 200 200" className={styles.chartCanvas}>
      <g transform="translate(100,100)">
        <circle r={radius} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth={22} />
        {data.map((item) => {
          const length = total === 0 ? 0 : (item.count / total) * circumference;
          const dashArray = `${length} ${circumference}`;
          const dashOffset = offset;
          offset -= length;
          return (
            <circle
              key={item.mission}
              r={radius}
              fill="none"
              stroke={missionPalette[item.mission]}
              strokeWidth={22}
              strokeDasharray={dashArray}
              strokeDashoffset={dashOffset}
              transform="rotate(-90)"
              strokeLinecap="butt"
            />
          );
        })}
        <text textAnchor="middle" dy="8" fontSize="22" fill="#ffffff">
          {total}
        </text>
        <text textAnchor="middle" dy="28" fontSize="12" fill="rgba(255,255,255,0.6)">
          signals
        </text>
      </g>
    </svg>
  );
}

function Legend({ missions }: { missions: MissionKey[] }) {
  return (
    <div className={styles.legendRow}>
      {missions.map((mission) => (
        <div key={mission} className={styles.legendItem}>
          <span className={styles.legendSwatch} style={{ background: missionPalette[mission] }} />
          <span>{mission}</span>
        </div>
      ))}
    </div>
  );
}

function MissingnessHeatmap({ data }: { data: MissingnessDatum[] }) {
  const features = Array.from(new Set(data.map((item) => item.feature)));
  const missions: MissionKey[] = ["Kepler", "TESS", "K2"];

  return (
    <div className={styles.tableWrapper}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Feature flag</th>
            {missions.map((mission) => (
              <th key={mission}>{mission}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {features.map((feature) => (
            <tr key={feature}>
              <td>{feature}</td>
              {missions.map((mission) => {
                const entry = data.find((item) => item.feature === feature && item.mission === mission);
                const rate = entry?.missingRate ?? 0;
                const background = `rgba(223, 106, 201, ${0.12 + rate * 0.9})`;
                return (
                  <td key={mission} style={{ background, fontVariantNumeric: "tabular-nums" }}>
                    {(rate * 100).toFixed(1)}%
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function HistogramStack({ config }: { config: HistogramConfig }) {
  const missions: MissionKey[] = ["Kepler", "TESS", "K2"];
  const maxCount = Math.max(
    ...config.bins.map((bin) =>
      missions.reduce((acc, mission) => Math.max(acc, bin.missions[mission]), 0)
    )
  );

  return (
    <div>
      <p style={{ margin: "0 0 4px" }}>
        {config.label} {config.logScale ? "(log)" : ""}
      </p>
      <Legend missions={missions} />
      <svg viewBox={`0 0 ${config.bins.length * 60} 160`} className={styles.chartCanvas}>
        {config.bins.map((bin, idx) => {
          const baseX = idx * 60 + 20;
          return (
            <g key={bin.bin} transform={`translate(${baseX},10)`}>
              <text x={10} y={140} fill="rgba(255,255,255,0.7)" fontSize={10} textAnchor="middle">
                {bin.bin}
              </text>
              {missions.map((mission, mIdx) => {
                const value = bin.missions[mission];
                const height = maxCount === 0 ? 0 : (value / maxCount) * 110;
                return (
                  <rect
                    key={mission}
                    x={mIdx * 12 - 12}
                    y={120 - height}
                    width={10}
                    height={height}
                    fill={missionPalette[mission]}
                    opacity={0.85}
                  />
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function ScatterPlot({ config }: { config: ScatterConfig }) {
  const missions: MissionKey[] = ["Kepler", "TESS", "K2"];
  const xs = config.points.map((p) => p.x);
  const ys = config.points.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  return (
    <div>
       <p style={{ margin: "0 0 4px" }}>{config.label}</p>
      <Legend missions={missions} />
      <svg viewBox="0 0 320 220" className={styles.sparkScatter}>
        <rect x={32} y={16} width={256} height={176} fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.1)" />
        {config.points.map((point) => {
          const xNorm = (point.x - xMin) / (xMax - xMin || 1);
          const yNorm = (point.y - yMin) / (yMax - yMin || 1);
          const cx = 32 + xNorm * 256;
          const cy = 16 + (1 - yNorm) * 176;
          return (
            <circle
              key={point.id}
              cx={cx}
              cy={cy}
              r={4}
              fill={missionPalette[point.mission]}
              opacity={0.7}
            />
          );
        })}
        <text x={160} y={210} textAnchor="middle" fill="rgba(255,255,255,0.7)" fontSize={11}>
          {config.xLabel}
        </text>
        <text x={12} y={104} transform="rotate(-90 12 104)" textAnchor="middle" fill="rgba(255,255,255,0.7)" fontSize={11}>
          {config.yLabel}
        </text>
      </svg>
    </div>
  );
}

function OutlierTable({ rows }: { rows: OutlierRow[] }) {
  return (
    <div className={styles.tableWrapper}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Candidate ID</th>
            <th>Mission</th>
            <th>Feature</th>
            <th>|z|</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.id}-${row.feature}`}>
              <td>{row.id}</td>
              <td>{row.mission}</td>
              <td>{row.feature}</td>
              <td>{row.zScore.toFixed(1)}</td>
              <td>
                {row.value.toLocaleString(undefined, {
                  maximumFractionDigits: 2,
                })} {row.units}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function QuantileRibbonChart({ config }: { config: QuantileRibbon }) {
  const bandPoints = config.train.map((d, idx) => ({
    x: idx,
    median: d.median,
    p10: d.p10,
    p90: d.p90,
  }));
  const densityMax = Math.max(...config.scoring.map((d) => d.density));

  return (
    <div>
      <p style={{ margin: "0 0 4px" }}>
        {config.feature} · {config.mission} {config.logScale ? "(log)" : ""}
      </p>
      <svg viewBox="0 0 320 200" className={styles.chartCanvas}>
        <rect x={28} y={20} width={264} height={140} fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.1)" />
        <path
          d={bandPoints
            .map((point, idx) => {
              const x = 28 + (idx / Math.max(1, bandPoints.length - 1)) * 264;
              const y = 160 - ((point.p90 - point.p10) / point.p90) * 140;
              return `${idx === 0 ? "M" : "L"}${x},${y}`;
            })
            .join(" ")}
          fill="none"
          stroke="rgba(127, 140, 255, 0.5)"
          strokeWidth={2}
        />
        <path
          d={config.scoring
            .map((point, idx) => {
              const x = 28 + (idx / Math.max(1, config.scoring.length - 1)) * 264;
              const y = 160 - (point.density / (densityMax || 1)) * 140;
              return `${idx === 0 ? "M" : "L"}${x},${y}`;
            })
            .join(" ")}
          fill="none"
          stroke="rgba(223, 106, 201, 0.8)"
          strokeWidth={2}
        />
        <text x={160} y={190} textAnchor="middle" fill="rgba(255,255,255,0.7)" fontSize={11}>
          Value grid
        </text>
        <text x={12} y={90} transform="rotate(-90 12 90)" textAnchor="middle" fill="rgba(255,255,255,0.7)" fontSize={11}>
          Density / band
        </text>
      </svg>
    </div>
  );
}

function DriftBars({ snapshots }: { snapshots: DriftSnapshot[] }) {
  const features = Array.from(new Set(snapshots.map((item) => item.feature)));
  return (
    <div>
      {features.map((feature) => (
        <div key={feature} style={{ marginBottom: 16 }}>
          <p style={{ margin: "0 0 6px" }}>{feature}</p>
          {(["Kepler", "TESS", "K2"] as MissionKey[]).map((mission) => {
            const entry = snapshots.find((item) => item.feature === feature && item.mission === mission);
            if (!entry) return null;
            const width = Math.min(100, Math.abs(entry.delta) * 100);
            return (
              <div key={mission} className={styles.sparkbarRow}>
                <span style={{ width: 60 }}>{mission}</span>
                <div className={styles.sparkbarTrack}>
                  <div
                    className={styles.sparkbarFill}
                    style={{
                      width: `${width}%`,
                      background: entry.delta >= 0 ? "#00dfd8" : "#df6ac9",
                    }}
                  />
                </div>
                <span>{entry.delta.toFixed(2)}</span>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function KLTable({ snapshots }: { snapshots: DriftSnapshot[] }) {
  const features = Array.from(new Set(snapshots.map((s) => s.feature)));
  const missions: MissionKey[] = ["Kepler", "TESS", "K2"];

  return (
    <div className={styles.tableWrapper}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Feature</th>
            {missions.map((mission) => (
              <th key={mission}>{mission}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {features.map((feature) => (
            <tr key={feature}>
              <td>{feature}</td>
              {missions.map((mission) => {
                const entry = snapshots.find((snap) => snap.feature === feature && snap.mission === mission);
                const value = entry?.kl ?? 0;
                const background = `rgba(127, 140, 255, ${Math.min(0.85, value * 3)})`;
                return (
                  <td key={mission} style={{ background }}>
                    {value.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CoverageHeatmap({ snapshots }: { snapshots: DriftSnapshot[] }) {
  const features = Array.from(new Set(snapshots.map((s) => s.feature)));
  const missions: MissionKey[] = ["Kepler", "TESS", "K2"];

  return (
    <div className={styles.tableWrapper}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Feature</th>
            {missions.map((mission) => (
              <th key={mission}>{mission}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {features.map((feature) => (
            <tr key={feature}>
              <td>{feature}</td>
              {missions.map((mission) => {
                const entry = snapshots.find((snap) => snap.feature === feature && snap.mission === mission);
                const value = entry?.coverage ?? 0;
                const background = `rgba(0, 223, 216, ${0.15 + value * 0.6})`;
                return (
                  <td key={mission} style={{ background }}>
                    {(value * 100).toFixed(0)}%
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ScoreHistogram({
  data,
  threshold,
  onThresholdChange,
}: {
  data: VisualizeDataset["scoreHistogram"];
  threshold: number;
  onThresholdChange: (value: number) => void;
}) {
  const missions: MissionKey[] = ["Kepler", "TESS", "K2"];
  const maxCount = Math.max(...data.map((bin) => bin.total));

  return (
    <div>
      <input
        className={styles.thresholdSlider}
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={threshold}
        onChange={(event) => onThresholdChange(Number(event.target.value))}
      />
      <div className={styles.thresholdLabel}>
        <span>Threshold: {threshold.toFixed(2)}</span>
        <span>Drag to update metrics</span>
      </div>
      <Legend missions={missions} />
      <svg viewBox="0 0 420 220" className={styles.chartCanvas}>
        <rect x={40} y={20} width={320} height={150} fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.1)" />
        {data.map((bin, idx) => {
          const width = 320 / data.length - 4;
          const baseX = 40 + idx * (width + 4);
          let offsetY = 0;
          return (
            <g key={`${bin.lower}-${bin.upper}`}>
              {missions.map((mission) => {
                const value = bin.missions[mission];
                const height = maxCount === 0 ? 0 : (value / maxCount) * 150;
                const rect = (
                  <rect
                    key={mission}
                    x={baseX}
                    y={170 - height - offsetY}
                    width={width}
                    height={height}
                    fill={missionPalette[mission]}
                    opacity={0.8}
                  />
                );
                offsetY += height;
                return rect;
              })}
            </g>
          );
        })}
        <line
          x1={40 + threshold * 320}
          x2={40 + threshold * 320}
          y1={20}
          y2={170}
          stroke="#ffdf6a"
          strokeDasharray="4 6"
          strokeWidth={2}
        />
        <text x={40 + threshold * 320} y={190} textAnchor="middle" fontSize={12} fill="#ffdf6a">
          τ
        </text>
      </svg>
    </div>
  );
}

function CurveChart({ data, title }: { data: CurvePoint[]; title: string }) {
  return (
    <div>
      <p style={{ margin: "0 0 4px" }}>{title}</p>
      <svg viewBox="0 0 320 200" className={styles.chartCanvas}>
        <rect x={32} y={20} width={256} height={150} fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.1)" />
        <polyline
          fill="none"
          stroke="#00dfd8"
          strokeWidth={2.5}
          points={data
            .map((point) => {
              const x = 32 + point.x * 256;
              const y = 170 - point.y * 150;
              return `${x},${y}`;
            })
            .join(" ")}
        />
      </svg>
    </div>
  );
}

function CalibrationChart({ data }: { data: CurvePoint[] }) {
  return (
    <div>
      <p style={{ margin: "0 0 4px" }}>Calibration curve</p>
      <svg viewBox="0 0 320 200" className={styles.chartCanvas}>
        <rect x={32} y={20} width={256} height={150} fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.1)" />
        <line x1={32} y1={170} x2={288} y2={20} stroke="rgba(255,255,255,0.2)" strokeDasharray="6 6" />
        <polyline
          fill="none"
          stroke="#df6ac9"
          strokeWidth={2.5}
          points={data
            .map((point) => {
              const x = 32 + point.x * 256;
              const y = 170 - point.y * 150;
              return `${x},${y}`;
            })
            .join(" ")}
        />
      </svg>
    </div>
  );
}

function ConfusionMatrix({ snapshot }: { snapshot: ThresholdSnapshot }) {
  const { tp, fp, fn, tn } = snapshot.matrix;
  return (
    <div className={styles.tableWrapper}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th></th>
            <th>Pred +</th>
            <th>Pred -</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th>True +</th>
            <td>{tp}</td>
            <td>{fn}</td>
          </tr>
          <tr>
            <th>True -</th>
            <td>{fp}</td>
            <td>{tn}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function MissionAcceptanceBars({ data }: { data: MissionAcceptance[] }) {
  return (
    <svg viewBox="0 0 360 200" className={styles.chartCanvas}>
      {data.map((row, idx) => {
        const y = 40 + idx * 50;
        const total = row.accepted + row.rejected;
        const acceptedWidth = (row.accepted / (total || 1)) * 260;
        const rejectedWidth = (row.rejected / (total || 1)) * 260;
        return (
          <g key={row.mission}>
            <text x={20} y={y + 15} fill="rgba(255,255,255,0.8)" fontSize={12}>
              {row.mission}
            </text>
            <rect x={80} y={y} width={acceptedWidth} height={18} fill="#00dfd8" rx={4} />
            <rect x={80 + acceptedWidth} y={y} width={rejectedWidth} height={18} fill="rgba(255,255,255,0.18)" rx={4} />
            <text x={80 + acceptedWidth + 6} y={y + 13} fill="rgba(255,255,255,0.7)" fontSize={11}>
              {row.accepted} accepted
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function TopCandidatesTable({ rows }: { rows: CandidateRow[] }) {
  return (
    <div className={styles.tableWrapper}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>ID</th>
            <th>Mission</th>
            <th>P(planet)</th>
            <th>Class</th>
            <th>Confidence</th>
            <th>Period (d)</th>
            <th>Duration (h)</th>
            <th>Depth (ppm)</th>
            <th>SNR proxy</th>
            <th>Depth / radii</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td>{row.id}</td>
              <td>{row.mission}</td>
              <td>{row.score.toFixed(2)}</td>
              <td>{row.predictedClass}</td>
              <td>{row.confidence}</td>
              <td>{row.period_days.toFixed(2)}</td>
              <td>{row.duration_hours.toFixed(1)}</td>
              <td>{row.depth_ppm}</td>
              <td>{row.snr_proxy.toFixed(1)}</td>
              <td>{row.depth_over_radii_model.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FeatureImportanceChart({ rows }: { rows: FeatureImportanceRow[] }) {
  const maxGain = Math.max(...rows.map((row) => row.gain));
  return (
    <div>
      {rows.map((row) => (
        <div key={row.feature} className={styles.sparkbarRow}>
          <span style={{ width: 150 }}>{row.feature}</span>
          <div className={styles.sparkbarTrack}>
            <div
              className={styles.sparkbarFill}
              style={{ width: `${(row.gain / (maxGain || 1)) * 100}%`, background: "#7f8cff" }}
            />
          </div>
          <span>{(row.gain * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

function ShapWaterfall({ shap }: { shap: VisualizeDataset["shapExample"] }) {
  const positive = shap.contributions.filter((c) => c.contribution >= 0);
  const negative = shap.contributions.filter((c) => c.contribution < 0);

  return (
    <div>
      <p style={{ margin: "0 0 8px" }}>
        SHAP contributions for <strong>{shap.id}</strong> · {shap.mission}
      </p>
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
        <div style={{ flex: 1, minWidth: 240 }}>
          <p style={{ margin: "0 0 4px" }}>Push up</p>
          {positive.map((row) => (
            <div key={row.feature} className={styles.sparkbarRow}>
              <span style={{ width: 140 }}>{row.feature}</span>
              <div className={styles.sparkbarTrack}>
                <div
                  className={styles.sparkbarFill}
                  style={{ width: `${row.contribution * 100}%`, background: "#00dfd8" }}
                />
              </div>
              <span>{(row.contribution * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
        <div style={{ flex: 1, minWidth: 240 }}>
          <p style={{ margin: "0 0 4px" }}>Push down</p>
          {negative.map((row) => (
            <div key={row.feature} className={styles.sparkbarRow}>
              <span style={{ width: 140 }}>{row.feature}</span>
              <div className={styles.sparkbarTrack}>
                <div
                  className={styles.sparkbarFill}
                  style={{ width: `${Math.abs(row.contribution) * 100}%`, background: "#df6ac9" }}
                />
              </div>
              <span>{(row.contribution * -100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function VisualizePage() {
  const [threshold, setThreshold] = useState(0.5);

  const thresholdSnapshot = useMemo(() => {
    let closest = sampleData.thresholdSnapshots[0];
    let minDiff = Math.abs(threshold - closest.threshold);
    sampleData.thresholdSnapshots.forEach((snap) => {
      const diff = Math.abs(threshold - snap.threshold);
      if (diff < minDiff) {
        minDiff = diff;
        closest = snap;
      }
    });
    return closest;
  }, [threshold]);

  const acceptance = useMemo(() => {
    const missions: MissionKey[] = ["Kepler", "TESS", "K2"];
    return missions.map((mission) => {
      const accepted = sampleData.candidates.filter((row) => row.mission === mission && row.score >= threshold).length;
      const total = sampleData.missionMix.find((mix) => mix.mission === mission)?.count ?? 0;
      return {
        mission,
        accepted,
        rejected: Math.max(total - accepted, 0),
      };
    });
  }, [threshold]);

  return (
    <div className={styles.pageWrapper}>
      <div className={styles.pageContent}>
        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <h2>Visualize</h2>
            <p>Three stacked dashboards to audit your upload, benchmark against training, and tune predictions.</p>
          </div>
          <div className={styles.callout}>
            <strong>Heads up:</strong> these charts render from a demo dataset so the layout is previewable without running the
            full pipeline. Swap <code>sampleData</code> for your API response to go live.
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <h2>A · Data overview &amp; QC</h2>
            <p>Mission composition, missingness, distributions, relationships, and notable outliers.</p>
          </div>
          <div className={`${styles.grid} ${styles.gridTwo}`}>
            <ChartCard title="Mission mix" subtitle="Counts by survey">
              <DonutChart data={sampleData.missionMix} />
              <Legend missions={["Kepler", "TESS", "K2"]} />
            </ChartCard>
            <ChartCard title="Missingness heatmap" subtitle="Boolean flags from validator">
              <MissingnessHeatmap data={sampleData.missingness} />
            </ChartCard>
          </div>

          <div className={`${styles.grid} ${styles.gridTwo}`}>
            <ChartCard title="Key feature histograms" subtitle="Overlaid per mission">
              <div style={{ display: "grid", gap: 16 }}>
                {sampleData.histograms.slice(0, 6).map((config) => (
                  <HistogramStack key={config.feature} config={config} />
                ))}
              </div>
            </ChartCard>
            <ChartCard title="More feature histograms" subtitle="Continue core list">
              <div style={{ display: "grid", gap: 16 }}>
                {sampleData.histograms.slice(6).map((config) => (
                  <HistogramStack key={config.feature} config={config} />
                ))}
              </div>
            </ChartCard>
          </div>

          <div className={`${styles.grid} ${styles.gridTwo}`}>
            <ChartCard title="Feature relationships" subtitle="Scatter by mission">
              <div style={{ display: "grid", gap: 16 }}>
                {sampleData.scatter.slice(0, 2).map((config) => (
                  <ScatterPlot key={config.id} config={config} />
                ))}
              </div>
            </ChartCard>
            <ChartCard title="Feature relationships" subtitle="Continued pairs">
              <div style={{ display: "grid", gap: 16 }}>
                {sampleData.scatter.slice(2).map((config) => (
                  <ScatterPlot key={config.id} config={config} />
                ))}
              </div>
            </ChartCard>
          </div>

          <ChartCard title="Outlier panel" subtitle="Z-score highlights">
            <OutlierTable rows={sampleData.outliers} />
          </ChartCard>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <h2>B · Compare to training</h2>
            <p>Distribution drift versus the bundled KOI/TOI/K2 training corpora.</p>
          </div>
          <div className={`${styles.grid} ${styles.gridTwo}`}>
            {sampleData.quantiles.map((config) => (
              <ChartCard key={`${config.feature}-${config.mission}`} title="Quantile ribbon" subtitle={`${config.feature} · ${config.mission}`}>
                <QuantileRibbonChart config={config} />
              </ChartCard>
            ))}
          </div>

          <div className={`${styles.grid} ${styles.gridTwo}`}>
            <ChartCard title="Normalized difference" subtitle="Median shift / train IQR">
              <DriftBars snapshots={sampleData.drift} />
            </ChartCard>
            <ChartCard title="Mission-wise KL" subtitle="Distribution divergence">
              <KLTable snapshots={sampleData.drift} />
            </ChartCard>
          </div>

          <ChartCard title="Coverage vs train envelope" subtitle="Percent inside train P5–P95">
            <CoverageHeatmap snapshots={sampleData.drift} />
          </ChartCard>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <h2>C · Predictions &amp; thresholds</h2>
            <p>Score distributions, performance curves, confusion matrix, and candidate tables.</p>
          </div>

          <div className={`${styles.grid} ${styles.gridTwo}`}>
            <ChartCard title="Score distribution" subtitle="Histogram + threshold">
              <ScoreHistogram data={sampleData.scoreHistogram} threshold={threshold} onThresholdChange={setThreshold} />
            </ChartCard>
            <ChartCard title="Mission acceptance" subtitle="Counts ≥ τ">
              <MissionAcceptanceBars data={acceptance} />
            </ChartCard>
          </div>

          <div className={`${styles.grid} ${styles.gridThree}`}>
            <ChartCard title="Precision–Recall" subtitle="From curves.pr">
              <CurveChart data={sampleData.prCurve} title="PR" />
            </ChartCard>
            <ChartCard title="ROC" subtitle="From curves.roc">
              <CurveChart data={sampleData.rocCurve} title="ROC" />
            </ChartCard>
            <ChartCard title="Calibration" subtitle="From curves.calibration">
              <CalibrationChart data={sampleData.calibration} />
            </ChartCard>
          </div>

          <ChartCard title="Confusion matrix & metrics" subtitle="Auto-selects nearest snapshot">
            <ConfusionMatrix snapshot={thresholdSnapshot} />
            <div className={styles.metricsGrid}>
              <div className={styles.metricBox}>
                <span className={styles.metricTitle}>PR AUC</span>
                <span className={styles.metricValue}>{thresholdSnapshot.prAuc.toFixed(2)}</span>
              </div>
              <div className={styles.metricBox}>
                <span className={styles.metricTitle}>ROC AUC</span>
                <span className={styles.metricValue}>{thresholdSnapshot.rocAuc.toFixed(2)}</span>
              </div>
              <div className={styles.metricBox}>
                <span className={styles.metricTitle}>F1</span>
                <span className={styles.metricValue}>{thresholdSnapshot.f1.toFixed(2)}</span>
              </div>
              <div className={styles.metricBox}>
                <span className={styles.metricTitle}>Recall</span>
                <span className={styles.metricValue}>{thresholdSnapshot.recall.toFixed(2)}</span>
              </div>
              <div className={styles.metricBox}>
                <span className={styles.metricTitle}>Brier</span>
                <span className={styles.metricValue}>{thresholdSnapshot.brier.toFixed(2)}</span>
              </div>
            </div>
          </ChartCard>

          <ChartCard title="Top candidates" subtitle="Filter/export controls go here">
            <TopCandidatesTable rows={sampleData.candidates} />
          </ChartCard>

          <div className={`${styles.grid} ${styles.gridTwo}`}>
            <ChartCard title="Feature importance" subtitle="Global model gain">
              <FeatureImportanceChart rows={sampleData.featureImportance} />
            </ChartCard>
            <ChartCard title="SHAP spotlight" subtitle="Selected candidate">
              <ShapWaterfall shap={sampleData.shapExample} />
            </ChartCard>
          </div>
        </section>
      </div>
    </div>
  );
}
