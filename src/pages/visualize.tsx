import { useEffect, useMemo, useState, type FormEvent, type ReactNode } from "react";
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
  train: {
    p10: number | null;
    median: number | null;
    p90: number | null;
    p5: number | null;
    p95: number | null;
    iqr: number | null;
  };
  scoring: {
    edges: number[];
    counts: number[];
  };
};

type DuplicateSummary = {
  duplicateIds: number;
  duplicateRows: number;
  perMission: Record<MissionKey, number>;
  starMultiplicity: Array<{ value: number; count: number }>;
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
  prAuc: number | null;
  rocAuc: number | null;
  f1: number | null;
  recall: number | null;
  brier: number | null;
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
  score: number | null;
  predictedClass: string | null;
  confidence: string | null;
  period_days: number | null;
  duration_hours: number | null;
  depth_ppm: number | null;
  snr_proxy: number | null;
  depth_over_radii_model: number | null;
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
  duplicates: DuplicateSummary;
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
  shapExample: { id: string; mission: MissionKey; contributions: ShapRow[] } | null;
  thresholdUsed?: number;
  perMissionThresholds?: Record<string, number>;
};

const missionPalette: Record<MissionKey, string> = {
  Kepler: "#00dfd8",
  TESS: "#df6ac9",
  K2: "#7f8cff",
};

const num = (v: number | null | undefined, fallback = 0) =>
  typeof v === "number" && Number.isFinite(v) ? v : fallback;

const nfmt = (v: number | null | undefined, digits = 2) =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";


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
  if (!config.bins.length) {
    return <p>No histogram data.</p>;
  }
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
  if (!config.points.length) {
    return <p>No scatter data.</p>;
  }
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
  const edges = config.scoring.edges ?? [];
  const counts = config.scoring.counts ?? [];
  if (edges.length < 2) {
    return <p>No distribution data available.</p>;
  }

  const transform = (value: number) => {
    const safeValue = config.logScale ? Math.max(value, 1e-9) : value;
    return config.logScale ? Math.log10(safeValue) : safeValue;
  };

  const transformedEdges = edges.map(transform);
  const minEdge = Math.min(...transformedEdges);
  const maxEdge = Math.max(...transformedEdges);
  const maxCount = Math.max(...counts, 0);
  const plotWidth = 264;
  const plotHeight = 140;

  const toX = (value: number | null | undefined) => {
    if (value == null || !Number.isFinite(value)) return null;
    const transformed = transform(value);
    if (!Number.isFinite(transformed)) return null;
    const norm = (transformed - minEdge) / (maxEdge - minEdge || 1);
    return 28 + norm * plotWidth;
  };

  const quantileLines = [
    { label: "P10", value: config.train.p10, color: "#df6ac9" },
    { label: "Median", value: config.train.median, color: "#00dfd8" },
    { label: "P90", value: config.train.p90, color: "#df6ac9" },
  ];

  return (
    <div>
      <p style={{ margin: "0 0 4px" }}>
        {config.feature} · {config.mission} {config.logScale ? "(log)" : ""}
      </p>
      <svg viewBox="0 0 320 200" className={styles.chartCanvas}>
        <rect x={28} y={20} width={plotWidth} height={plotHeight} fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.1)" />
        {counts.map((count, idx) => {
          if (idx >= transformedEdges.length - 1) return null;
          const lo = transformedEdges[idx];
          const hi = transformedEdges[idx + 1];
          const width = ((hi - lo) / (maxEdge - minEdge || 1)) * plotWidth;
          const x = 28 + ((lo - minEdge) / (maxEdge - minEdge || 1)) * plotWidth;
          const height = (count / (maxCount || 1)) * plotHeight;
          return (
            <rect
              key={`bin-${idx}`}
              x={x}
              y={20 + (plotHeight - height)}
              width={Math.max(width - 2, 1)}
              height={height}
              fill="rgba(223, 106, 201, 0.6)"
            />
          );
        })}
        {quantileLines.map((line) => {
          const x = toX(line.value);
          if (x == null) return null;
          return (
            <g key={line.label}>
              <line x1={x} x2={x} y1={20} y2={160} stroke={line.color} strokeDasharray="4 6" strokeWidth={1.5} />
              <text x={x} y={170} textAnchor="middle" fontSize={10} fill={line.color}>
                {line.label}
              </text>
            </g>
          );
        })}
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
            const d = num(entry.delta, 0);
            const width = Math.min(100, Math.abs(d) * 100);
            return (
              <div key={mission} className={styles.sparkbarRow}>
                <span style={{ width: 60 }}>{mission}</span>
                <div className={styles.sparkbarTrack}>
                  <div
                    className={styles.sparkbarFill}
                    style={{
                      width: `${width}%`,
                      background: d >= 0 ? "#00dfd8" : "#df6ac9",
                    }}
                  />
                </div>
                <span>{nfmt(entry.delta, 2)}</span>
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
                const value = num(entry?.kl, 0);
                const background = `rgba(127, 140, 255, ${Math.min(0.85, value * 3)})`;
                return (
                  <td key={mission} style={{ background }}>
                    {nfmt(value, 2)}
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
                    {nfmt(value * 100, 0)}%
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
  if (!data.length) {
    return (
      <div>
        <input
          className={styles.thresholdSlider}
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={threshold}
          onChange={() => {}}
          disabled
        />
        <p style={{ marginTop: 12, opacity: 0.7 }}>Upload a CSV to view score distributions.</p>
      </div>
    );
  }
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

function ConfusionMatrix({ snapshot }: { snapshot: ThresholdSnapshot | null }) {
  if (!snapshot) {
    return <p>Upload ground-truth labels to enable confusion matrix diagnostics.</p>;
  }
  const matrix = snapshot.matrix || { tp: 0, fp: 0, fn: 0, tn: 0 };
  const { tp, fp, fn, tn } = matrix;
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
  if (!data.length) {
    return <p>No acceptance counts yet.</p>;
  }
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
              <td>{nfmt(row.score, 2)}</td>
              <td>{row.predictedClass ?? "—"}</td>
              <td>{row.confidence ?? "—"}</td>
              <td>{nfmt(row.period_days, 2)}</td>
              <td>{nfmt(row.duration_hours, 1)}</td>
              <td>{row.depth_ppm != null && Number.isFinite(row.depth_ppm) ? row.depth_ppm.toLocaleString() : "—"}</td>
              <td>{nfmt(row.snr_proxy, 1)}</td>
              <td>
                {row.depth_over_radii_model != null
                  ? row.depth_over_radii_model.toFixed(2)
                  : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FeatureImportanceChart({ rows }: { rows: FeatureImportanceRow[] }) {
  const maxGain = rows.length ? Math.max(...rows.map((row) => row.gain)) : 0;
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
          <span>{nfmt(row.gain * 100, 1)}%</span>
        </div>
      ))}
    </div>
  );
}

function ShapWaterfall({ shap }: { shap: VisualizeDataset["shapExample"] }) {
  if (!shap || !Array.isArray(shap.contributions) || shap.contributions.length === 0) {
    return <p>No SHAP breakdown available for this upload.</p>;
  }
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

function DuplicateSummaryCard({ summary }: { summary: DuplicateSummary | null }) {
  if (!summary) {
    return <p>No duplicate metrics yet. Upload a CSV to populate this panel.</p>;
  }

  const missions: MissionKey[] = ["Kepler", "TESS", "K2"];

  return (
    <div className={styles.duplicateCard}>
      <p>
        Duplicate IDs: <strong>{summary.duplicateIds}</strong> · Rows affected: <strong>{summary.duplicateRows}</strong>
      </p>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <div style={{ flex: 1, minWidth: 180 }}>
          <p style={{ margin: "0 0 4px" }}>By mission</p>
          {missions.map((mission) => (
            <div key={mission} className={styles.sparkbarRow}>
              <span style={{ width: 80 }}>{mission}</span>
              <span>{summary.perMission[mission] ?? 0}</span>
            </div>
          ))}
        </div>
        <div style={{ flex: 1, minWidth: 180 }}>
          <p style={{ margin: "0 0 4px" }}>Star multiplicity</p>
          {summary.starMultiplicity.length === 0 && <span>—</span>}
          {summary.starMultiplicity.map((item) => (
            <div key={item.value} className={styles.sparkbarRow}>
              <span style={{ width: 80 }}>×{item.value}</span>
              <span>{item.count}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function VisualizePage() {
  const [file, setFile] = useState<File | null>(null);
  const [data, setData] = useState<VisualizeDataset | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.5);

  useEffect(() => {
    if (data?.thresholdUsed != null && Number.isFinite(data.thresholdUsed)) {
      setThreshold(data.thresholdUsed);
    }
  }, [data?.thresholdUsed]);

  const thresholdSnapshot = useMemo(() => {
    if (!data?.thresholdSnapshots?.length) return null;
    let closest = data.thresholdSnapshots[0];
    let minDiff = Math.abs(threshold - closest.threshold);
    data.thresholdSnapshots.forEach((snap) => {
      const diff = Math.abs(threshold - snap.threshold);
      if (diff < minDiff) {
        minDiff = diff;
        closest = snap;
      }
    });
    return closest;
  }, [data, threshold]);

  const acceptance = useMemo(() => {
    if (!data) return [];
    const missions: MissionKey[] = ["Kepler", "TESS", "K2"];
    return missions.map((mission) => {
      let accepted = 0;
      data.scoreHistogram.forEach((bin) => {
        const count = bin.missions[mission] ?? 0;
        const lower = bin.lower;
        const upper = bin.upper;
        if (threshold >= upper) return;
        if (threshold <= lower) {
          accepted += count;
        } else {
          const portion = (upper - threshold) / (upper - lower || 1);
          accepted += count * Math.max(Math.min(portion, 1), 0);
        }
      });
      const total = data.missionMix.find((mix) => mix.mission === mission)?.count ?? 0;
      const acceptedRounded = Math.round(accepted);
      return {
        mission,
        accepted: acceptedRounded,
        rejected: Math.max(total - acceptedRounded, 0),
      };
    });
  }, [data, threshold]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch("/api/visualize", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok || payload.error) {
        throw new Error(payload.error || "Visualization failed");
      }
      setData(payload as VisualizeDataset);
      if (payload.thresholdUsed != null && Number.isFinite(payload.thresholdUsed)) {
        setThreshold(payload.thresholdUsed);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const formatMetric = (value: number | null | undefined) =>
    value != null && Number.isFinite(value) ? value.toFixed(2) : "—";

  return (
    <div className={styles.pageWrapper}>
      <div className={styles.pageContent}>
        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <h2>Visualize</h2>
            <p>Upload a CSV to audit its health, compare against training corpora, and inspect model outputs.</p>
          </div>
          <form className={styles.uploadForm} onSubmit={handleSubmit}>
            <input
              type="file"
              accept=".csv,.zip"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              disabled={loading}
            />
            <button type="submit" disabled={!file || loading}>
              {loading ? "Processing..." : "Generate dashboard"}
            </button>
          </form>
          {error && <p className={styles.error}>{error}</p>}
        </section>

        {data ? (
          <>
            <section className={styles.section}>
              <div className={styles.sectionHeader}>
                <h2>A · Data overview &amp; QC</h2>
                <p>Mission composition, missingness, duplicates, distributions, and notable outliers.</p>
              </div>
              <div className={`${styles.grid} ${styles.gridThree}`}>
                <ChartCard title="Mission mix" subtitle="Counts by survey">
                  <DonutChart data={data.missionMix} />
                  <Legend missions={["Kepler", "TESS", "K2"]} />
                </ChartCard>
                <ChartCard title="Missingness heatmap" subtitle="Boolean flags from validator">
                  <MissingnessHeatmap data={data.missingness} />
                </ChartCard>
                <ChartCard title="Duplicates &amp; multiplicity" subtitle="ID and star reuse">
                  <DuplicateSummaryCard summary={data.duplicates ?? null} />
                </ChartCard>
              </div>

              <div className={`${styles.grid} ${styles.gridTwo}`}>
                <ChartCard title="Key feature histograms" subtitle="Overlaid per mission">
                  <div style={{ display: "grid", gap: 16 }}>
                    {data.histograms.slice(0, Math.ceil(data.histograms.length / 2)).map((config) => (
                      <HistogramStack key={config.feature} config={config} />
                    ))}
                  </div>
                </ChartCard>
                <ChartCard title="More feature histograms" subtitle="Continue core list">
                  <div style={{ display: "grid", gap: 16 }}>
                    {data.histograms.slice(Math.ceil(data.histograms.length / 2)).map((config) => (
                      <HistogramStack key={config.feature} config={config} />
                    ))}
                  </div>
                </ChartCard>
              </div>

              <div className={`${styles.grid} ${styles.gridTwo}`}>
                <ChartCard title="Feature relationships" subtitle="Scatter by mission">
                  <div style={{ display: "grid", gap: 16 }}>
                    {data.scatter.slice(0, 2).map((config) => (
                      <ScatterPlot key={config.id} config={config} />
                    ))}
                  </div>
                </ChartCard>
                <ChartCard title="Feature relationships" subtitle="Continued pairs">
                  <div style={{ display: "grid", gap: 16 }}>
                    {data.scatter.slice(2).map((config) => (
                      <ScatterPlot key={config.id} config={config} />
                    ))}
                  </div>
                </ChartCard>
              </div>

              <ChartCard title="Outlier panel" subtitle="Z-score highlights">
                <OutlierTable rows={data.outliers} />
              </ChartCard>
            </section>

            <section className={styles.section}>
              <div className={styles.sectionHeader}>
                <h2>B · Compare to training</h2>
                <p>Distribution drift versus the bundled KOI/TOI/K2 training corpora.</p>
              </div>
              <div className={`${styles.grid} ${styles.gridTwo}`}>
                {data.quantiles.map((config) => (
                  <ChartCard key={`${config.feature}-${config.mission}`} title="Quantile overlay" subtitle={`${config.feature} · ${config.mission}`}>
                    <QuantileRibbonChart config={config} />
                  </ChartCard>
                ))}
              </div>

              <div className={`${styles.grid} ${styles.gridTwo}`}>
                <ChartCard title="Normalized difference" subtitle="Median shift / train IQR">
                  <DriftBars snapshots={data.drift} />
                </ChartCard>
                <ChartCard title="Mission-wise KL" subtitle="Distribution divergence">
                  <KLTable snapshots={data.drift} />
                </ChartCard>
              </div>

              <ChartCard title="Coverage vs train envelope" subtitle="Percent inside train P5–P95">
                <CoverageHeatmap snapshots={data.drift} />
              </ChartCard>
            </section>

            <section className={styles.section}>
              <div className={styles.sectionHeader}>
                <h2>C · Predictions &amp; thresholds</h2>
                <p>Score distributions, performance curves, confusion matrix, and candidate tables.</p>
              </div>

              <div className={`${styles.grid} ${styles.gridTwo}`}>
                <ChartCard title="Score distribution" subtitle="Histogram + threshold">
                  <ScoreHistogram data={data.scoreHistogram} threshold={threshold} onThresholdChange={setThreshold} />
                </ChartCard>
                <ChartCard title="Mission acceptance" subtitle="Counts ≥ τ">
                  <MissionAcceptanceBars data={acceptance} />
                </ChartCard>
              </div>

              <div className={`${styles.grid} ${styles.gridThree}`}>
                <ChartCard title="Precision–Recall" subtitle="From curves.pr">
                  <CurveChart data={data.prCurve} title="PR" />
                </ChartCard>
                <ChartCard title="ROC" subtitle="From curves.roc">
                  <CurveChart data={data.rocCurve} title="ROC" />
                </ChartCard>
                <ChartCard title="Calibration" subtitle="From curves.calibration">
                  <CalibrationChart data={data.calibration} />
                </ChartCard>
              </div>

              <ChartCard title="Confusion matrix & metrics" subtitle="Auto-selects nearest snapshot">
                <ConfusionMatrix snapshot={thresholdSnapshot} />
                <div className={styles.metricsGrid}>
                  <div className={styles.metricBox}>
                    <span className={styles.metricTitle}>PR AUC</span>
                    <span className={styles.metricValue}>{formatMetric(thresholdSnapshot?.prAuc)}</span>
                  </div>
                  <div className={styles.metricBox}>
                    <span className={styles.metricTitle}>ROC AUC</span>
                    <span className={styles.metricValue}>{formatMetric(thresholdSnapshot?.rocAuc)}</span>
                  </div>
                  <div className={styles.metricBox}>
                    <span className={styles.metricTitle}>F1</span>
                    <span className={styles.metricValue}>{formatMetric(thresholdSnapshot?.f1)}</span>
                  </div>
                  <div className={styles.metricBox}>
                    <span className={styles.metricTitle}>Recall</span>
                    <span className={styles.metricValue}>{formatMetric(thresholdSnapshot?.recall)}</span>
                  </div>
                  <div className={styles.metricBox}>
                    <span className={styles.metricTitle}>Brier</span>
                    <span className={styles.metricValue}>{formatMetric(thresholdSnapshot?.brier)}</span>
                  </div>
                </div>
              </ChartCard>

              <ChartCard title="Top candidates" subtitle="Highest scoring signals">
                <TopCandidatesTable rows={data.candidates} />
              </ChartCard>

              <div className={`${styles.grid} ${styles.gridTwo}`}>
                <ChartCard title="Feature importance" subtitle="Global model gain">
                  <FeatureImportanceChart rows={data.featureImportance} />
                </ChartCard>
                <ChartCard title="SHAP spotlight" subtitle="Selected candidate">
                  <ShapWaterfall shap={data.shapExample ?? null} />
                </ChartCard>
              </div>
            </section>
          </>
        ) : (
          <section className={styles.section}>
            <p style={{ opacity: 0.7 }}>
              No charts yet. Upload a candidate CSV that passes validation to explore its quality, compare to the bundled training
              corpora, and inspect prediction metrics.
            </p>
          </section>
        )}
      </div>
    </div>
  );
}
