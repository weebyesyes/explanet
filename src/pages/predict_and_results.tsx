import { useEffect, useMemo, useState } from "react";
import styles from "../styles/pages/predict_and_results.module.css";

type PredictionRow = Record<string, unknown>;

type PredictionResponse = {
  threshold_used?: number;
  per_mission_thresholds?: Record<string, number>;
  metrics?: Record<string, unknown> | null;
  curves?: Record<string, unknown> | null;
  preds?: PredictionRow[];
  csv?: string;
  error?: string;
};

export default function PredictPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null;
    if (loading) {
      setProgress(10);
      timer = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 95) return prev;
          const next = prev + Math.random() * 12;
          return next > 95 ? 95 : next;
        });
      }, 400);
    } else if (!loading && progress !== 0 && progress !== 100) {
      setProgress(100);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [loading, progress]);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

    const text = await response.text();
      let payload: PredictionResponse = {};
      if (text) {
        try {
          payload = JSON.parse(text);
        } catch (jsonErr) {
          throw new Error(`Invalid response: ${(jsonErr as Error).message}`);
        }
      }

      if (!response.ok || payload.error) {
        throw new Error(payload.error || "Prediction failed");
      }

      setResult(payload);
      setProgress(100);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const columns = useMemo(() => {
    if (!result?.preds || result.preds.length === 0) return [];
    return Object.keys(result.preds[0]);
  }, [result?.preds]);

  function handleDownload() {
    if (!result?.csv) return;
    const blob = new Blob([result.csv], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.setAttribute("download", "predictions.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  return (
    <div className={styles.container}>
      <p className={styles.title}>Run Predictions</p>
      <p style={{ marginBottom: "1rem" }}>
        Upload a validated CSV to trigger the inference pipeline. The Python bundle will load the
        trained models, compute predictions, and return the tri-class outcomes.
      </p>

      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 12, alignItems: "center" }}>
        <input
          type="file"
          accept=".csv,.zip"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          disabled={loading}
          className={styles.inputFile}
        />
        
        <button className={styles.submitButton} type="submit" disabled={!file || loading}>
          {loading ? "Running inference..." : "Predict"}
        </button>
      </form>

      {loading && (
        <div style={{ marginTop: "1.5rem" }}>
          <p>Model running...</p>
          <div style={{ background: "#1f1f1f", height: 12, borderRadius: 6, overflow: "hidden" }}>
            <div
              style={{
                width: `${progress}%`,
                height: "100%",
                background: "linear-gradient(90deg, #007cf0, #00dfd8)",
                transition: "width 0.3s ease",
              }}
            />
          </div>
        </div>
      )}

      {error && (
        <p style={{ color: "#b00020", marginTop: "1rem" }}>Prediction error: {error}</p>
      )}

      {result && !error && (
        <div style={{ marginTop: "2rem" }}>
          <h2>Results</h2>
          {typeof result.threshold_used === "number" && (
            <p>Threshold used: {result.threshold_used.toFixed(4)}</p>
          )}
          {result.per_mission_thresholds && Object.keys(result.per_mission_thresholds).length > 0 && (
            <div>
              <h3>Per-mission thresholds</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                {Object.entries(result.per_mission_thresholds).map(([mission, thr]) => {
                  const value = typeof thr === "number" ? thr : Number(thr);
                  return (
                    <p key={mission} style={{ margin: 0, textAlign: "left" }}>
                      {mission}: {Number.isFinite(value) ? value.toFixed(4) : String(thr)}
                    </p>
                  );
                })}
              </div>
            </div>
          )}

          {columns.length > 0 && result.preds && (
            <div style={{ overflowX: "auto", marginTop: "1.5rem" }}>
              <div
                style={{
                  borderRadius: 8,
                  border: "1px solid rgba(122, 160, 255, 0.5)",
                  overflow: "hidden",
                  minWidth: "max-content",
                }}
              >
                <table
                  style={{
                    borderCollapse: "collapse",
                    width: "100%",
                    background: "rgba(255, 255, 255, 0.92)",
                    color: "#111111",
                  }}
                >
                  <thead>
                    <tr>
                      {columns.map((col) => (
                        <th
                          key={col}
                        style={{
                          textAlign: "left",
                          padding: "0.75rem",
                          background: "rgba(122, 160, 255, 0.2)",
                          border: "1px solid rgba(122, 160, 255, 0.5)",
                          fontWeight: 600,
                        }}
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.preds.map((row, idx) => (
                    <tr key={idx}>
                      {columns.map((col) => (
                        <td
                          key={col}
                          style={{
                            padding: "0.75rem",
                            border: "1px solid rgba(122, 160, 255, 0.35)",
                            background: "rgba(255, 255, 255, 0.96)",
                          }}
                        >
                          {(() => {
                            const value = row[col];
                            if (typeof value === "boolean") return value ? "True" : "False";
                            if (typeof value === "number") return value.toString();
                            if (value === null || value === undefined) return "";
                            return String(value);
                          })()}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
                </table>
              </div>
            </div>
          )}

          {result.csv && (
            <button style={{ marginTop: "1.5rem" }} onClick={handleDownload}>
              Download CSV
            </button>
          )}

          {(result.metrics || result.curves) && (
            <details style={{ marginTop: "1.5rem" }}>
              <summary>Diagnostics</summary>
              <pre style={{ background: "#111", color: "#0f0", padding: "1rem", overflowX: "auto" }}>
                {JSON.stringify(
                  {
                    metrics: result.metrics,
                    curves: result.curves,
                  },
                  null,
                  2
                )}
              </pre>
            </details>
          )}
        </div>
      )}

    </div>
  );
}
