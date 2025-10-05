import { useState } from "react";
import styles from "../styles/pages/data.module.css";

type ValidatorIssue = {
  severity: string;
  code: string;
  msg: string;
};

type ValidatorReport = {
  status: string;
  row_count: number;
  detected_columns: string[];
  missing_pct: Record<string, number>;
  issues: ValidatorIssue[];
  tips: string[];
};

type ValidateResponse = {
  ok?: boolean;
  report?: ValidatorReport;
  error?: string;
};

export default function DataValidatePage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ValidateResponse | null>(null);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/validate", {
        method: "POST",
        body: formData,
      });

      const payload: ValidateResponse = await response.json();
      if (!response.ok || payload.error) {
        throw new Error(payload.error || "Validation failed");
      }
      setResult(payload);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.container}>
      <p className={styles.title}>Validate Observation Data</p>
      <p style={{ marginBottom: "1rem", textAlign: 'center' }}>Upload your candidate CSV to run the schema and sanity validator before prediction.</p>
      <a
        className={styles.downloadButton}
        href="/Demo/demo_strict_fixed.csv"
        download
      >
        Download Demo CSV
      </a>
      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 12, alignItems: 'center' }}>
        <input
          type="file"
          accept=".csv,.zip"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          disabled={loading}
          className={styles.inputFile}
        />
        <button type="submit" disabled={!file || loading}>
          {loading ? "Validating..." : "Validate"}
        </button>
      </form>

      {error && (
        <p style={{ color: "#b00020", marginTop: "1rem" }}>Validation error: {error}</p>
      )}

      {result?.report && (
        <div style={{ marginTop: "2rem", fontFamily: 'Cantarell', color: 'white', maxWidth: '100%' }}>
          <p>Validator Report</p>
          <p>
            Status: <strong>{result.report.status}</strong> · Rows analysed: {result.report.row_count}
          </p>
          {typeof result.ok === "boolean" && (
            <p>Gate status: {result.ok ? "Ready for scoring" : "Blocked"}</p>
          )}

          {result.report.issues.length > 0 ? (
            <div>
              <p>Issues</p>
              <ul>
                {result.report.issues.map((issue, idx) => (
                  <li key={`${issue.code}-${idx}`}>
                    <strong>{issue.severity.toUpperCase()}</strong> [{issue.code}]: {issue.msg}
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p>No issues detected.</p>
          )}

          <details style={{ marginTop: "1rem" }}>
            <summary>Raw JSON</summary>
            <pre style={{ background: "#111", color: "#0f0", padding: "1rem", overflowX: 'auto', borderRadius: 10 }}>
              {JSON.stringify(result.report, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  );
}
