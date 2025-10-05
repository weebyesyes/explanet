import fs from "fs";
import path from "path";
import formidable, { File as FormidableFile } from "formidable";
import { NextApiRequest, NextApiResponse } from "next";
import { spawn } from "child_process";

export const config = {
  api: {
    bodyParser: false,
  },
};

type ParseResult = {
  fields: formidable.Fields;
  files: formidable.Files;
};

type PredictionResponse = {
  threshold_used?: number;
  per_mission_thresholds?: Record<string, number>;
  metrics?: unknown;
  curves?: unknown;
  preds?: Array<Record<string, unknown>>;
  error?: string;
};

function parseUpload(req: NextApiRequest, uploadDir: string): Promise<ParseResult> {
  return new Promise((resolve, reject) => {
    const form = formidable({ uploadDir, keepExtensions: true });
    form.parse(req, (err, fields, files) => {
      if (err) {
        reject(err);
      } else {
        resolve({ fields, files });
      }
    });
  });
}

function findFile(files: formidable.Files): FormidableFile | null {
  const entry = files.file;
  if (!entry) return null;
  if (Array.isArray(entry)) {
    return entry[0] ?? null;
  }
  return entry as FormidableFile;
}

async function runPythonPredict(bundleDir: string, csvPath: string): Promise<PredictionResponse> {
  const scriptPath = path.join(process.cwd(), "exo_infer.py");
  if (!fs.existsSync(scriptPath)) {
    return { error: "Inference script not found" };
  }
  if (!fs.existsSync(bundleDir)) {
    return { error: "Model bundle directory not found" };
  }

  return new Promise((resolve, reject) => {
    const py =
      process.env.PYTHON_BIN ||
      (process.platform === "win32" ? "python" : "python3");

    const proc = spawn(py, [scriptPath, "--mode", "predict", bundleDir, csvPath]);

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    proc.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `Predictor exited with code ${code}`));
        return;
      }
      try {
        const parsed = JSON.parse(stdout.trim() || "{}");
        resolve(parsed);
      } catch (err) {
        reject(new Error(`Unable to parse predictor output: ${(err as Error).message}\n${stdout}`));
      }
    });
  });
}

function toCsv(preds: Array<Record<string, unknown>>): string {
  if (!preds.length) return "";
  const headers = Object.keys(preds[0]);
  const rows = preds.map((row) =>
    headers
      .map((key) => {
        const value = row[key];
        if (value === null || value === undefined) return "";
        const normalized =
          typeof value === "boolean" ? (value ? "True" : "False") : value;
        const text = String(normalized);
        return text.includes(",") || text.includes("\n") ? `"${text.replace(/"/g, '""')}"` : text;
      })
      .join(",")
  );
  return [headers.join(","), ...rows].join("\n");
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const uploadDir = path.join(process.cwd(), "public", "uploads");
  await fs.promises.mkdir(uploadDir, { recursive: true });

  try {
    const { files } = await parseUpload(req, uploadDir);
    const file = findFile(files);
    if (!file) {
      res.status(400).json({ error: "No file uploaded" });
      return;
    }

    const csvPath = file.filepath
      ? file.filepath
      : file.newFilename
      ? path.join(uploadDir, file.newFilename)
      : null;

    if (!csvPath) {
      res.status(400).json({ error: "Unable to determine uploaded file path" });
      return;
    }

    const bundleDir = path.join(process.cwd(), "exo_bundle_v4");
    const result = await runPythonPredict(bundleDir, csvPath);

    if (result.error) {
      res.status(500).json({ error: result.error });
      return;
    }

    const preds = result.preds ?? [];
    const csv = Array.isArray(preds) ? toCsv(preds as Array<Record<string, unknown>>) : "";

    res.status(200).json({ ...result, csv });
  } catch (err) {
    console.error("Prediction upload error", err);
    res.status(500).json({ error: (err as Error).message });
  }
}