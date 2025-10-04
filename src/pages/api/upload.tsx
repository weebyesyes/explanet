import fs from "fs";
import path from "path";
import formidable from "formidable";
import { NextApiRequest, NextApiResponse } from "next";

export const config = {
  api: {
    bodyParser: false,
  },
};

// ✅ Upload handler
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  // Ensure upload directory exists
  const uploadDir = path.join(process.cwd(), "public", "uploads");
  await fs.promises.mkdir(uploadDir, { recursive: true });

  // Create formidable instance
  const form = formidable({
    uploadDir, // where files are saved
    keepExtensions: true, // keep .jpg, .png, etc.
  });

  // Parse the incoming form
  form.parse(req, (err: any, fields: any, files: any) => {
    if (err) {
      console.error("Formidable error:", err);
      res.status(500).json({ error: "Error parsing form data" });
      return;
    }

    console.log("Uploaded files:", files);
    res.status(200).json({ success: true, files });
  });
}