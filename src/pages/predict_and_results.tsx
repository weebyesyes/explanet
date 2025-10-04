import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState("");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault(); // 👈 stops reload
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch('/api/upload', {
      method: "POST",
      body: formData
    });

    if (res.status === 200) {
      setStatus("Success");
    } else {
      setStatus("Failed");
    }
  }

  return (
    <div style={{justifyItems: 'center', alignItems: 'center'}}>
      <p>AI for Hackathon Competition NASA 2025</p>
      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <button type="submit">submit</button>
      </form>
      <p>{status}</p>
    </div>
  );
}
