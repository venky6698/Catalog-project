import React, { useState } from "react";
import axios from "axios";

function UploadForm() {
  const [file, setFile] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => setFile(e.target.files[0]);
  const handlePromptChange = (e) => setPrompt(e.target.value);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("prompt", prompt);

    try {
      setLoading(true);
      const response = await axios.post("http://localhost:5000/upload", formData);

      // Get raw result
      const rawData = response.data?.results;

      // Set raw JSON output directly
      setResults(rawData);
    } catch (err) {
      console.error("Upload failed:", err);
      alert("Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit} style={{ marginBottom: "1rem" }}>
        <input type="file" onChange={handleFileChange} accept="image/*,.pdf" required />
        <br />
        <textarea
          placeholder="Enter custom prompt to guide the model..."
          value={prompt}
          onChange={handlePromptChange}
          rows={3}
          style={{ width: "100%", marginTop: "1rem" }}
        />
        <button type="submit" style={{ marginTop: "1rem" }}>
          Upload & Analyze
        </button>
      </form>

      {loading && <p>Analyzing...</p>}

      {results && (
        <div style={{ marginTop: "1rem", backgroundColor: "#f7f7f7", padding: "1rem", borderRadius: "5px" }}>
          <h3>Model Response:</h3>
          <pre style={{ whiteSpace: "pre-wrap", marginTop: "1rem", wordBreak: "break-word" }}>
            {results}
          </pre>
        </div>
      )}
    </div>
  );
}

export default UploadForm;
