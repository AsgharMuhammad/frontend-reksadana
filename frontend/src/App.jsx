import React, { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (f) setSelectedFile(f);
  };

  const handleUpload = async () => {
    if (!selectedFile) return alert("Pilih file CSV dulu!");

    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const resp = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        body: formData,
      });

      const json = await resp.json();
      console.log("RESP JSON:", json);

      if (!json.data) {
        alert("Gagal membaca data hasil prediksi dari server!");
        return;
      }

      const merged = json.data.map((d) => ({
        date: d.Tanggal,
        actual: d.Actual,
        predicted: d.Predicted,
      }));

      setChartData(merged);
      setMetrics(json.evaluasi);
    } catch (err) {
      console.error("Upload error:", err);
      alert("Terjadi kesalahan saat memproses file. Silakan coba lagi.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="main-container">
        <div className="header">
          <h1 className="main-title">
            Model Prediksi Harga Reksa Dana Berbasis Data Mining Dengan Algoritma Random Forest
          </h1>
          <p className="author">Muhammad Asghar (F1G121006)</p>
        </div>

        <div className="upload-section">
          <h2 className="upload-title">Upload Data CSV</h2>
          <div className="upload-controls">
            {/* INPUT FILE TERSAMAR — hanya label kustom yang tampak */}
            <input
              type="file"
              id="file-upload"
              accept=".csv"
              onChange={handleFileChange}
              style={{ display: "none" }} // <-- sembunyikan input asli
            />

            {/* Label bertindak sebagai tombol pilih file */}
            <label htmlFor="file-upload" className="file-input-label">
              {selectedFile ? selectedFile.name : "Pilih file CSV"}
            </label>

            <button
              onClick={handleUpload}
              className={`upload-button ${isLoading ? "loading" : ""}`}
              disabled={!selectedFile || isLoading}
            >
              {isLoading ? "Memproses..." : "Upload & Predict"}
            </button>
          </div>
        </div>

        {/* Hasil Evaluasi Model */}
        {metrics && (
          <div className="metrics-section">
            <h2 className="section-title metrics">Hasil Evaluasi Model</h2>
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-label">Mean Absolute Error</div>
                <div className="metric-value">{metrics.MAE ? metrics.MAE.toFixed(4) : "-"}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Mean Squared Error</div>
                <div className="metric-value">{metrics.MSE ? metrics.MSE.toFixed(4) : "-"}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Root Mean Squared Error</div>
                <div className="metric-value">{metrics.RMSE ? metrics.RMSE.toFixed(4) : "-"}</div>
              </div>
            </div>
          </div>
        )}

        {/* Grafik */}
        {chartData.length > 0 && (
          <div className="chart-section">
            <h2 className="section-title chart">Grafik Prediksi vs Aktual</h2>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="actual" stroke="#2563eb" strokeWidth={2} name="Actual" dot={false} />
                  <Line type="monotone" dataKey="predicted" stroke="#dc2626" strokeWidth={2} name="Predicted" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Tabel */}
        {chartData.length > 0 && (
          <div className="table-section">
            <h2 className="section-title table">Tabel Data Aktual vs Prediksi</h2>
            <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>No</th>
                  <th>Tanggal</th>
                  <th>Actual</th>
                  <th>Predicted</th>
                  <th>Error</th>
                </tr>
              </thead>
              <tbody>
                {chartData.map((row, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>{row.date}</td>
                    <td>{row.actual !== null && row.actual !== undefined ? row.actual.toFixed(4) : "-"}</td>
                    <td>{row.predicted !== null && row.predicted !== undefined ? row.predicted.toFixed(4) : "-"}</td>
                    <td>{row.actual !== null && row.predicted !== null ? (row.actual - row.predicted).toFixed(4) : "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
