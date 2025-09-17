import React, { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [metrics, setMetrics] = useState(null);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) return alert("Pilih file CSV dulu!");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const resp = await fetch("http://127.0.0.1:8000/upload_predict", {
        method: "POST",
        body: formData,
      });

      const json = await resp.json();
      console.log("RESP JSON:", json);

      // Gabungkan dates, actuals, dan predictions
      const merged = json.dates.map((d, i) => ({
        date: d,
        actual: json.actuals ? json.actuals[i] : null,
        prediction: json.predictions[i],
      }));

      setChartData(merged);
      setMetrics(json.metrics);
    } catch (err) {
      console.error("Upload error:", err);
    }
  };

  return (
    <div className="App" style={{ padding: "20px", background: "#111", color: "white", minHeight: "100vh" }}>
      <h1>Model Prediksi Harga Reksa Dana Berbasis Data Mining Dengan Algoritma Random Forest</h1>
      <h1>Muhammad Asghar (F1G121006)</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload & Predict</button>

      {/* Hasil Evaluasi Model */}
      {metrics && (
        <div style={{ marginTop: "20px" }}>
          <h2>Hasil Evaluasi Model</h2>
          <table border="1" style={{ borderCollapse: "collapse", width: "50%", color: "white" }}>
            <tbody>
              <tr><td>MAE</td><td>{metrics.mae}</td></tr>
              <tr><td>MSE</td><td>{metrics.mse}</td></tr>
              <tr><td>RMSE</td><td>{metrics.rmse}</td></tr>
            </tbody>
          </table>
        </div>
      )}

      {/* Grafik */}
      {chartData.length > 0 ? (
        <div style={{ marginTop: "40px" }}>
          <h2>Grafik Prediksi vs Aktual</h2>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={false} /> {/* tanggal disembunyikan kalau panjang */}
              <YAxis />
              <Tooltip />
              <Legend />
              {chartData[0]?.actual !== null && (
                <Line type="monotone" dataKey="actual" stroke="#8884d8" dot={false} name="Actual" />
              )}
              <Line type="monotone" dataKey="prediction" stroke="#82ca9d" dot={false} name="Prediction" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <p style={{ marginTop: "20px" }}>Belum ada data prediksi.</p>
      )}
      {/* 🔹 Tambahan: Tabel Data Aktual vs Prediksi */}
{chartData.length > 0 && (
  <div style={{ marginTop: "40px" }}>
    <h2>Tabel Data Aktual vs Prediksi</h2>
    <table border="1" style={{ borderCollapse: "collapse", width: "100%", color: "white" }}>
      <thead>
        <tr>
          <th>No</th>
          <th>Tanggal</th>
          <th>Actual</th>
          <th>Prediction</th>
        </tr>
      </thead>
      <tbody>
        {chartData.slice(0, 50).map((row, i) => (
          <tr key={i}>
            <td>{i + 1}</td>
            <td>{row.date}</td>
            <td>{row.actual !== null ? row.actual.toFixed(3) : "-"}</td>
            <td>{row.prediction.toFixed(3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
    <p style={{ marginTop: "10px" }}>*Hanya ditampilkan 50 data pertama.</p>
  </div>
)}
    </div>
      
  );
}

export default App;
