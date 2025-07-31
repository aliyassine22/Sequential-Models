import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [category, setCategory] = useState("");
  const [modelType, setModelType] = useState("LSTM"); // default model type
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    if (!text.trim()) {
      setError("Please enter some text to classify.");
      return;
    }

    setLoading(true);
    setError("");
    setCategory("");

    try {
      const response = await axios.post("http://127.0.0.1:8000/predict", {
        text: text,
        model_type: modelType, // Send model type
      });
      setCategory(response.data.class);
    } catch (err) {
      setError("Failed to fetch classification. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setText("");
    setCategory("");
    setError("");
  };

  return (
    <div className="container">
      <h2>Text Classifier</h2>
      <textarea
        className="textarea"
        placeholder="Type your text here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      {/* Model selection dropdown */}
      <div className="model-selector">
        <label htmlFor="model">Choose Model:</label>
        <select
          id="model"
          value={modelType}
          onChange={(e) => setModelType(e.target.value)}
        >
          <option value="LSTM">LSTM</option>
          <option value="GRU">GRU</option>
        </select>
      </div>

      <div className="controls">
        <button onClick={handleSubmit} disabled={loading}>
          {loading ? "Classifying..." : "Classify"}
        </button>
        <button onClick={handleClear} disabled={!text && !category}>
          Clear
        </button>
      </div>

      <p className="word-count">
        Word Count: {text.trim().split(/\s+/).filter(Boolean).length}
      </p>

      {error && <p className="error">{error}</p>}
      {category && (
        <p className="result">
          <strong>Predicted Category:</strong> {category}
        </p>
      )}
    </div>
  );
}

export default App;