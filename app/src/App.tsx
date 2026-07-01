import { useEffect, useState } from "react";
import { loadPredictions } from "./api";
import type { Predictions } from "./types";
import { Layout } from "./components/Layout";
import { Predictor } from "./components/Predictor";
import { About } from "./components/About";
import { EdaCharts } from "./components/EdaCharts";
import { Scorecard } from "./components/Scorecard";

export default function App() {
  const [data, setData] = useState<Predictions | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadPredictions().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return (
      <Layout>
        <div className="card" role="alert">
          <h3>Could not load dashboard data</h3>
          <p style={{ color: "var(--ink-2)" }}>{error}</p>
        </div>
      </Layout>
    );
  }

  if (!data) {
    return (
      <Layout>
        <div className="card">Loading analytics…</div>
      </Layout>
    );
  }

  return (
    <Layout summary={data.charts.summary}>
      {/* Predictor first — the one interactive tool for the owner persona. */}
      <Predictor daily={data.daily} />
      <About />
      <EdaCharts charts={data.charts} />
      <Scorecard leaderboard={data.leaderboard} models={data.models} diagnostics={data.model_diagnostics} />
    </Layout>
  );
}
