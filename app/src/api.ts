import type { DailyRecord, PredictResponse, Predictions } from "./types";

const API_BASE = import.meta.env.VITE_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export async function loadPredictions(): Promise<Predictions> {
  const res = await fetch("/predictions.json");
  if (!res.ok) throw new Error(`predictions.json ${res.status}`);
  return res.json();
}

/** Call the live API. Times out so a cold/asleep Railway service falls back fast. */
export async function predict(date: string, timeoutMs = 25000): Promise<PredictResponse> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ date }),
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`/predict ${res.status}`);
    return res.json();
  } finally {
    clearTimeout(t);
  }
}

/** Offline fallback: look up the baked prediction for a date in predictions.json. */
export function fallbackFromBaked(date: string, daily: DailyRecord[]): PredictResponse | null {
  const row = daily.find((d) => d.date === date);
  if (!row) return null;
  return {
    date,
    predicted_revenue: row.predicted_revenue,
    high_demand: {
      label: row.high_demand_pred ? "High" : "Normal",
      probability: row.high_demand_prob,
    },
    out_of_training_range: !date.startsWith("2015"),
  };
}
