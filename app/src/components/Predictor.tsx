import { useRef, useState } from "react";
import { fallbackFromBaked, predict } from "../api";
import type { DailyRecord, PredictResponse } from "../types";
import { InfoTip } from "./InfoTip";

type Status = "idle" | "loading" | "live" | "fallback";

const iso = (d: Date) => d.toISOString().slice(0, 10);

/** Quick-pick dates: real planning dates (today / next Friday) exercise the
 *  model's any-future-date extrapolation; the 2015 Friday stays inside the
 *  training year and works offline via the baked predictions. */
function quickPicks(): { label: string; date: string }[] {
  const today = new Date();
  const nextFriday = new Date(today);
  nextFriday.setDate(today.getDate() + ((5 - today.getDay() + 7) % 7 || 7));
  return [
    { label: "Today", date: iso(today) },
    { label: "Next Friday", date: iso(nextFriday) },
    { label: "Fri, Jul 3 '15", date: "2015-07-03" },
  ];
}

export function Predictor({ daily }: { daily: DailyRecord[] }) {
  const [date, setDate] = useState("2015-07-03");
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [note, setNote] = useState<string>("");
  const [slow, setSlow] = useState(false);          // true once the request has been pending >3s
  const slowTimer = useRef<number | null>(null);

  const valid = /^\d{4}-\d{2}-\d{2}$/.test(date);

  async function run() {
    if (!valid) return;
    setStatus("loading");
    setSlow(false);
    setNote("");
    // Only escalate to the "waking the server" message if the call is genuinely slow,
    // so fast responses don't flash a scary cold-start warning.
    slowTimer.current = window.setTimeout(() => setSlow(true), 3000);
    try {
      const r = await predict(date);
      setResult(r);
      setStatus("live");
    } catch {
      // API unreachable or cold-starting — fall back to the baked predictions.
      const fb = fallbackFromBaked(date, daily);
      if (fb) {
        setResult(fb);
        setStatus("fallback");
        setNote("Live predictor is waking up or unavailable — showing the pre-computed estimate for this date.");
      } else {
        setResult(null);
        setStatus("fallback");
        setNote("Live predictor is unavailable, and pre-computed estimates only cover 2015. Retry in a moment, or pick a 2015 date.");
      }
    } finally {
      if (slowTimer.current) { clearTimeout(slowTimer.current); slowTimer.current = null; }
      setSlow(false);
    }
  }

  return (
    <section className="card">
      <div className="eyebrow">Predictor</div>
      <h2 style={{ fontSize: 20, margin: "4px 0 6px" }}>Forecast a day</h2>
      <p style={{ color: "var(--ink-2)", fontSize: 13, margin: "0 0 14px" }}>
        Pick any date — past or future. The model needs only the calendar, so it can forecast
        next Friday as easily as a 2015 day (future dates carry an extrapolation note).
      </p>

      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <input
          type="date"
          value={date}
          min="2015-01-01"
          max="2027-12-31"
          onChange={(e) => setDate(e.target.value)}
          aria-label="Date to forecast"
        />
        <button className="btn" onClick={run} disabled={!valid || status === "loading"}>
          {status === "loading" ? "Forecasting…" : "Forecast"}
        </button>
        {!valid && <span style={{ color: "var(--red)", fontSize: 13 }}>Pick a valid date.</span>}
      </div>

      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginTop: 10 }}>
        <span style={{ color: "var(--ink-3)", fontSize: 12 }}>Try:</span>
        {quickPicks().map((p) => (
          <button
            key={p.label}
            onClick={() => setDate(p.date)}
            style={{
              background: p.date === date ? "var(--orange-soft)" : "var(--card-muted)",
              border: `1px solid ${p.date === date ? "var(--orange)" : "var(--line)"}`,
              color: p.date === date ? "var(--orange-deep)" : "var(--ink-2)",
              borderRadius: 999, padding: "4px 12px", fontSize: 12, fontWeight: 600,
              fontFamily: "inherit", cursor: "pointer",
            }}
          >
            {p.label}
          </button>
        ))}
      </div>

      {status === "loading" && (
        <p style={{ color: "var(--ink-2)", marginTop: 14 }}>
          {slow
            ? "Waking the prediction server — the first request can take up to ~30s on a free-tier host. Hang tight…"
            : "Forecasting…"}
        </p>
      )}

      {result && (status === "live" || status === "fallback") && (
        <div
          style={{
            marginTop: 18, padding: 18, borderRadius: "var(--r-md)",
            background: status === "fallback" ? "var(--card-muted)" : "var(--orange-soft)",
            border: status === "fallback" ? "1px dashed var(--ink-3)" : "1px solid var(--orange)",
          }}
        >
          <div className="eyebrow" style={{ display: "flex", alignItems: "center" }}>
            Predicted daily revenue
            <InfoTip>The model's estimate of total revenue for the chosen date, based only on the calendar (weekday, month, season). A planning ballpark, not a guarantee.</InfoTip>
          </div>
          <div style={{ fontFamily: "Poppins, sans-serif", fontSize: 28, fontWeight: 800, color: "var(--ink)" }}>
            ${Math.round(result.predicted_revenue).toLocaleString()}
          </div>
          <div style={{ marginTop: 8, display: "flex", gap: 12, alignItems: "center" }}>
            <span className={result.high_demand.label === "High" ? "badge badge-high" : "badge badge-normal"}>
              {result.high_demand.label === "High" ? "High demand" : "Normal demand"}
            </span>
            {result.high_demand.probability != null && (
              <span style={{ color: "var(--ink-2)", fontSize: 13, display: "flex", alignItems: "center" }}>
                {Math.round(result.high_demand.probability * 100)}% confidence
                <InfoTip>How sure the model is that this is a high-demand day. 50% is a coin-flip; closer to 100% means more confident.</InfoTip>
              </span>
            )}
            {status === "fallback" && (
              <span className="badge" style={{ background: "var(--line)", color: "var(--ink-2)" }}>offline estimate</span>
            )}
          </div>
          {result.out_of_training_range && (
            <p style={{ color: "var(--amber)", fontSize: 13, marginTop: 10, marginBottom: 0 }}>
              Note: this date is outside the model's 2015 training year — treat as a rough extrapolation.
            </p>
          )}
        </div>
      )}

      {note && <p style={{ color: "var(--ink-2)", fontSize: 13, marginTop: 12, marginBottom: 0 }}>{note}</p>}
    </section>
  );
}
