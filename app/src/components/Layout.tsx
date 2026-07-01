import type { ReactNode } from "react";
import type { Charts } from "../types";
import { InfoTip } from "./InfoTip";

const money = (v: number) => `$${Math.round(v).toLocaleString()}`;

export function Layout({ children, summary }: { children: ReactNode; summary?: Charts["summary"] }) {
  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      {/* Dark sidebar — Around dashboard shell */}
      <aside
        className="app-sidebar"
        style={{
          width: "var(--sidebar-w)", background: "var(--sidebar)", color: "#fff",
          padding: "24px 20px", flexShrink: 0,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 32 }}>
          <span style={{ width: 18, height: 18, background: "var(--orange)", borderRadius: 5, display: "inline-block" }} />
          <strong style={{ fontFamily: "Poppins, sans-serif", fontSize: 16 }}>Around</strong>
        </div>
        <div className="eyebrow" style={{ color: "rgba(255,255,255,0.45)", marginBottom: 10 }}>Insights</div>
        <nav style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={{ background: "var(--orange)", color: "#fff", padding: "9px 12px", borderRadius: 10, fontWeight: 600 }}>
            Sales Statistics
          </span>
        </nav>
      </aside>

      {/* Main content */}
      <main style={{ flex: 1, padding: "28px clamp(20px, 4vw, 48px)", maxWidth: "100%" }}>
        <header style={{ marginBottom: 24 }}>
          <div className="eyebrow">Demand Analytics · 2015</div>
          <h1 style={{ fontSize: 28, marginTop: 4 }}>Sales Statistics</h1>
        </header>

        {/* Stats and the sections below share ONE max-width container so their
            right edges line up (the stat row used to run wider than the cards below). */}
        <div className="content">
          {summary && (
            <div className="stat-grid">
              <Stat label="Total revenue" value={money(summary.total_revenue)}
                    tip={<>Total money taken across the whole year (2015) — every order's <strong>price × quantity</strong>, summed.</>} />
              <Stat label="Avg daily revenue" value={money(summary.mean_daily_revenue)}
                    tip={<>The average day's revenue across all 365 days. A typical day brings in about this much.</>} />
              <Stat label="Avg daily orders" value={summary.mean_daily_orders.toFixed(0)}
                    tip={<>The average number of orders placed per day across the year.</>} />
              <Stat label="High-demand days" value={`${Math.round(summary.high_demand_share * 100)}%`} align="end"
                    tip={<>Share of days we call <strong>"high demand"</strong> — days whose order count is above the year's average. These are the days to staff and stock up for.</>} />
            </div>
          )}

          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            {children}
          </div>
        </div>
      </main>
    </div>
  );
}

function Stat({ label, value, tip, align }: { label: string; value: string; tip: ReactNode; align?: "center" | "end" }) {
  return (
    <div className="card" style={{ padding: 16 }}>
      <div className="eyebrow" style={{ display: "flex", alignItems: "center" }}>
        {label}
        <InfoTip placement="bottom" align={align}>{tip}</InfoTip>
      </div>
      <div style={{ fontFamily: "Poppins, sans-serif", fontSize: 24, fontWeight: 700, marginTop: 6 }}>{value}</div>
    </div>
  );
}
