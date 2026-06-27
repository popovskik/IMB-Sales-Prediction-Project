import type { ReactNode } from "react";
import type { Charts } from "../types";

const money = (v: number) => `$${Math.round(v).toLocaleString()}`;

export function Layout({ children, summary }: { children: ReactNode; summary?: Charts["summary"] }) {
  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      {/* Dark sidebar — Around dashboard shell */}
      <aside
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
          {["Live Orders", "Menu Performance", "Reviews", "Staff"].map((s) => (
            <span key={s} style={{ color: "rgba(255,255,255,0.55)", padding: "9px 12px" }}>{s}</span>
          ))}
        </nav>
      </aside>

      {/* Main content */}
      <main style={{ flex: 1, padding: "28px clamp(20px, 4vw, 48px)", maxWidth: "100%" }}>
        <header style={{ marginBottom: 24 }}>
          <div className="eyebrow">Demand Analytics · 2015</div>
          <h1 style={{ fontSize: 28, marginTop: 4 }}>Sales Statistics</h1>
        </header>

        {summary && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 14, marginBottom: 24 }}>
            <Stat label="Total revenue" value={money(summary.total_revenue)} />
            <Stat label="Avg daily revenue" value={money(summary.mean_daily_revenue)} />
            <Stat label="Avg daily orders" value={summary.mean_daily_orders.toFixed(0)} />
            <Stat label="High-demand days" value={`${Math.round(summary.high_demand_share * 100)}%`} />
          </div>
        )}

        <div style={{ display: "flex", flexDirection: "column", gap: 24, maxWidth: "var(--content-max)" }}>
          {children}
        </div>
      </main>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="card" style={{ padding: 16 }}>
      <div className="eyebrow">{label}</div>
      <div style={{ fontFamily: "Poppins, sans-serif", fontSize: 24, fontWeight: 700, marginTop: 6 }}>{value}</div>
    </div>
  );
}
