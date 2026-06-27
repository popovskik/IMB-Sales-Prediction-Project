import type { ReactNode } from "react";

/** Small "i" affordance that reveals an explanatory bubble on hover/focus.
 *  Keeps the UI professional and direct while making every number and concept
 *  self-explaining for a non-technical manager. */
export function InfoTip({
  children,
  placement = "top",
  align = "center",
}: {
  children: ReactNode;
  placement?: "top" | "bottom";
  align?: "center" | "end";
}) {
  const cls = `infotip infotip--${placement}${align === "end" ? " infotip--end" : ""}`;
  return (
    <span className={cls} tabIndex={0}>
      <span className="infotip-icon" aria-hidden="true">i</span>
      <span className="infotip-bubble" role="tooltip">{children}</span>
    </span>
  );
}
