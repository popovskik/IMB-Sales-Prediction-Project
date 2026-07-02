import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";

/** Only one bubble may be open across the app; opening a new one closes the last. */
let closeActive: (() => void) | null = null;

type Pos = { top: number; left: number; arrowLeft: number; above: boolean };

/** Small "i" button that explains a number or concept.
 *  Hover reveals the bubble transiently (it closes when the pointer leaves);
 *  a click pins it open until you close it, tap outside, or press Escape.
 *  The bubble is position: fixed and clamped to the viewport, so it can never
 *  be clipped by a card edge or a horizontally scrolled table; it repositions
 *  on every scroll and resize so it stays attached to its icon. */
export function InfoTip({
  children,
  placement = "top",
}: {
  children: ReactNode;
  placement?: "top" | "bottom";
  /** Kept for call-site compatibility; viewport clamping supersedes it. */
  align?: "center" | "end";
}) {
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [pos, setPos] = useState<Pos | null>(null);
  const rootRef = useRef<HTMLSpanElement>(null);
  const bubbleRef = useRef<HTMLDivElement>(null);

  // Stable, self-referencing close so it can clear the single-open slot.
  const closeRef = useRef<(() => void) | null>(null);
  if (!closeRef.current) {
    closeRef.current = () => {
      setOpen(false);
      setPinned(false);
      if (closeActive === closeRef.current) closeActive = null;
    };
  }
  const closeSelf = closeRef.current;

  function openTip() {
    if (open) return;
    closeActive?.();          // close whichever other bubble is showing
    closeActive = closeSelf;
    setOpen(true);
  }

  // Hover: show transiently. Leaving hides it unless a click has pinned it.
  function handleEnter() {
    if (!pinned) openTip();
  }
  function handleLeave() {
    if (!pinned) closeSelf();
  }

  // Click: pin it open, or unpin+close if already pinned.
  function handleClick() {
    if (pinned) {
      closeSelf();
    } else {
      openTip();
      setPinned(true);
    }
  }

  // Place the bubble next to the icon: centred when it fits, clamped to the
  // viewport when it doesn't, flipped above/below to whichever side has room.
  // The capture-phase scroll listener also fires for inner scrollers (the
  // overflow-x tables), so the bubble follows its icon while they scroll.
  useLayoutEffect(() => {
    if (!open) {
      setPos(null);
      return;
    }
    const place = () => {
      const icon = rootRef.current;
      const bubble = bubbleRef.current;
      if (!icon || !bubble) return;
      const t = icon.getBoundingClientRect();
      const bw = bubble.offsetWidth;
      const bh = bubble.offsetHeight;
      const margin = 8;
      const gap = 9;
      const left = Math.max(margin, Math.min(t.left + t.width / 2 - bw / 2, window.innerWidth - bw - margin));
      const spaceAbove = t.top - gap - margin;
      const spaceBelow = window.innerHeight - t.bottom - gap - margin;
      let above = placement === "top";
      if (above && bh > spaceAbove && spaceBelow >= spaceAbove) above = false;
      else if (!above && bh > spaceBelow && spaceAbove > spaceBelow) above = true;
      const top = above ? t.top - gap - bh : t.bottom + gap;
      const arrowLeft = Math.max(14, Math.min(t.left + t.width / 2 - left, bw - 14));
      setPos({ top, left, arrowLeft, above });
    };
    place();
    window.addEventListener("scroll", place, true);
    window.addEventListener("resize", place);
    return () => {
      window.removeEventListener("scroll", place, true);
      window.removeEventListener("resize", place);
    };
  }, [open, placement]);

  // When pinned: tap outside or press Escape to close. (Hover-only bubbles
  // close on mouse-leave, so these listeners only matter once pinned.)
  useEffect(() => {
    if (!pinned) return;
    const onDown = (e: PointerEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) closeSelf();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeSelf();
    };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [pinned, closeSelf]);

  // Release the single-open slot if this tip unmounts while open.
  useEffect(
    () => () => {
      if (closeActive === closeSelf) closeActive = null;
    },
    [closeSelf],
  );

  return (
    <span className="infotip" ref={rootRef} onMouseEnter={handleEnter} onMouseLeave={handleLeave}>
      <button
        type="button"
        className="infotip-icon"
        aria-expanded={open}
        aria-label="Explanation"
        onClick={handleClick}
        onFocus={openTip}
        onBlur={() => { if (!pinned) closeSelf(); }}
      >
        i
      </button>
      {open && (
        <div
          className="infotip-bubble"
          role="tooltip"
          ref={bubbleRef}
          style={{ top: pos?.top ?? 0, left: pos?.left ?? 0, visibility: pos ? "visible" : "hidden" }}
        >
          {pinned && (
            <button type="button" className="infotip-close" aria-label="Close" onClick={closeSelf}>
              ×
            </button>
          )}
          {children}
          {pos && (
            <span
              className={`infotip-arrow ${pos.above ? "infotip-arrow--down" : "infotip-arrow--up"}`}
              style={{ left: pos.arrowLeft }}
            />
          )}
        </div>
      )}
    </span>
  );
}
