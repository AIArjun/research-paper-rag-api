"use client";

import { useEffect, useState } from "react";

/** Whole seconds since `startedAt` (real wall-clock time, refreshed each second). */
export function useElapsed(startedAt: number | null): number {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (startedAt === null) {
      setSeconds(0);
      return;
    }
    const tick = () => setSeconds(Math.max(0, Math.floor((Date.now() - startedAt) / 1000)));
    tick();
    const id = window.setInterval(tick, 1000);
    return () => window.clearInterval(id);
  }, [startedAt]);
  return seconds;
}
