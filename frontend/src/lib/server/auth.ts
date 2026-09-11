import { createHash, timingSafeEqual } from "node:crypto";

const MAX_PASSCODE_INPUT_CHARS = 512;

/** Constant-time passcode comparison over fixed-length digests. */
export function passcodeMatches(supplied: unknown, expected: string): boolean {
  if (typeof supplied !== "string" || supplied.length === 0 || supplied.length > MAX_PASSCODE_INPUT_CHARS) {
    return false;
  }
  const a = createHash("sha256").update(supplied, "utf8").digest();
  const b = createHash("sha256").update(expected, "utf8").digest();
  return timingSafeEqual(a, b);
}

/**
 * Best-effort, per-instance login throttle. Serverless instances do not share
 * memory, so this slows brute force on one instance; the passcode's length is
 * the real defence. The map is hard-capped so unique keys cannot grow it
 * without bound.
 */
export interface Throttle {
  check(key: string, nowMs: number): { allowed: boolean; retryAfterSeconds: number };
  recordFailure(key: string, nowMs: number): void;
  reset(): void;
  /** Number of tracked keys (bounded by maxEntries). */
  size(): number;
}

export function createLoginThrottle(maxFailures = 10, windowMs = 15 * 60 * 1000, maxEntries = 5000): Throttle {
  const failures = new Map<string, { count: number; windowStart: number }>();
  /** Hard ceiling on tracked keys: expired entries go first, then the oldest windows. */
  const enforceCeiling = (nowMs: number) => {
    if (failures.size < maxEntries) return;
    for (const [key, entry] of failures) {
      if (nowMs - entry.windowStart > windowMs) failures.delete(key);
    }
    if (failures.size < maxEntries) return;
    const oldestFirst = [...failures.entries()].sort((a, b) => a[1].windowStart - b[1].windowStart);
    const excess = failures.size - maxEntries + 1;
    for (const [key] of oldestFirst.slice(0, excess)) failures.delete(key);
  };
  return {
    check(key, nowMs) {
      const entry = failures.get(key);
      if (!entry || nowMs - entry.windowStart > windowMs) return { allowed: true, retryAfterSeconds: 0 };
      if (entry.count < maxFailures) return { allowed: true, retryAfterSeconds: 0 };
      return { allowed: false, retryAfterSeconds: Math.max(1, Math.ceil((entry.windowStart + windowMs - nowMs) / 1000)) };
    },
    recordFailure(key, nowMs) {
      const entry = failures.get(key);
      if (entry && nowMs - entry.windowStart <= windowMs) {
        entry.count += 1;
        return;
      }
      enforceCeiling(nowMs);
      failures.set(key, { count: 1, windowStart: nowMs });
    },
    reset() {
      failures.clear();
    },
    size() {
      return failures.size;
    },
  };
}

export const loginThrottle = createLoginThrottle();

/** Derive a throttle key from the proxy-supplied client address (best effort only). */
export function clientKey(headers: Headers): string {
  const forwarded = headers.get("x-forwarded-for")?.split(",")[0]?.trim() || headers.get("x-real-ip")?.trim() || "";
  return forwarded.slice(0, 64) || "unknown";
}
