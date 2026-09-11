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
 * the real defence.
 */
export interface Throttle {
  check(key: string, nowMs: number): { allowed: boolean; retryAfterSeconds: number };
  recordFailure(key: string, nowMs: number): void;
  reset(): void;
}

export function createLoginThrottle(maxFailures = 10, windowMs = 15 * 60 * 1000): Throttle {
  const failures = new Map<string, { count: number; windowStart: number }>();
  const prune = (nowMs: number) => {
    if (failures.size < 1000) return;
    for (const [key, entry] of failures) {
      if (nowMs - entry.windowStart > windowMs) failures.delete(key);
    }
  };
  return {
    check(key, nowMs) {
      const entry = failures.get(key);
      if (!entry || nowMs - entry.windowStart > windowMs) return { allowed: true, retryAfterSeconds: 0 };
      if (entry.count < maxFailures) return { allowed: true, retryAfterSeconds: 0 };
      return { allowed: false, retryAfterSeconds: Math.max(1, Math.ceil((entry.windowStart + windowMs - nowMs) / 1000)) };
    },
    recordFailure(key, nowMs) {
      prune(nowMs);
      const entry = failures.get(key);
      if (!entry || nowMs - entry.windowStart > windowMs) failures.set(key, { count: 1, windowStart: nowMs });
      else entry.count += 1;
    },
    reset() {
      failures.clear();
    },
  };
}

export const loginThrottle = createLoginThrottle();

/** Derive a throttle key from the proxy-supplied client address (best effort only). */
export function clientKey(headers: Headers): string {
  const forwarded = headers.get("x-forwarded-for") ?? headers.get("x-real-ip") ?? "";
  const first = forwarded.split(",")[0]?.trim() ?? "";
  return first.slice(0, 64) || "unknown";
}
