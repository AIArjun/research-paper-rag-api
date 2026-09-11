import { describe, it, expect } from "vitest";
import { clientKey, createLoginThrottle, loginThrottle, passcodeMatches } from "@/lib/server/auth";

const PASSCODE = "test-passcode-0123456789abcdef";

describe("passcodeMatches", () => {
  it("accepts only the exact passcode", () => {
    expect(passcodeMatches(PASSCODE, PASSCODE)).toBe(true);
  });

  it("rejects a value of a different length", () => {
    expect(passcodeMatches(PASSCODE.slice(0, -1), PASSCODE)).toBe(false);
    expect(passcodeMatches(`${PASSCODE}x`, PASSCODE)).toBe(false);
  });

  it("rejects a near miss of the same length", () => {
    const nearMiss = `${PASSCODE.slice(0, -1)}${PASSCODE.endsWith("f") ? "e" : "f"}`;
    expect(nearMiss).toHaveLength(PASSCODE.length);
    expect(passcodeMatches(nearMiss, PASSCODE)).toBe(false);
    expect(passcodeMatches(PASSCODE.toUpperCase(), PASSCODE)).toBe(false);
  });

  it.each([
    ["number", 1234],
    ["null", null],
    ["undefined", undefined],
    ["object", { toString: () => PASSCODE }],
    ["array", [PASSCODE]],
    ["boolean", true],
  ])("rejects a non-string supplied value (%s) without throwing", (_label, supplied) => {
    expect(() => passcodeMatches(supplied, PASSCODE)).not.toThrow();
    expect(passcodeMatches(supplied, PASSCODE)).toBe(false);
  });

  it("rejects an empty string", () => {
    expect(passcodeMatches("", PASSCODE)).toBe(false);
    expect(passcodeMatches("", "")).toBe(false);
  });

  it("rejects input longer than 512 characters even when it would otherwise match", () => {
    const long = "a".repeat(513);
    expect(passcodeMatches(long, long)).toBe(false);
    const max = "a".repeat(512);
    expect(passcodeMatches(max, max)).toBe(true);
  });
});

describe("createLoginThrottle", () => {
  const MAX = 3;
  const WINDOW_MS = 60_000;
  const T0 = 1_800_000_000_000;

  it("allows attempts until maxFailures is reached, then blocks with a positive retry delay", () => {
    const throttle = createLoginThrottle(MAX, WINDOW_MS);
    expect(throttle.check("k", T0)).toEqual({ allowed: true, retryAfterSeconds: 0 });

    for (let i = 0; i < MAX - 1; i += 1) throttle.recordFailure("k", T0 + i);
    expect(throttle.check("k", T0 + MAX)).toEqual({ allowed: true, retryAfterSeconds: 0 });

    throttle.recordFailure("k", T0 + MAX);
    const blocked = throttle.check("k", T0 + MAX);
    expect(blocked.allowed).toBe(false);
    expect(blocked.retryAfterSeconds).toBeGreaterThan(0);
    expect(blocked.retryAfterSeconds).toBeLessThanOrEqual(WINDOW_MS / 1000);
  });

  it("counts the retry delay down from the start of the window and never reports zero while blocked", () => {
    const throttle = createLoginThrottle(MAX, WINDOW_MS);
    for (let i = 0; i < MAX; i += 1) throttle.recordFailure("k", T0);
    expect(throttle.check("k", T0).retryAfterSeconds).toBe(WINDOW_MS / 1000);
    expect(throttle.check("k", T0 + 30_000).retryAfterSeconds).toBe(30);
    expect(throttle.check("k", T0 + 59_500).retryAfterSeconds).toBe(1);
    const atBoundary = throttle.check("k", T0 + WINDOW_MS);
    expect(atBoundary.allowed).toBe(false);
    expect(atBoundary.retryAfterSeconds).toBeGreaterThanOrEqual(1);
  });

  it("re-allows attempts once the window has expired and starts a fresh count", () => {
    const throttle = createLoginThrottle(MAX, WINDOW_MS);
    for (let i = 0; i < MAX; i += 1) throttle.recordFailure("k", T0);
    expect(throttle.check("k", T0).allowed).toBe(false);

    const later = T0 + WINDOW_MS + 1;
    expect(throttle.check("k", later)).toEqual({ allowed: true, retryAfterSeconds: 0 });

    // A failure after expiry opens a new window with count 1 rather than resuming the old count.
    throttle.recordFailure("k", later);
    expect(throttle.check("k", later).allowed).toBe(true);
    throttle.recordFailure("k", later);
    throttle.recordFailure("k", later);
    expect(throttle.check("k", later).allowed).toBe(false);
  });

  it("tracks keys independently", () => {
    const throttle = createLoginThrottle(MAX, WINDOW_MS);
    for (let i = 0; i < MAX; i += 1) throttle.recordFailure("blocked", T0);
    expect(throttle.check("blocked", T0).allowed).toBe(false);
    expect(throttle.check("other", T0).allowed).toBe(true);
  });

  it("reset clears every recorded failure", () => {
    const throttle = createLoginThrottle(MAX, WINDOW_MS);
    for (let i = 0; i < MAX; i += 1) throttle.recordFailure("k", T0);
    expect(throttle.check("k", T0).allowed).toBe(false);
    throttle.reset();
    expect(throttle.check("k", T0)).toEqual({ allowed: true, retryAfterSeconds: 0 });
  });

  it("caps the number of tracked keys, evicting expired entries first and then the oldest windows", () => {
    const throttle = createLoginThrottle(MAX, WINDOW_MS, 3);
    throttle.recordFailure("k0", T0);
    throttle.recordFailure("k1", T0 + 1);
    throttle.recordFailure("k2", T0 + 2);
    expect(throttle.size()).toBe(3);

    // Nothing has expired, so the oldest window (k0) makes room for the new key.
    throttle.recordFailure("k3", T0 + 3);
    expect(throttle.size()).toBe(3);
    expect(throttle.check("k0", T0 + 3).allowed).toBe(true);
    for (let i = 0; i < 100; i += 1) throttle.recordFailure(`flood-${i}`, T0 + 4 + i);
    expect(throttle.size()).toBe(3);

    // At the ceiling, an expired entry is dropped before any live window is sacrificed.
    throttle.reset();
    throttle.recordFailure("stale", T0);
    const t1 = T0 + WINDOW_MS;
    for (let i = 0; i < MAX; i += 1) throttle.recordFailure("blockedLive", t1);
    throttle.recordFailure("otherLive", t1);
    expect(throttle.size()).toBe(3);

    throttle.recordFailure("newcomer", t1 + 1);
    expect(throttle.size()).toBe(3);
    expect(throttle.check("stale", t1 + 1).allowed).toBe(true); // the expired key was evicted
    expect(throttle.check("blockedLive", t1 + 1).allowed).toBe(false); // the live block survived
  });

  it("exports a shared throttle instance with the same interface", () => {
    expect(typeof loginThrottle.check).toBe("function");
    expect(typeof loginThrottle.recordFailure).toBe("function");
    expect(typeof loginThrottle.reset).toBe("function");
    expect(typeof loginThrottle.size).toBe("function");
    expect(loginThrottle.check("auth-test-unused-key", T0).allowed).toBe(true);
  });
});

describe("clientKey", () => {
  it("uses the first x-forwarded-for entry, trimmed", () => {
    const headers = new Headers({ "x-forwarded-for": "  203.0.113.5 , 10.0.0.1, 10.0.0.2" });
    expect(clientKey(headers)).toBe("203.0.113.5");
  });

  it("falls back to x-real-ip when x-forwarded-for is absent", () => {
    expect(clientKey(new Headers({ "x-real-ip": "198.51.100.7" }))).toBe("198.51.100.7");
  });

  it("returns \"unknown\" when no client address header is present or it is blank", () => {
    expect(clientKey(new Headers())).toBe("unknown");
    expect(clientKey(new Headers({ "x-forwarded-for": "   " }))).toBe("unknown");
    expect(clientKey(new Headers({ "x-forwarded-for": ", 10.0.0.1" }))).toBe("unknown");
  });

  it("bounds the key length to 64 characters", () => {
    const key = clientKey(new Headers({ "x-forwarded-for": "x".repeat(500) }));
    expect(key).toHaveLength(64);
    expect(key).toBe("x".repeat(64));
  });
});
