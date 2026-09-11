import { createHmac } from "node:crypto";
import { describe, it, expect, vi } from "vitest";
import {
  SESSION_COOKIE,
  clearedSessionCookie,
  createSessionToken,
  nowSeconds,
  sessionCookie,
  verifySessionToken,
} from "@/lib/server/session";
import { SESSION_TTL_SECONDS } from "@/lib/shared/limits";

const SECRET = "test-secret-0123456789abcdef0123456789abcdef";
const OTHER_SECRET = "test-secret-fedcba9876543210fedcba9876543210";
const NOW = 1_800_000_000; // fixed "now" in seconds; tokens never touch the wall clock

/** Build a token whose signature is valid for SECRET but whose claims are chosen by the test. */
function signedToken(claims: unknown, secret = SECRET): string {
  const payload = Buffer.from(JSON.stringify(claims), "utf8").toString("base64url");
  const signature = createHmac("sha256", secret).update(payload).digest("base64url");
  return `${payload}.${signature}`;
}

function decodeClaims(token: string): Record<string, unknown> {
  const [payload] = token.split(".") as [string];
  return JSON.parse(Buffer.from(payload, "base64url").toString("utf8")) as Record<string, unknown>;
}

describe("createSessionToken / verifySessionToken", () => {
  it("round-trips a freshly minted token with the expected claims", () => {
    const token = createSessionToken(SECRET, NOW);
    expect(token).toMatch(/^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/);

    const verdict = verifySessionToken(SECRET, token, NOW + 5);
    expect(verdict.ok).toBe(true);
    if (!verdict.ok) throw new Error("unreachable");
    expect(verdict.claims.v).toBe(1);
    expect(verdict.claims.sid).toMatch(/^[a-f0-9]{32}$/);
    expect(verdict.claims.iat).toBe(NOW);
    expect(verdict.claims.exp).toBe(NOW + SESSION_TTL_SECONDS);
  });

  it("mints a distinct session id every time", () => {
    const a = decodeClaims(createSessionToken(SECRET, NOW));
    const b = decodeClaims(createSessionToken(SECRET, NOW));
    expect(a.sid).not.toBe(b.sid);
  });

  it("rejects a token signed with a different secret", () => {
    const token = createSessionToken(SECRET, NOW);
    expect(verifySessionToken(OTHER_SECRET, token, NOW)).toEqual({ ok: false, reason: "bad_signature" });
  });

  it("rejects a token whose payload was tampered with after signing", () => {
    const token = createSessionToken(SECRET, NOW);
    const [, signature] = token.split(".") as [string, string];
    const claims = decodeClaims(token);
    const forged = Buffer.from(JSON.stringify({ ...claims, exp: NOW + 10 * SESSION_TTL_SECONDS }), "utf8").toString(
      "base64url",
    );
    expect(verifySessionToken(SECRET, `${forged}.${signature}`, NOW)).toEqual({ ok: false, reason: "bad_signature" });
  });

  it("rejects an expired token, including at the exact expiry second", () => {
    const token = createSessionToken(SECRET, NOW);
    expect(verifySessionToken(SECRET, token, NOW + SESSION_TTL_SECONDS)).toEqual({ ok: false, reason: "expired" });
    expect(verifySessionToken(SECRET, token, NOW + SESSION_TTL_SECONDS + 1)).toEqual({ ok: false, reason: "expired" });
    expect(verifySessionToken(SECRET, token, NOW + SESSION_TTL_SECONDS - 1).ok).toBe(true);
  });

  it("rejects a token whose lifetime exceeds the session TTL as malformed", () => {
    const token = createSessionToken(SECRET, NOW, SESSION_TTL_SECONDS * 2);
    expect(verifySessionToken(SECRET, token, NOW)).toEqual({ ok: false, reason: "malformed" });
  });

  it("rejects a token issued in the future as not_yet_valid", () => {
    const token = createSessionToken(SECRET, NOW + 3600);
    expect(verifySessionToken(SECRET, token, NOW)).toEqual({ ok: false, reason: "not_yet_valid" });
  });

  it("treats a missing token as missing", () => {
    expect(verifySessionToken(SECRET, undefined, NOW)).toEqual({ ok: false, reason: "missing" });
    expect(verifySessionToken(SECRET, null, NOW)).toEqual({ ok: false, reason: "missing" });
    expect(verifySessionToken(SECRET, "", NOW)).toEqual({ ok: false, reason: "missing" });
  });

  it.each([
    ["no dot", "abcdefghijklmnop"],
    ["two dots", "abc.def.ghi"],
    ["non-base64url characters", "abc+def/ghi=.sig"],
    ["whitespace", "abc def.sig"],
    ["longer than 512 characters", `${"a".repeat(600)}.${"b".repeat(43)}`],
  ])("rejects a malformed string (%s) without throwing", (_label, token) => {
    expect(() => verifySessionToken(SECRET, token, NOW)).not.toThrow();
    expect(verifySessionToken(SECRET, token, NOW)).toEqual({ ok: false, reason: "malformed" });
  });

  it("rejects a correctly signed payload that is not a claims object", () => {
    const notJson = Buffer.from("not json at all", "utf8").toString("base64url");
    const sig = createHmac("sha256", SECRET).update(notJson).digest("base64url");
    expect(verifySessionToken(SECRET, `${notJson}.${sig}`, NOW)).toEqual({ ok: false, reason: "malformed" });

    const sid = "0".repeat(32);
    expect(verifySessionToken(SECRET, signedToken({ v: 2, sid, iat: NOW, exp: NOW + 60 }), NOW)).toEqual({
      ok: false,
      reason: "malformed",
    });
    expect(verifySessionToken(SECRET, signedToken({ v: 1, sid: "short", iat: NOW, exp: NOW + 60 }), NOW)).toEqual({
      ok: false,
      reason: "malformed",
    });
    expect(verifySessionToken(SECRET, signedToken({ v: 1, sid, iat: "0", exp: NOW + 60 }), NOW)).toEqual({
      ok: false,
      reason: "malformed",
    });
  });
});

describe("sessionCookie / clearedSessionCookie", () => {
  it("sets the hardened attributes and defaults maxAge to the session TTL", () => {
    const cookie = sessionCookie("tok", true);
    expect(cookie).toEqual({
      name: SESSION_COOKIE,
      value: "tok",
      httpOnly: true,
      secure: true,
      sameSite: "lax",
      path: "/",
      maxAge: SESSION_TTL_SECONDS,
    });
    expect(sessionCookie("tok", true, 30).maxAge).toBe(30);
  });

  it("follows the secure argument in both directions", () => {
    expect(sessionCookie("tok", false).secure).toBe(false);
    expect(sessionCookie("tok", true).secure).toBe(true);
    expect(clearedSessionCookie(false).secure).toBe(false);
    expect(clearedSessionCookie(true).secure).toBe(true);
  });

  it("clears the cookie with an empty value and a zero max-age on the same name and path", () => {
    const cleared = clearedSessionCookie(true);
    expect(cleared).toEqual({
      name: SESSION_COOKIE,
      value: "",
      httpOnly: true,
      secure: true,
      sameSite: "lax",
      path: "/",
      maxAge: 0,
    });
  });
});

describe("nowSeconds", () => {
  it("returns whole seconds derived from Date.now()", () => {
    vi.spyOn(Date, "now").mockReturnValue(1_800_000_000_999);
    expect(nowSeconds()).toBe(1_800_000_000);
  });
});
