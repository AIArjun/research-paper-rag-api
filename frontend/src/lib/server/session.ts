import { createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import { SESSION_TTL_SECONDS } from "@/lib/shared/limits";

export const SESSION_COOKIE = "ro_session";
const VERSION = 1;
const MAX_TOKEN_CHARS = 512;
const CLOCK_SKEW_SECONDS = 60;

export interface SessionClaims {
  v: number;
  sid: string;
  iat: number; // seconds
  exp: number; // seconds
}

export type SessionVerdict =
  | { ok: true; claims: SessionClaims }
  | { ok: false; reason: "missing" | "malformed" | "bad_signature" | "expired" | "not_yet_valid" };

function sign(secret: string, payload: string): Buffer {
  return createHmac("sha256", secret).update(payload).digest();
}

export function createSessionToken(secret: string, nowSeconds: number, ttlSeconds = SESSION_TTL_SECONDS): string {
  const claims: SessionClaims = {
    v: VERSION,
    sid: randomBytes(16).toString("hex"),
    iat: nowSeconds,
    exp: nowSeconds + ttlSeconds,
  };
  const payload = Buffer.from(JSON.stringify(claims), "utf8").toString("base64url");
  const signature = sign(secret, payload).toString("base64url");
  return `${payload}.${signature}`;
}

function isClaims(value: unknown): value is SessionClaims {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    v.v === VERSION &&
    typeof v.sid === "string" && /^[a-f0-9]{32}$/.test(v.sid) &&
    typeof v.iat === "number" && Number.isFinite(v.iat) &&
    typeof v.exp === "number" && Number.isFinite(v.exp)
  );
}

export function verifySessionToken(secret: string, token: string | undefined | null, nowSeconds: number): SessionVerdict {
  if (!token) return { ok: false, reason: "missing" };
  if (token.length > MAX_TOKEN_CHARS || !/^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/.test(token)) {
    return { ok: false, reason: "malformed" };
  }
  const [payload, signature] = token.split(".") as [string, string];
  const expected = sign(secret, payload);
  const given = Buffer.from(signature, "base64url");
  if (given.length !== expected.length || !timingSafeEqual(given, expected)) {
    return { ok: false, reason: "bad_signature" };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(Buffer.from(payload, "base64url").toString("utf8"));
  } catch {
    return { ok: false, reason: "malformed" };
  }
  if (!isClaims(parsed)) return { ok: false, reason: "malformed" };
  if (parsed.exp <= nowSeconds) return { ok: false, reason: "expired" };
  if (parsed.iat > nowSeconds + CLOCK_SKEW_SECONDS) return { ok: false, reason: "not_yet_valid" };
  if (parsed.exp - parsed.iat > SESSION_TTL_SECONDS + CLOCK_SKEW_SECONDS) return { ok: false, reason: "malformed" };
  return { ok: true, claims: parsed };
}

export interface CookieAttributes {
  name: string;
  value: string;
  httpOnly: true;
  secure: boolean;
  sameSite: "lax";
  path: "/";
  maxAge: number;
}

export function sessionCookie(value: string, secure: boolean, maxAge = SESSION_TTL_SECONDS): CookieAttributes {
  return { name: SESSION_COOKIE, value, httpOnly: true, secure, sameSite: "lax", path: "/", maxAge };
}

export function clearedSessionCookie(secure: boolean): CookieAttributes {
  return { name: SESSION_COOKIE, value: "", httpOnly: true, secure, sameSite: "lax", path: "/", maxAge: 0 };
}

export function nowSeconds(): number {
  return Math.floor(Date.now() / 1000);
}
