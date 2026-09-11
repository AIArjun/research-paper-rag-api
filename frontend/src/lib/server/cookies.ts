import type { CookieAttributes } from "./session";

/** Serialize a cookie for a Set-Cookie header (attributes are fixed by session.ts). */
export function serializeCookie(cookie: CookieAttributes): string {
  const parts = [
    `${cookie.name}=${cookie.value}`,
    `Path=${cookie.path}`,
    `Max-Age=${cookie.maxAge}`,
    `SameSite=${cookie.sameSite === "lax" ? "Lax" : "Strict"}`,
    "HttpOnly",
  ];
  if (cookie.secure) parts.push("Secure");
  return parts.join("; ");
}
