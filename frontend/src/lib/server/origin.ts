/**
 * Origin validation for state-changing requests. The allowlist is built from
 * configuration (see env.ts); request-controlled headers such as Host and
 * X-Forwarded-Host are deliberately not consulted.
 */
export function isTrustedOrigin(headers: Headers, allowedOrigins: readonly string[]): boolean {
  const origin = headers.get("origin");
  if (origin !== null) {
    if (origin === "null") return false;
    let normalized: string;
    try {
      normalized = new URL(origin).origin;
    } catch {
      return false;
    }
    return allowedOrigins.includes(normalized);
  }
  // Browsers always send Origin on cross-site and on same-origin POST requests
  // made with fetch(); Sec-Fetch-Site is a forbidden (browser-controlled)
  // header, so accepting it here does not admit a cross-site page.
  return headers.get("sec-fetch-site") === "same-origin";
}
