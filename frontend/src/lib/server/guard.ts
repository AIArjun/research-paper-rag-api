import { loadConfig, type ServerConfig } from "./env";
import { apiError, errorResponse } from "./errors";
import { isTrustedOrigin } from "./origin";
import { SESSION_COOKIE, nowSeconds, verifySessionToken } from "./session";

export type Guarded = { ok: true; config: ServerConfig } | { ok: false; response: Response };

function cookieValue(request: Request, name: string): string | undefined {
  const header = request.headers.get("cookie");
  if (!header) return undefined;
  for (const part of header.split(";")) {
    const [rawName, ...rest] = part.split("=");
    if (rawName?.trim() === name) return rest.join("=").trim();
  }
  return undefined;
}

/** Configuration must be complete before anything else happens (fail closed). */
export function requireConfig(): Guarded {
  const result = loadConfig();
  if (!result.ok) return { ok: false, response: errorResponse(503, apiError("not_configured")) };
  return { ok: true, config: result.config };
}

/** A valid, unexpired session cookie is required before any corpus or model operation. */
export function requireSession(request: Request, config: ServerConfig): Response | null {
  const verdict = verifySessionToken(config.sessionSecret, cookieValue(request, SESSION_COOKIE), nowSeconds());
  return verdict.ok ? null : errorResponse(401, apiError("unauthenticated"));
}

/** State-changing requests must also originate from a trusted origin. */
export function requireTrustedOrigin(request: Request, config: ServerConfig): Response | null {
  return isTrustedOrigin(request.headers, config.allowedOrigins)
    ? null
    : errorResponse(403, apiError("forbidden_origin"));
}

/** Session first, then origin: nothing about the corpus is touched for anonymous callers. */
export function guardMutation(request: Request): Guarded {
  const configured = requireConfig();
  if (!configured.ok) return configured;
  const unauthenticated = requireSession(request, configured.config);
  if (unauthenticated) return { ok: false, response: unauthenticated };
  const badOrigin = requireTrustedOrigin(request, configured.config);
  if (badOrigin) return { ok: false, response: badOrigin };
  return configured;
}

export function guardRead(request: Request): Guarded {
  const configured = requireConfig();
  if (!configured.ok) return configured;
  const unauthenticated = requireSession(request, configured.config);
  if (unauthenticated) return { ok: false, response: unauthenticated };
  return configured;
}

export function hasValidSessionCookie(cookieHeaderValue: string | undefined, config: ServerConfig): boolean {
  return verifySessionToken(config.sessionSecret, cookieHeaderValue, nowSeconds()).ok;
}
