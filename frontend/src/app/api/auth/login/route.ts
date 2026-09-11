import { clientKey, loginThrottle, passcodeMatches } from "@/lib/server/auth";
import { parseJsonObject, readBoundedBody } from "@/lib/server/body";
import { serializeCookie } from "@/lib/server/cookies";
import { apiError, errorResponse, NO_STORE_HEADERS } from "@/lib/server/errors";
import { requireConfig, requireTrustedOrigin } from "@/lib/server/guard";
import { createSessionToken, nowSeconds, sessionCookie } from "@/lib/server/session";
import { MAX_LOGIN_BODY_BYTES } from "@/lib/shared/limits";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request): Promise<Response> {
  const configured = requireConfig();
  if (!configured.ok) return configured.response;
  const { config } = configured;

  const badOrigin = requireTrustedOrigin(request, config);
  if (badOrigin) return badOrigin;

  const key = clientKey(request.headers);
  const now = Date.now();
  const throttle = loginThrottle.check(key, now);
  if (!throttle.allowed) {
    return errorResponse(429, apiError("too_many_attempts", { retry_after: throttle.retryAfterSeconds }));
  }

  const body = await readBoundedBody(request, MAX_LOGIN_BODY_BYTES);
  if (!body.ok) {
    return errorResponse(body.reason === "too_large" ? 413 : 400, apiError(body.reason === "too_large" ? "request_too_large" : "invalid_request"));
  }
  const parsed = parseJsonObject(body.bytes);
  const passcode = parsed?.passcode;
  if (typeof passcode !== "string") return errorResponse(400, apiError("invalid_request"));

  if (!passcodeMatches(passcode, config.demoPasscode)) {
    loginThrottle.recordFailure(key, now);
    return errorResponse(401, apiError("invalid_passcode"));
  }

  const token = createSessionToken(config.sessionSecret, nowSeconds());
  return new Response(null, {
    status: 204,
    headers: { ...NO_STORE_HEADERS, "Set-Cookie": serializeCookie(sessionCookie(token, config.isProduction)) },
  });
}
