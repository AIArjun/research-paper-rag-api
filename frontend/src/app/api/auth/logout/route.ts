import { serializeCookie } from "@/lib/server/cookies";
import { NO_STORE_HEADERS } from "@/lib/server/errors";
import { requireConfig, requireTrustedOrigin } from "@/lib/server/guard";
import { clearedSessionCookie } from "@/lib/server/session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request): Promise<Response> {
  const configured = requireConfig();
  // Clearing a cookie is harmless even when configuration is incomplete.
  const secure = configured.ok ? configured.config.isProduction : process.env.NODE_ENV === "production";
  if (configured.ok) {
    const badOrigin = requireTrustedOrigin(request, configured.config);
    if (badOrigin) return badOrigin;
  }
  return new Response(null, {
    status: 204,
    headers: { ...NO_STORE_HEADERS, "Set-Cookie": serializeCookie(clearedSessionCookie(secure)) },
  });
}
