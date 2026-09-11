import { apiError, errorResponse, jsonResponse } from "@/lib/server/errors";
import { guardRead } from "@/lib/server/guard";
import { callUpstream } from "@/lib/server/upstream";
import { parseReadiness } from "@/lib/server/validate";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 30;

/** GET /ready is public and cheap upstream (no model call); it is still session-gated here. */
export async function GET(request: Request): Promise<Response> {
  const guarded = guardRead(request);
  if (!guarded.ok) return guarded.response;

  const result = await callUpstream(guarded.config, { route: "/ready", method: "GET", timeoutMs: 15_000 });
  // /ready answers 503 with the same body shape while unready; surface that as status, not failure.
  if (!result.ok && result.status !== 503 && result.error.category !== "backend_unavailable") {
    return errorResponse(result.status, result.error);
  }
  if (!result.ok) {
    return jsonResponse({ ready: false, configured_model: "", effective_generation: "unavailable", effective_retrieval: "unavailable", init_error: result.error.upstream_category ?? null, budget_state: null, usage: null });
  }
  const status = parseReadiness(result.body, true);
  if (!status) return errorResponse(502, apiError("backend_error"));
  return jsonResponse(status);
}
