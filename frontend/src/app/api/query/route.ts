import { parseJsonObject, readBoundedBody } from "@/lib/server/body";
import { apiError, errorResponse, jsonResponse } from "@/lib/server/errors";
import { guardMutation } from "@/lib/server/guard";
import { validateQuery } from "@/lib/server/query-validation";
import { callUpstream } from "@/lib/server/upstream";
import { parseQueryResponse } from "@/lib/server/validate";
import { MAX_QUERY_BODY_BYTES } from "@/lib/shared/limits";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
/** Render's provider timeout is 30 s; retrieval and a cold instance add to it. */
export const maxDuration = 90;
const UPSTREAM_TIMEOUT_MS = 75_000;

export async function POST(request: Request): Promise<Response> {
  const guarded = guardMutation(request);
  if (!guarded.ok) return guarded.response;

  const body = await readBoundedBody(request, MAX_QUERY_BODY_BYTES);
  if (!body.ok) {
    return body.reason === "too_large"
      ? errorResponse(413, apiError("request_too_large"))
      : errorResponse(400, apiError("invalid_request"));
  }
  const parsed = parseJsonObject(body.bytes);
  const query = parsed ? validateQuery(parsed) : null;
  if (!query) return errorResponse(422, apiError("invalid_request"));

  const result = await callUpstream(guarded.config, {
    route: "/query",
    method: "POST",
    body: JSON.stringify(query),
    contentType: "application/json",
    timeoutMs: UPSTREAM_TIMEOUT_MS,
  });
  if (!result.ok) return errorResponse(result.status, result.error);

  const answer = parseQueryResponse(result.body);
  if (!answer) return errorResponse(502, apiError("backend_error"));
  return jsonResponse(answer);
}
