import { readBoundedBody } from "@/lib/server/body";
import { apiError, errorResponse, jsonResponse } from "@/lib/server/errors";
import { guardMutation } from "@/lib/server/guard";
import { checkUploadForm } from "@/lib/server/upload-check";
import { callUpstream } from "@/lib/server/upstream";
import { parseUploadResponse } from "@/lib/server/validate";
import { MAX_UPLOAD_REQUEST_BYTES } from "@/lib/shared/limits";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
/** Indexing a 15-20 page paper on the Render instance takes seconds; leave headroom, no retry. */
export const maxDuration = 120;
const UPSTREAM_TIMEOUT_MS = 100_000;

export async function POST(request: Request): Promise<Response> {
  // Authentication and origin checks happen before a single body byte is read.
  const guarded = guardMutation(request);
  if (!guarded.ok) return guarded.response;

  const body = await readBoundedBody(request, MAX_UPLOAD_REQUEST_BYTES);
  if (!body.ok) {
    return body.reason === "too_large"
      ? errorResponse(413, apiError("file_too_large"))
      : errorResponse(400, apiError("invalid_request"));
  }
  const checked = await checkUploadForm(body.bytes, request.headers.get("content-type"));
  if (!checked.ok) return errorResponse(checked.status, apiError(checked.category));

  const form = new FormData();
  form.append("file", new File([checked.bytes], checked.name, { type: "application/pdf" }), checked.name);
  const result = await callUpstream(guarded.config, {
    route: "/papers/upload",
    method: "POST",
    body: form,
    timeoutMs: UPSTREAM_TIMEOUT_MS,
  });
  if (!result.ok) return errorResponse(result.status, result.error);

  const uploaded = parseUploadResponse(result.body);
  if (!uploaded) return errorResponse(502, apiError("backend_error"));
  return jsonResponse(uploaded, 201);
}
