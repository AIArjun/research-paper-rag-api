import type { ApiError, ApiErrorBody, ErrorCategory } from "@/lib/shared/types";
import { MAX_PDF_LABEL } from "@/lib/shared/limits";

export const NO_STORE_HEADERS: Record<string, string> = {
  "Cache-Control": "no-store, max-age=0",
  Pragma: "no-cache",
};

export function jsonResponse(body: unknown, status = 200, extraHeaders: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", ...NO_STORE_HEADERS, ...extraHeaders },
  });
}

export function errorResponse(status: number, error: ApiError, extraHeaders: Record<string, string> = {}): Response {
  const body: ApiErrorBody = { error };
  const headers = { ...extraHeaders };
  if (error.retry_after && !headers["Retry-After"]) headers["Retry-After"] = String(error.retry_after);
  return jsonResponse(body, status, headers);
}

export const MESSAGES: Record<ErrorCategory, string> = {
  not_configured: "The observatory is not configured yet. The operator must set the server environment.",
  unauthenticated: "Your demo session has ended. Enter the passcode again to continue.",
  invalid_passcode: "That passcode was not accepted.",
  too_many_attempts: "Too many attempts from this address. Wait a little before trying again.",
  forbidden_origin: "This request did not come from the observatory itself and was refused.",
  invalid_request: "The request was not valid. Check the question length and paper selection.",
  request_too_large: "The request body is larger than this demo accepts.",
  invalid_pdf: "That file could not be read as a text PDF. Scanned or image-only PDFs are not supported.",
  file_too_large: `The PDF is larger than ${MAX_PDF_LABEL}, the ceiling for this demo.`,
  pdf_limit: "The PDF exceeds the demo's page or chunk ceiling.",
  corpus_full: "The shared corpus is full. A paper must be removed before another can be added.",
  empty_corpus: "The corpus is empty. Add a paper before asking a question.",
  busy: "The backend is busy with another request. Try again in a moment; nothing is retried automatically.",
  budget_exhausted: "The shared model-call allowance for this demo is used up. No model call was made.",
  backend_unavailable: "The research backend is not ready right now.",
  backend_credential: "The backend refused the observatory's credential. The server configuration needs attention.",
  storage: "The backend's paper storage needs cleanup before it can continue.",
  generation_failed: "The model could not produce an answer this time. The attempt still counted toward the allowance.",
  not_found: "That paper is no longer in the corpus.",
  timeout: "The backend took too long to answer. Nothing is retried automatically.",
  unreachable: "The research backend could not be reached.",
  backend_error: "The research backend answered in an unexpected way.",
};

export function apiError(category: ErrorCategory, extra: Partial<Omit<ApiError, "category" | "message">> = {}): ApiError {
  return { category, message: MESSAGES[category], ...extra };
}

/** Shape of the backend's `detail` object (documented in app/main.py). */
interface UpstreamDetail {
  category?: unknown;
  request_id?: unknown;
}

const SAFE_ID = /^[A-Za-z0-9_-]{1,64}$/;

function safeId(value: unknown): string | undefined {
  return typeof value === "string" && SAFE_ID.test(value) ? value : undefined;
}

function upstreamCategory(body: unknown): { category?: string; request_id?: string } {
  if (typeof body !== "object" || body === null) return {};
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail !== "object" || detail === null) return {};
  const d = detail as UpstreamDetail;
  return { category: safeId(d.category), request_id: safeId(d.request_id) };
}

function retryAfter(headers: Headers | undefined): number | undefined {
  const raw = headers?.get("retry-after");
  if (!raw) return undefined;
  const seconds = Number(raw);
  return Number.isFinite(seconds) && seconds > 0 ? Math.min(seconds, 3600) : undefined;
}

/**
 * Map an upstream (non-2xx) response onto a sanitized category. Upstream
 * message text is never forwarded; only its category and request id, which
 * the backend documents as safe identifiers.
 */
export function mapUpstreamError(status: number, body: unknown, headers?: Headers): { status: number; error: ApiError } {
  const { category, request_id } = upstreamCategory(body);
  const extra = { request_id, upstream_category: category };
  const withRetry = { ...extra, retry_after: retryAfter(headers) };

  if (status === 401) return { status: 502, error: apiError("backend_credential", extra) };
  if (status === 404) return { status: 404, error: apiError("not_found", extra) };
  if (status === 429) {
    if (category === "budget_exhausted") return { status: 429, error: apiError("budget_exhausted", withRetry) };
    return { status: 429, error: apiError("busy", withRetry) };
  }
  if (status === 413) {
    if (category === "file_too_large" || category === "request_too_large") {
      return { status: 413, error: apiError("file_too_large", extra) };
    }
    return { status: 413, error: apiError("pdf_limit", extra) };
  }
  if (status === 409) return { status: 409, error: apiError("corpus_full", extra) };
  if (status === 422) return { status: 422, error: apiError("invalid_request", extra) };
  if (status === 400) {
    if (category === "empty_corpus") return { status: 409, error: apiError("empty_corpus", extra) };
    if (category === "invalid_request") return { status: 422, error: apiError("invalid_request", extra) };
    return { status: 400, error: apiError("invalid_pdf", extra) };
  }
  if (status === 502) return { status: 502, error: apiError("generation_failed", extra) };
  if (status === 503) {
    if (category === "access_not_configured") return { status: 502, error: apiError("backend_credential", extra) };
    if (category === "storage_cleanup_required" || category === "storage_failure") {
      return { status: 503, error: apiError("storage", extra) };
    }
    return { status: 503, error: apiError("backend_unavailable", withRetry) };
  }
  return { status: 502, error: apiError("backend_error", extra) };
}
