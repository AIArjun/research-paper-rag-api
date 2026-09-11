import type { ServerConfig } from "./env";
import { apiError, mapUpstreamError } from "./errors";
import type { ApiError } from "@/lib/shared/types";

/** The only backend routes this BFF will ever call. */
export type UpstreamRoute = "/ready" | "/papers" | "/papers/upload" | "/query";
const ALLOWED_ROUTES: ReadonlySet<UpstreamRoute> = new Set(["/ready", "/papers", "/papers/upload", "/query"]);

/** Routes that require the bearer token; /ready is public and gets no credential. */
const PROTECTED_ROUTES: ReadonlySet<UpstreamRoute> = new Set(["/papers", "/papers/upload", "/query"]);

export interface UpstreamRequest {
  route: UpstreamRoute;
  method: "GET" | "POST";
  body?: BodyInit;
  contentType?: string;
  timeoutMs: number;
}

export type UpstreamResult =
  | { ok: true; status: number; body: unknown }
  | { ok: false; status: number; error: ApiError };

export type FetchLike = (input: string, init: RequestInit) => Promise<Response>;

/**
 * One bounded call to the Render backend. No retries: a timeout or network
 * failure is reported as such and the caller decides what to show.
 */
export async function callUpstream(
  config: ServerConfig,
  request: UpstreamRequest,
  fetchImpl: FetchLike = fetch,
): Promise<UpstreamResult> {
  if (!ALLOWED_ROUTES.has(request.route)) {
    throw new Error("upstream route is not allowlisted");
  }
  const headers: Record<string, string> = { Accept: "application/json" };
  if (PROTECTED_ROUTES.has(request.route)) headers.Authorization = `Bearer ${config.ragApiToken}`;
  if (request.contentType) headers["Content-Type"] = request.contentType;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), request.timeoutMs);
  let response: Response;
  try {
    response = await fetchImpl(`${config.ragApiUrl}${request.route}`, {
      method: request.method,
      headers,
      body: request.body,
      signal: controller.signal,
      redirect: "error",
      cache: "no-store",
    });
  } catch (error) {
    clearTimeout(timer);
    const aborted = error instanceof Error && error.name === "AbortError";
    return aborted
      ? { ok: false, status: 504, error: apiError("timeout") }
      : { ok: false, status: 502, error: apiError("unreachable") };
  }

  let body: unknown = null;
  try {
    const text = await response.text();
    body = text.length > 0 ? JSON.parse(text) : null;
  } catch {
    clearTimeout(timer);
    if (response.ok) return { ok: false, status: 502, error: apiError("backend_error") };
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    const mapped = mapUpstreamError(response.status, body, response.headers);
    return { ok: false, status: mapped.status, error: mapped.error };
  }
  return { ok: true, status: response.status, body };
}
