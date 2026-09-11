import type { ApiError, ApiErrorBody } from "@/lib/shared/types";

export class RequestFailure extends Error {
  constructor(public readonly status: number, public readonly error: ApiError) {
    super(error.message);
    this.name = "RequestFailure";
  }
}

function isErrorBody(value: unknown): value is ApiErrorBody {
  if (typeof value !== "object" || value === null) return false;
  const error = (value as { error?: unknown }).error;
  return typeof error === "object" && error !== null && typeof (error as { category?: unknown }).category === "string";
}

const NETWORK_ERROR: ApiError = {
  category: "unreachable",
  message: "The observatory could not be reached. Check the connection and try again; nothing is retried automatically.",
};

/**
 * Same-origin JSON call to the BFF. Never retries. Throws RequestFailure with a
 * categorized error for any non-2xx answer.
 */
export async function callApi<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, { credentials: "same-origin", cache: "no-store", ...init });
  } catch {
    throw new RequestFailure(0, NETWORK_ERROR);
  }
  if (response.status === 204) return undefined as T;
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    const error: ApiError = isErrorBody(body)
      ? body.error
      : { category: "backend_error", message: "The observatory answered in an unexpected way." };
    throw new RequestFailure(response.status, error);
  }
  return body as T;
}

export function postJson<T>(path: string, payload: unknown): Promise<T> {
  return callApi<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function toApiError(error: unknown): ApiError {
  if (error instanceof RequestFailure) return error.error;
  return NETWORK_ERROR;
}
