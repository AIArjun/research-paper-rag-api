import { describe, expect, it } from "vitest";
import { MESSAGES, NO_STORE_HEADERS, apiError, errorResponse, jsonResponse, mapUpstreamError } from "@/lib/server/errors";
import type { ApiErrorBody, ErrorCategory } from "@/lib/shared/types";

/** Upstream message text that must never reach the browser. */
const SENTINEL = "SECRET-UPSTREAM-TEXT";
const REQUEST_ID = "req-abc123";

/** The backend's documented error body: `detail` carries message, category and request_id. */
function upstreamBody(category?: string, request_id: string = REQUEST_ID): unknown {
  return { detail: { message: SENTINEL, category, request_id } };
}

describe("mapUpstreamError", () => {
  it.each<[number, string | undefined, number, ErrorCategory]>([
    [401, "invalid_token", 502, "backend_credential"],
    [404, "paper_not_found", 404, "not_found"],
    [429, undefined, 429, "busy"],
    [429, "busy", 429, "busy"],
    [429, "budget_exhausted", 429, "budget_exhausted"],
    [413, "file_too_large", 413, "file_too_large"],
    [413, "request_too_large", 413, "file_too_large"],
    [413, "too_many_pages", 413, "pdf_limit"],
    [409, "corpus_full", 409, "corpus_full"],
    [422, undefined, 422, "invalid_request"],
    [400, "empty_corpus", 409, "empty_corpus"],
    [400, "invalid_request", 422, "invalid_request"],
    [400, "invalid_pdf", 400, "invalid_pdf"],
    [502, "generation_failed", 502, "generation_failed"],
    [503, "access_not_configured", 502, "backend_credential"],
    [503, "storage_failure", 503, "storage"],
    [503, "storage_cleanup_required", 503, "storage"],
    [503, "backend_not_ready", 503, "backend_unavailable"],
    [500, undefined, 502, "backend_error"],
    [418, "teapot", 502, "backend_error"],
  ])("upstream %i (%s) becomes %i %s with the observatory's own message", (upstreamStatus, category, status, expected) => {
    const mapped = mapUpstreamError(upstreamStatus, upstreamBody(category));
    expect(mapped.status).toBe(status);
    expect(mapped.error.category).toBe(expected);
    expect(mapped.error.message).toBe(MESSAGES[expected]);
    expect(mapped.error.request_id).toBe(REQUEST_ID);
    expect(mapped.error.upstream_category).toBe(category);
    expect(JSON.stringify(mapped)).not.toContain(SENTINEL);
  });

  describe("sanitizing upstream identifiers", () => {
    it("drops a request_id or category containing unsafe characters", () => {
      const body = { detail: { category: "<script>alert(1)</script>", request_id: "abc def", message: SENTINEL } };
      const { error } = mapUpstreamError(500, body);
      expect(error.request_id).toBeUndefined();
      expect(error.upstream_category).toBeUndefined();
      const serialized = JSON.stringify(error);
      expect(serialized).not.toContain("<script>");
      expect(serialized).not.toContain("abc def");
      expect(serialized).not.toContain(SENTINEL);
    });

    it("keeps identifiers up to 64 safe characters and drops longer ones", () => {
      const ok = mapUpstreamError(404, upstreamBody("not_found", "a".repeat(64)));
      expect(ok.error.request_id).toBe("a".repeat(64));
      const tooLong = mapUpstreamError(404, upstreamBody("not_found", "a".repeat(65)));
      expect(tooLong.error.request_id).toBeUndefined();
    });

    it("never copies a plain-string detail (FastAPI's default error shape)", () => {
      const { error } = mapUpstreamError(404, { detail: SENTINEL });
      expect(error.category).toBe("not_found");
      expect(error.request_id).toBeUndefined();
      expect(JSON.stringify(error)).not.toContain(SENTINEL);
    });

    it.each([null, undefined, "text", 42, ["list"], { detail: null }, { detail: ["x"] }])(
      "tolerates a body that is not the documented object (%j)",
      (body) => {
        const mapped = mapUpstreamError(500, body);
        expect(mapped).toMatchObject({ status: 502, error: { category: "backend_error" } });
        expect(mapped.error.request_id).toBeUndefined();
      },
    );
  });

  describe("Retry-After", () => {
    it("is carried for 429 and for a generic 503, bounded to an hour", () => {
      expect(mapUpstreamError(429, upstreamBody("busy"), new Headers({ "retry-after": "30" })).error.retry_after).toBe(30);
      expect(mapUpstreamError(503, upstreamBody("x"), new Headers({ "retry-after": "120" })).error.retry_after).toBe(120);
      expect(mapUpstreamError(429, upstreamBody("busy"), new Headers({ "retry-after": "99999" })).error.retry_after).toBe(3600);
    });

    it("is omitted when the header is absent, zero, negative or an HTTP date", () => {
      expect(mapUpstreamError(429, upstreamBody("busy")).error.retry_after).toBeUndefined();
      expect(mapUpstreamError(429, upstreamBody("busy"), new Headers()).error.retry_after).toBeUndefined();
      expect(mapUpstreamError(429, upstreamBody("busy"), new Headers({ "retry-after": "0" })).error.retry_after).toBeUndefined();
      expect(mapUpstreamError(429, upstreamBody("busy"), new Headers({ "retry-after": "-5" })).error.retry_after).toBeUndefined();
      const date = new Headers({ "retry-after": "Wed, 21 Oct 2015 07:28:00 GMT" });
      expect(mapUpstreamError(429, upstreamBody("busy"), date).error.retry_after).toBeUndefined();
    });

    it("is not attached to categories that are not retriable, even when upstream sends it", () => {
      const headers = new Headers({ "retry-after": "30" });
      expect(mapUpstreamError(401, upstreamBody("invalid_token"), headers).error.retry_after).toBeUndefined();
      expect(mapUpstreamError(503, upstreamBody("storage_failure"), headers).error.retry_after).toBeUndefined();
    });
  });
});

describe("apiError", () => {
  it("builds exactly category and the fixed message", () => {
    expect(apiError("timeout")).toEqual({ category: "timeout", message: MESSAGES.timeout });
  });

  it("merges the optional identifiers", () => {
    expect(apiError("busy", { retry_after: 5, request_id: "r1", upstream_category: "busy" })).toEqual({
      category: "busy",
      message: MESSAGES.busy,
      retry_after: 5,
      request_id: "r1",
      upstream_category: "busy",
    });
  });

  it("has a non-empty message for every category", () => {
    for (const [category, message] of Object.entries(MESSAGES)) {
      expect(message.length, category).toBeGreaterThan(0);
      expect(apiError(category as ErrorCategory).message).toBe(message);
    }
  });
});

describe("jsonResponse", () => {
  it("serializes the body as JSON with no-store caching by default", async () => {
    const res = jsonResponse({ papers: [] });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toBe("application/json; charset=utf-8");
    expect(res.headers.get("cache-control")).toBe("no-store, max-age=0");
    expect(res.headers.get("cache-control")).toBe(NO_STORE_HEADERS["Cache-Control"]);
    expect(res.headers.get("pragma")).toBe("no-cache");
    expect(await res.json()).toEqual({ papers: [] });
  });

  it("honours a custom status and extra headers", () => {
    const res = jsonResponse({ ok: true }, 201, { "X-Demo": "1" });
    expect(res.status).toBe(201);
    expect(res.headers.get("x-demo")).toBe("1");
    expect(res.headers.get("cache-control")).toContain("no-store");
  });
});

describe("errorResponse", () => {
  it("wraps the error in { error } with the status and no-store caching", async () => {
    const res = errorResponse(401, apiError("unauthenticated"));
    expect(res.status).toBe(401);
    expect(res.headers.get("cache-control")).toContain("no-store");
    expect(res.headers.get("content-type")).toContain("application/json");
    expect(res.headers.get("retry-after")).toBeNull();
    const body = (await res.json()) as ApiErrorBody;
    expect(body).toEqual({ error: { category: "unauthenticated", message: MESSAGES.unauthenticated } });
  });

  it("sets Retry-After from retry_after when present", () => {
    const res = errorResponse(429, apiError("busy", { retry_after: 30 }));
    expect(res.status).toBe(429);
    expect(res.headers.get("retry-after")).toBe("30");
  });

  it("lets an explicit Retry-After header win over the error's retry_after", () => {
    const res = errorResponse(429, apiError("busy", { retry_after: 30 }), { "Retry-After": "7" });
    expect(res.headers.get("retry-after")).toBe("7");
  });
});
