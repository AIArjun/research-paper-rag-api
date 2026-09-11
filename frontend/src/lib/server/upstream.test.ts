import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import type { ServerConfig } from "@/lib/server/env";
import { callUpstream, type FetchLike, type UpstreamRequest, type UpstreamRoute } from "@/lib/server/upstream";

const TOKEN = "test-token-0123456789abcdef0123456789abcdef";
const BASE_URL = "https://api.example.test";
const SENTINEL = "SECRET-UPSTREAM-TEXT";

const CONFIG: ServerConfig = {
  ragApiUrl: BASE_URL,
  ragApiToken: TOKEN,
  demoPasscode: "test-passcode-abcdefghijklmnop",
  sessionSecret: "test-secret-0123456789abcdef0123456789abcdef",
  allowedOrigins: ["https://observatory.test"],
  isProduction: false,
};

function req(route: UpstreamRoute, overrides: Partial<UpstreamRequest> = {}): UpstreamRequest {
  const method = route === "/ready" || route === "/papers" ? "GET" : "POST";
  return { route, method, timeoutMs: 5_000, ...overrides };
}

function json(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });
}

function abortError(): Error {
  return Object.assign(new Error("This operation was aborted"), { name: "AbortError" });
}

/** The single recorded call of a fake fetch; fails loudly unless it was called exactly once. */
function onlyCall(fetchImpl: Mock<FetchLike>): { url: string; init: RequestInit } {
  expect(fetchImpl).toHaveBeenCalledTimes(1);
  const call = fetchImpl.mock.calls[0];
  if (!call) throw new Error("fetch was not called");
  return { url: call[0], init: call[1] };
}

function expectNoToken(value: unknown): void {
  expect(JSON.stringify(value)).not.toContain(TOKEN);
}

/** A fetch that never answers on its own and only rejects when its signal is aborted. */
function fetchThatHangsUntilAborted(): FetchLike {
  return (_url, init) =>
    new Promise<Response>((_resolve, reject) => {
      const signal = init.signal;
      if (!signal) throw new Error("fetch was called without an abort signal");
      signal.addEventListener("abort", () => reject(signal.reason as unknown), { once: true });
    });
}

describe("callUpstream", () => {
  describe("request shape", () => {
    it.each(["/papers", "/papers/upload", "/query"] as const)(
      "sends the bearer token to the protected route %s at the configured base URL",
      async (route) => {
        const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(json({}));
        await callUpstream(CONFIG, req(route), fetchImpl);
        const { url, init } = onlyCall(fetchImpl);
        expect(url).toBe(`${BASE_URL}${route}`);
        expect(new Headers(init.headers).get("authorization")).toBe(`Bearer ${TOKEN}`);
      },
    );

    it("sends no credential at all to the public /ready route", async () => {
      const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(json({ ready: true }));
      await callUpstream(CONFIG, req("/ready"), fetchImpl);
      const { url, init } = onlyCall(fetchImpl);
      expect(url).toBe(`${BASE_URL}/ready`);
      expect(new Headers(init.headers).has("authorization")).toBe(false);
      expect(JSON.stringify(init)).not.toContain(TOKEN);
    });

    it("forwards method, body and content type, asks for JSON, and never follows redirects", async () => {
      const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(json({}));
      const body = JSON.stringify({ question: "What is attention?", top_k: 3 });
      await callUpstream(CONFIG, req("/query", { body, contentType: "application/json" }), fetchImpl);
      const { init } = onlyCall(fetchImpl);
      const headers = new Headers(init.headers);
      expect(init.method).toBe("POST");
      expect(init.body).toBe(body);
      expect(headers.get("content-type")).toBe("application/json");
      expect(headers.get("accept")).toBe("application/json");
      expect(init.redirect).toBe("error");
      expect(init.cache).toBe("no-store");
      expect(init.signal).toBeInstanceOf(AbortSignal);
    });

    it("leaves Content-Type unset when none is given, so a multipart body keeps its own boundary", async () => {
      const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(json({}));
      await callUpstream(CONFIG, req("/papers/upload", { body: new FormData() }), fetchImpl);
      const { init } = onlyCall(fetchImpl);
      expect(new Headers(init.headers).has("content-type")).toBe(false);
    });

    it("rejects a route outside the allowlist before touching the network", async () => {
      const fetchImpl = vi.fn<FetchLike>();
      const rogue = req("/admin" as UpstreamRoute);
      await expect(callUpstream(CONFIG, rogue, fetchImpl)).rejects.toThrow("not allowlisted");
      expect(fetchImpl).not.toHaveBeenCalled();
    });
  });

  describe("2xx responses", () => {
    it("returns the parsed JSON body", async () => {
      const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(json([{ paper_id: "p1" }]));
      const result = await callUpstream(CONFIG, req("/papers"), fetchImpl);
      expect(result).toEqual({ ok: true, status: 200, body: [{ paper_id: "p1" }] });
      expect(fetchImpl).toHaveBeenCalledTimes(1);
    });

    it("returns body null for an empty 2xx body", async () => {
      const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(new Response("", { status: 200 }));
      const result = await callUpstream(CONFIG, req("/papers"), fetchImpl);
      expect(result).toEqual({ ok: true, status: 200, body: null });
    });

    it("answers 502 backend_error when a 2xx body is not JSON", async () => {
      const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(new Response("<html>not json</html>", { status: 200 }));
      const result = await callUpstream(CONFIG, req("/query"), fetchImpl);
      expect(result.ok).toBe(false);
      expect(result).toMatchObject({ status: 502, error: { category: "backend_error" } });
      expect(fetchImpl).toHaveBeenCalledTimes(1);
      expectNoToken(result);
    });
  });

  describe("failures are reported after exactly one attempt", () => {
    it("maps a rejection named AbortError to 504 timeout", async () => {
      const fetchImpl = vi.fn<FetchLike>().mockRejectedValue(abortError());
      const result = await callUpstream(CONFIG, req("/query"), fetchImpl);
      expect(result).toMatchObject({ ok: false, status: 504, error: { category: "timeout" } });
      expect(fetchImpl).toHaveBeenCalledTimes(1);
      expectNoToken(result);
    });

    it("maps a TypeError (network failure) to 502 unreachable", async () => {
      const fetchImpl = vi.fn<FetchLike>().mockRejectedValue(new TypeError("fetch failed"));
      const result = await callUpstream(CONFIG, req("/papers"), fetchImpl);
      expect(result).toMatchObject({ ok: false, status: 502, error: { category: "unreachable" } });
      expect(fetchImpl).toHaveBeenCalledTimes(1);
      expectNoToken(result);
    });

    it("maps an unexpected 5xx to 502 backend_error without echoing upstream text", async () => {
      const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(json({ detail: SENTINEL }, 500));
      const result = await callUpstream(CONFIG, req("/query"), fetchImpl);
      expect(result).toMatchObject({ ok: false, status: 502, error: { category: "backend_error" } });
      expect(fetchImpl).toHaveBeenCalledTimes(1);
      expect(JSON.stringify(result)).not.toContain(SENTINEL);
      expectNoToken(result);
    });

    it("maps a 429 budget_exhausted with its Retry-After and request id", async () => {
      const upstream = json(
        { detail: { category: "budget_exhausted", request_id: "abc", message: SENTINEL } },
        429,
        { "retry-after": "30" },
      );
      const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(upstream);
      const result = await callUpstream(CONFIG, req("/query"), fetchImpl);
      expect(result.ok).toBe(false);
      if (result.ok) return;
      expect(result.status).toBe(429);
      expect(result.error.category).toBe("budget_exhausted");
      expect(result.error.retry_after).toBe(30);
      expect(result.error.request_id).toBe("abc");
      expect(fetchImpl).toHaveBeenCalledTimes(1);
      expect(JSON.stringify(result)).not.toContain(SENTINEL);
      expectNoToken(result);
    });

    it("maps a 503 with Retry-After to backend_unavailable and keeps the delay", async () => {
      const upstream = json({ detail: { category: "backend_not_ready", request_id: "r1" } }, 503, { "retry-after": "5" });
      const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(upstream);
      const result = await callUpstream(CONFIG, req("/papers"), fetchImpl);
      expect(result).toMatchObject({
        ok: false,
        status: 503,
        error: { category: "backend_unavailable", retry_after: 5, upstream_category: "backend_not_ready" },
      });
      expect(fetchImpl).toHaveBeenCalledTimes(1);
    });
  });

  describe("the timeout clock", () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });
    afterEach(() => {
      vi.useRealTimers();
    });

    it("aborts a fetch that has not answered within timeoutMs and reports 504 timeout", async () => {
      const fetchImpl = vi.fn<FetchLike>(fetchThatHangsUntilAborted());
      let settled = false;
      const pending = callUpstream(CONFIG, req("/query", { timeoutMs: 1_000 }), fetchImpl).then((result) => {
        settled = true;
        return result;
      });

      await vi.advanceTimersByTimeAsync(999);
      expect(settled).toBe(false);

      await vi.advanceTimersByTimeAsync(1);
      const result = await pending;
      expect(result).toMatchObject({ ok: false, status: 504, error: { category: "timeout" } });
      expect(fetchImpl).toHaveBeenCalledTimes(1);
      expectNoToken(result);
    });

    it("keeps the clock running while the body streams: a body still open at timeoutMs is a 504", async () => {
      const fetchImpl = vi.fn<FetchLike>(async (_url, init) => {
        const signal = init.signal;
        if (!signal) throw new Error("fetch was called without an abort signal");
        const body = new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(new TextEncoder().encode('{"answer": "partial'));
            signal.addEventListener("abort", () => controller.error(signal.reason), { once: true });
          },
        });
        return new Response(body, { status: 200, headers: { "content-type": "application/json" } });
      });

      const pending = callUpstream(CONFIG, req("/query", { timeoutMs: 2_000 }), fetchImpl);
      await vi.advanceTimersByTimeAsync(2_001);
      const result = await pending;
      expect(result).toMatchObject({ ok: false, status: 504, error: { category: "timeout" } });
      expect(fetchImpl).toHaveBeenCalledTimes(1);
      expectNoToken(result);
    });

    it("clears the timer once a response has been fully read, so nothing fires afterwards", async () => {
      const fetchImpl = vi.fn<FetchLike>().mockResolvedValue(json({ ok: 1 }));
      const result = await callUpstream(CONFIG, req("/papers", { timeoutMs: 1_000 }), fetchImpl);
      expect(result).toEqual({ ok: true, status: 200, body: { ok: 1 } });
      expect(vi.getTimerCount()).toBe(0);
      const { init } = onlyCall(fetchImpl);
      await vi.advanceTimersByTimeAsync(5_000);
      expect(init.signal?.aborted).toBe(false);
    });
  });
});
