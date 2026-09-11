import { beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import { POST as login } from "@/app/api/auth/login/route";
import { POST as logout } from "@/app/api/auth/logout/route";
import { GET as listPapers } from "@/app/api/papers/route";
import { POST as uploadPaper } from "@/app/api/papers/upload/route";
import { POST as ask } from "@/app/api/query/route";
import { GET as readStatus } from "@/app/api/status/route";
import { loginThrottle } from "@/lib/server/auth";
import type { ApiError, ApiErrorBody } from "@/lib/shared/types";

const ORIGIN = "https://observatory.test";
const FOREIGN_ORIGIN = "https://attacker.example.net";
const RAG_API_URL = "https://api.example.test";
const TOKEN = "test-token-0123456789abcdef0123456789abcdef";
const PASSCODE = "test-passcode-abcdefghijklmnop";
const WRONG_PASSCODE = "wrong-passcode-abcdefghijklmn";
const SECRET = "test-secret-0123456789abcdef0123456789abcdef";
const DIGEST = "a".repeat(64);

let fetchMock: Mock<typeof fetch>;

beforeEach(() => {
  vi.stubEnv("RAG_API_URL", RAG_API_URL);
  vi.stubEnv("RAG_API_TOKEN", TOKEN);
  vi.stubEnv("DEMO_PASSCODE", PASSCODE);
  vi.stubEnv("SESSION_SECRET", SECRET);
  vi.stubEnv("APP_ORIGIN", ORIGIN);
  vi.stubEnv("NODE_ENV", "test");
  fetchMock = vi.fn<typeof fetch>();
  vi.stubGlobal("fetch", fetchMock);
  // The throttle is a module singleton shared by every test in this process.
  loginThrottle.reset();
});

type HeaderMap = Record<string, string>;

function request(path: string, init: { method?: string; headers?: HeaderMap; body?: BodyInit | null } = {}): Request {
  return new Request(`${ORIGIN}${path}`, init);
}

function jsonRequest(path: string, body: unknown, headers: HeaderMap = {}): Request {
  return request(path, { method: "POST", headers: { "content-type": "application/json", ...headers }, body: JSON.stringify(body) });
}

/** A body whose reads are observable: with highWaterMark 0 the stream never pre-fills, so `pull` fires only on a real read. */
function spiedBody(text: string): { stream: ReadableStream<Uint8Array>; pull: Mock<(c: ReadableStreamDefaultController<Uint8Array>) => void> } {
  const pull = vi.fn((controller: ReadableStreamDefaultController<Uint8Array>) => {
    controller.enqueue(new TextEncoder().encode(text));
    controller.close();
  });
  return { stream: new ReadableStream<Uint8Array>({ pull }, { highWaterMark: 0 }), pull };
}

function streamRequest(path: string, headers: HeaderMap, stream: ReadableStream<Uint8Array>): Request {
  return new Request(`${ORIGIN}${path}`, { method: "POST", headers, body: stream, duplex: "half" } as RequestInit);
}

function upstreamJson(body: unknown, status = 200, headers: HeaderMap = {}): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });
}

async function errorOf(res: Response): Promise<ApiError> {
  const body = (await res.json()) as ApiErrorBody;
  return body.error;
}

/** Logs in through the real route and returns a `cookie` header value for the session. */
async function sessionCookie(): Promise<string> {
  const res = await login(jsonRequest("/api/auth/login", { passcode: PASSCODE }, { origin: ORIGIN }));
  expect(res.status).toBe(204);
  const match = /ro_session=([^;]+)/.exec(res.headers.get("set-cookie") ?? "");
  const value = match?.[1];
  if (!value) throw new Error("login did not set ro_session");
  return `ro_session=${value}`;
}

function onlyFetchCall(): { url: string; init: RequestInit } {
  expect(fetchMock).toHaveBeenCalledTimes(1);
  const call = fetchMock.mock.calls[0];
  if (!call) throw new Error("fetch was not called");
  return { url: String(call[0]), init: call[1] ?? {} };
}

function pdfBytes(): Uint8Array<ArrayBuffer> {
  return new TextEncoder().encode("%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\n%%EOF\n");
}

describe("POST /api/auth/login", () => {
  it("(1) refuses a foreign Origin before looking at the passcode and sets no cookie", async () => {
    const res = await login(jsonRequest("/api/auth/login", { passcode: PASSCODE }, { origin: FOREIGN_ORIGIN }));
    expect(res.status).toBe(403);
    expect((await errorOf(res)).category).toBe("forbidden_origin");
    expect(res.headers.get("set-cookie")).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("(2) answers 401 invalid_passcode for a wrong passcode without a cookie", async () => {
    const res = await login(jsonRequest("/api/auth/login", { passcode: WRONG_PASSCODE }, { origin: ORIGIN }));
    expect(res.status).toBe(401);
    expect((await errorOf(res)).category).toBe("invalid_passcode");
    expect(res.headers.get("set-cookie")).toBeNull();
    expect(res.headers.get("cache-control")).toContain("no-store");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("(3) sets a signed HttpOnly session cookie for the right passcode", async () => {
    const res = await login(jsonRequest("/api/auth/login", { passcode: PASSCODE }, { origin: ORIGIN }));
    expect(res.status).toBe(204);
    expect(res.headers.get("cache-control")).toContain("no-store");
    const cookie = res.headers.get("set-cookie") ?? "";
    expect(cookie).toMatch(/^ro_session=[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+;/);
    expect(cookie).toContain("HttpOnly");
    expect(cookie).toContain("SameSite=Lax");
    expect(cookie).toContain("Path=/");
    expect(cookie).toContain("Max-Age=43200");
    expect(cookie).not.toContain(TOKEN);
    expect(cookie).not.toContain(SECRET);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("(12) throttles the 11th attempt from one address, even with the right passcode", async () => {
    const headers = { origin: ORIGIN, "x-forwarded-for": "203.0.113.9" };
    for (let attempt = 0; attempt < 10; attempt += 1) {
      const res = await login(jsonRequest("/api/auth/login", { passcode: WRONG_PASSCODE }, headers));
      expect(res.status, `attempt ${attempt + 1}`).toBe(401);
    }
    const blocked = await login(jsonRequest("/api/auth/login", { passcode: PASSCODE }, headers));
    expect(blocked.status).toBe(429);
    expect((await errorOf(blocked)).category).toBe("too_many_attempts");
    expect(Number(blocked.headers.get("retry-after"))).toBeGreaterThan(0);
    expect(blocked.headers.get("set-cookie")).toBeNull();

    // Another address is unaffected by that block.
    const other = { origin: ORIGIN, "x-forwarded-for": "198.51.100.1" };
    expect((await login(jsonRequest("/api/auth/login", { passcode: PASSCODE }, other))).status).toBe(204);
  });

  it("fails closed with 503 not_configured when the server environment is incomplete", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    vi.stubEnv("RAG_API_TOKEN", "");
    const res = await login(jsonRequest("/api/auth/login", { passcode: PASSCODE }, { origin: ORIGIN }));
    expect(res.status).toBe(503);
    expect((await errorOf(res)).category).toBe("not_configured");
    expect(res.headers.get("set-cookie")).toBeNull();
    expect((await listPapers(request("/api/papers"))).status).toBe(503);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("the session gate", () => {
  it("(4) GET /api/papers without a cookie is 401 and never reaches the backend", async () => {
    const res = await listPapers(request("/api/papers"));
    expect(res.status).toBe(401);
    expect((await errorOf(res)).category).toBe("unauthenticated");
    expect(res.headers.get("cache-control")).toContain("no-store");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("(5) POST /api/query without a cookie is 401 before a single body byte is read", async () => {
    const { stream, pull } = spiedBody(JSON.stringify({ question: "What is attention?", top_k: 3 }));
    const req = streamRequest("/api/query", { origin: ORIGIN, "content-type": "application/json" }, stream);
    const res = await ask(req);
    expect(res.status).toBe(401);
    expect((await errorOf(res)).category).toBe("unauthenticated");
    expect(fetchMock).not.toHaveBeenCalled();
    expect(pull).not.toHaveBeenCalled();
    expect(req.bodyUsed).toBe(false);
  });

  it("(6) POST /api/papers/upload without a cookie is 401 before a single body byte is read", async () => {
    const { stream, pull } = spiedBody("--boundary\r\nContent-Disposition: form-data; name=\"file\"\r\n\r\n%PDF-\r\n--boundary--\r\n");
    const headers = { origin: ORIGIN, "content-type": "multipart/form-data; boundary=boundary" };
    const req = streamRequest("/api/papers/upload", headers, stream);
    const res = await uploadPaper(req);
    expect(res.status).toBe(401);
    expect((await errorOf(res)).category).toBe("unauthenticated");
    expect(fetchMock).not.toHaveBeenCalled();
    expect(pull).not.toHaveBeenCalled();
    expect(req.bodyUsed).toBe(false);
  });

  it("(7) POST /api/query with a session but a foreign Origin is 403 and never reaches the backend", async () => {
    const cookie = await sessionCookie();
    const { stream, pull } = spiedBody(JSON.stringify({ question: "What is attention?", top_k: 3 }));
    const req = streamRequest("/api/query", { cookie, origin: FOREIGN_ORIGIN, "content-type": "application/json" }, stream);
    const res = await ask(req);
    expect(res.status).toBe(403);
    expect((await errorOf(res)).category).toBe("forbidden_origin");
    expect(fetchMock).not.toHaveBeenCalled();
    expect(pull).not.toHaveBeenCalled();
    expect(req.bodyUsed).toBe(false);
  });

  it("rejects a tampered cookie like a missing one", async () => {
    const cookie = await sessionCookie();
    const flipped = cookie.endsWith("A") ? "B" : "A";
    const tampered = `${cookie.slice(0, -1)}${flipped}`;
    const res = await listPapers(request("/api/papers", { headers: { cookie: tampered } }));
    expect(res.status).toBe(401);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("POST /api/query", () => {
  it("(8) rejects an out-of-range top_k with 422 before any backend call", async () => {
    const cookie = await sessionCookie();
    const res = await ask(jsonRequest("/api/query", { question: "What is attention?", top_k: 9 }, { cookie, origin: ORIGIN }));
    expect(res.status).toBe(422);
    expect((await errorOf(res)).category).toBe("invalid_request");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("(9) forwards a valid question exactly once with the bearer token and returns the sanitized answer", async () => {
    const cookie = await sessionCookie();
    fetchMock.mockResolvedValue(
      upstreamJson({
        request_id: "req-9",
        question: "What is attention?",
        answer: "Attention weighs every token against every other token.",
        citations: [{ text: "Attention is all you need.", page: 3, paper: "attention.pdf", relevance_score: 0.87, paper_id: DIGEST, chunk_id: `${DIGEST}_0003`, embedding: [0.1, 0.2] }],
        papers_searched: 1,
        retrieval_time_ms: 12,
        generation_time_ms: 800,
        total_time_ms: 812,
        model_used: "gpt-4o-mini",
        model_usage: null,
        internal_debug: "trace-xyz",
      }),
    );

    const res = await ask(jsonRequest("/api/query", { question: "What is attention?", top_k: 3 }, { cookie, origin: ORIGIN }));
    expect(res.status).toBe(200);
    expect(res.headers.get("cache-control")).toContain("no-store");

    const { url, init } = onlyFetchCall();
    expect(url).toBe(`${RAG_API_URL}/query`);
    expect(init.method).toBe("POST");
    const headers = new Headers(init.headers);
    expect(headers.get("authorization")).toBe(`Bearer ${TOKEN}`);
    expect(headers.get("content-type")).toBe("application/json");
    expect(JSON.parse(String(init.body))).toEqual({ question: "What is attention?", top_k: 3 });

    const body = (await res.json()) as Record<string, unknown>;
    expect(body.answer).toBe("Attention weighs every token against every other token.");
    expect(body.request_id).toBe("req-9");
    expect(body).not.toHaveProperty("internal_debug");
    expect((body.citations as Record<string, unknown>[])[0]).not.toHaveProperty("embedding");
    expect(JSON.stringify(body)).not.toContain(TOKEN);
  });

  it("(10) reports 502 unreachable when the backend cannot be reached, after exactly one attempt", async () => {
    const cookie = await sessionCookie();
    fetchMock.mockRejectedValue(new TypeError("fetch failed"));
    const res = await ask(jsonRequest("/api/query", { question: "What is attention?" }, { cookie, origin: ORIGIN }));
    expect(res.status).toBe(502);
    expect((await errorOf(res)).category).toBe("unreachable");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("maps a backend 429 budget_exhausted onto the browser response with Retry-After and no upstream text", async () => {
    const cookie = await sessionCookie();
    fetchMock.mockResolvedValue(
      upstreamJson({ detail: { message: "SECRET-UPSTREAM-TEXT", category: "budget_exhausted", request_id: "req-429" } }, 429, { "retry-after": "60" }),
    );
    const res = await ask(jsonRequest("/api/query", { question: "What is attention?" }, { cookie, origin: ORIGIN }));
    expect(res.status).toBe(429);
    expect(res.headers.get("retry-after")).toBe("60");
    const error = await errorOf(res);
    expect(error).toMatchObject({ category: "budget_exhausted", retry_after: 60, request_id: "req-429" });
    expect(JSON.stringify(error)).not.toContain("SECRET-UPSTREAM-TEXT");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("POST /api/papers/upload", () => {
  it("forwards a bounded PDF once as multipart with the bearer token and answers 201", async () => {
    const cookie = await sessionCookie();
    fetchMock.mockResolvedValue(
      upstreamJson({ paper_id: DIGEST, filename: "paper.pdf", pages: 1, chunks: 2, processing_time_ms: 55.5, message: "indexed", extra: "leak" }),
    );
    const form = new FormData();
    form.append("file", new File([pdfBytes()], "paper.pdf", { type: "application/pdf" }));
    const res = await uploadPaper(request("/api/papers/upload", { method: "POST", headers: { cookie, origin: ORIGIN }, body: form }));
    expect(res.status).toBe(201);

    const { url, init } = onlyFetchCall();
    expect(url).toBe(`${RAG_API_URL}/papers/upload`);
    expect(new Headers(init.headers).get("authorization")).toBe(`Bearer ${TOKEN}`);
    expect(init.body).toBeInstanceOf(FormData);
    const forwarded = (init.body as FormData).get("file");
    expect(forwarded).toBeInstanceOf(File);
    expect((forwarded as File).name).toBe("paper.pdf");
    expect((forwarded as File).size).toBe(pdfBytes().byteLength);

    const body = (await res.json()) as Record<string, unknown>;
    expect(body).toMatchObject({ paper_id: DIGEST, filename: "paper.pdf", pages: 1, chunks: 2 });
    expect(body).not.toHaveProperty("extra");
  });

  it("rejects a file that does not start like a PDF before any backend call", async () => {
    const cookie = await sessionCookie();
    const form = new FormData();
    form.append("file", new File([new TextEncoder().encode("hello, not a pdf")], "paper.pdf", { type: "application/pdf" }));
    const res = await uploadPaper(request("/api/papers/upload", { method: "POST", headers: { cookie, origin: ORIGIN }, body: form }));
    expect(res.status).toBe(400);
    expect((await errorOf(res)).category).toBe("invalid_pdf");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("GET /api/status", () => {
  it("(11) calls /ready without a credential and projects the readiness payload", async () => {
    const cookie = await sessionCookie();
    fetchMock.mockResolvedValue(
      upstreamJson({
        ready: true,
        configured_provider: "openai",
        configured_model: "gpt-4o-mini",
        effective_retrieval: "chroma",
        effective_generation: "openai",
        init_error: null,
        pending_cleanup_ids: [],
        provider_connection_verified: true,
        access_configured: true,
        model_budget: {
          state: "ok",
          configured: true,
          usage: { calls_today: 3, calls_total: 7, tokens_charged_today: 900, tokens_charged_total: 2100, calls_unsettled: 0, daily_call_allowance: 20, total_call_allowance: 20, daily_token_allowance: 100_000, total_token_allowance: 100_000, ledger_created_at: "2026-09-01T00:00:00Z" },
          token_bound: "tiktoken/o200k_base",
        },
        limits: { max_pdf_bytes: 10_485_760, max_pages: 40 },
      }),
    );

    const res = await readStatus(request("/api/status", { headers: { cookie } }));
    expect(res.status).toBe(200);

    const { url, init } = onlyFetchCall();
    expect(url).toBe(`${RAG_API_URL}/ready`);
    expect(new Headers(init.headers).has("authorization")).toBe(false);
    expect(JSON.stringify(init)).not.toContain(TOKEN);

    const body = (await res.json()) as Record<string, unknown>;
    expect(body).toMatchObject({ ready: true, configured_model: "gpt-4o-mini", budget_state: "ok", usage: { calls_today: 3, daily_call_allowance: 20 } });
    const serialized = JSON.stringify(body);
    for (const hidden of ["limits", "ledger_created_at", "token_bound", "access_configured"]) {
      expect(serialized, hidden).not.toContain(hidden);
    }
  });
});

describe("POST /api/auth/logout", () => {
  it("(13) clears the session cookie", async () => {
    const res = await logout(request("/api/auth/logout", { method: "POST", headers: { origin: ORIGIN } }));
    expect(res.status).toBe(204);
    expect(res.headers.get("cache-control")).toContain("no-store");
    const cookie = res.headers.get("set-cookie") ?? "";
    expect(cookie).toMatch(/^ro_session=;/);
    expect(cookie).toContain("Max-Age=0");
    expect(cookie).toContain("HttpOnly");
    expect(cookie).toContain("Path=/");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("refuses a foreign Origin", async () => {
    const res = await logout(request("/api/auth/logout", { method: "POST", headers: { origin: FOREIGN_ORIGIN } }));
    expect(res.status).toBe(403);
    expect(res.headers.get("set-cookie")).toBeNull();
  });
});
