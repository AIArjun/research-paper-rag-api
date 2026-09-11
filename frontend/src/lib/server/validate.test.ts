import { describe, expect, it } from "vitest";
import { parsePaperList, parseQueryResponse, parseReadiness, parseUploadResponse } from "@/lib/server/validate";

const DIGEST = "a".repeat(64);

const PAPER = {
  paper_id: DIGEST,
  filename: "attention.pdf",
  pages: 15,
  chunks: 42,
  uploaded_at: "2026-09-01T10:00:00Z",
  status: "ready",
};

const CITATION = {
  text: "Attention is all you need.",
  page: 3,
  paper: "attention.pdf",
  relevance_score: 0.87,
  paper_id: DIGEST,
  chunk_id: `${DIGEST}_0003`,
};

const USAGE = {
  accounting: "measured",
  input_tokens: 1200,
  output_tokens: 300,
  tokens_charged: 1500,
  tokens_reserved: 0,
  context_chars: 4800,
  reservation_bound: null,
};

const QUERY = {
  request_id: "req-1",
  question: "What is attention?",
  answer: "Attention weighs every token against every other token.",
  citations: [CITATION],
  papers_searched: 2,
  retrieval_time_ms: 12.5,
  generation_time_ms: 800,
  total_time_ms: 812.5,
  model_used: "gpt-4o-mini",
  model_usage: USAGE,
};

const UPLOAD = {
  paper_id: DIGEST,
  filename: "attention.pdf",
  pages: 15,
  chunks: 42,
  processing_time_ms: 3210.5,
  message: "Paper indexed.",
};

const BUDGET_USAGE = {
  calls_today: 3,
  calls_total: 7,
  tokens_charged_today: 900,
  tokens_charged_total: 2100,
  calls_unsettled: 0,
  daily_call_allowance: 20,
  total_call_allowance: 20,
  daily_token_allowance: 100_000,
  total_token_allowance: 100_000,
};

/** GET /ready as the backend really sends it, including the fields that must stay server-side. */
const READY = {
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
    usage: { ...BUDGET_USAGE, ledger_created_at: "2026-09-01T00:00:00Z" },
    token_bound: "tiktoken/o200k_base",
  },
  limits: { max_pdf_bytes: 10_485_760, max_pages: 40 },
};

describe("parsePaperList", () => {
  it("accepts the documented shape and drops unknown fields", () => {
    const papers = parsePaperList([{ ...PAPER, extra: "leak" }]);
    expect(papers).toEqual([PAPER]);
    expect(papers?.[0]).not.toHaveProperty("extra");
  });

  it("accepts an empty corpus", () => {
    expect(parsePaperList([])).toEqual([]);
  });

  it.each([
    ["an object instead of an array", { papers: [PAPER] }],
    ["an item without paper_id", [{ ...PAPER, paper_id: undefined }]],
    ["a numeric paper_id", [{ ...PAPER, paper_id: 42 }]],
    ["a non-object item", [DIGEST]],
    ["null", null],
  ])("returns null for %s", (_label, input) => {
    expect(parsePaperList(input)).toBeNull();
  });

  it("coerces optional fields: non-integers to null, missing counts to 0, missing status to ready", () => {
    const [paper] = parsePaperList([{ paper_id: DIGEST, pages: 12.5, chunks: null, uploaded_at: 123 }]) ?? [];
    expect(paper).toEqual({ paper_id: DIGEST, filename: null, pages: null, chunks: 0, uploaded_at: null, status: "ready" });
    const [stringy] = parsePaperList([{ paper_id: DIGEST, pages: "12", chunks: "3", status: "pending_cleanup" }]) ?? [];
    expect(stringy).toMatchObject({ pages: null, chunks: 0, status: "pending_cleanup" });
  });
});

describe("parseQueryResponse", () => {
  it("accepts the documented shape and drops unknown fields at every level", () => {
    const input = {
      ...QUERY,
      extra: "leak",
      citations: [{ ...CITATION, embedding: [0.1, 0.2] }],
      model_usage: { ...USAGE, extra: "leak" },
    };
    const parsed = parseQueryResponse(input);
    expect(parsed).toEqual(QUERY);
    expect(parsed).not.toHaveProperty("extra");
    expect(parsed?.citations[0]).not.toHaveProperty("embedding");
    expect(parsed?.model_usage).not.toHaveProperty("extra");
  });

  it("accepts an answer with no citations", () => {
    expect(parseQueryResponse({ ...QUERY, citations: [] })?.citations).toEqual([]);
  });

  it.each([
    ["a missing answer", { ...QUERY, answer: undefined }],
    ["a non-string answer", { ...QUERY, answer: 42 }],
    ["a missing question", { ...QUERY, question: undefined }],
    ["citations that are not an array", { ...QUERY, citations: "none" }],
    ["citations given as an object", { ...QUERY, citations: { 0: CITATION } }],
    ["a citation without text", { ...QUERY, citations: [{ ...CITATION, text: undefined }] }],
    ["a citation without paper", { ...QUERY, citations: [{ ...CITATION, paper: null }] }],
    ["a citation that is not an object", { ...QUERY, citations: ["quote"] }],
    ["an array", [QUERY]],
    ["null", null],
    ["a string", "answer"],
  ])("returns null for %s", (_label, input) => {
    expect(parseQueryResponse(input)).toBeNull();
  });

  it("fills documented fallbacks for missing or invalid optional fields", () => {
    const parsed = parseQueryResponse({
      question: "q?",
      answer: "a",
      citations: [{ text: "t", paper: "p.pdf", page: 2.5, relevance_score: "0.9" }],
      retrieval_time_ms: null,
      total_time_ms: Number.NaN,
    });
    expect(parsed).toEqual({
      request_id: "unknown",
      question: "q?",
      answer: "a",
      citations: [{ text: "t", page: null, paper: "p.pdf", relevance_score: 0, paper_id: null, chunk_id: null }],
      papers_searched: 0,
      retrieval_time_ms: 0,
      generation_time_ms: 0,
      total_time_ms: 0,
      model_used: "unknown",
      model_usage: null,
    });
  });

  it("drops model_usage that lacks its accounting label and nulls non-integer token counts", () => {
    expect(parseQueryResponse({ ...QUERY, model_usage: { input_tokens: 5 } })?.model_usage).toBeNull();
    expect(parseQueryResponse({ ...QUERY, model_usage: null })?.model_usage).toBeNull();
    const reserved = parseQueryResponse({
      ...QUERY,
      model_usage: { accounting: "reserved", input_tokens: null, output_tokens: 1.5, tokens_reserved: 2000, reservation_bound: "tiktoken/o200k_base" },
    });
    expect(reserved?.model_usage).toEqual({
      accounting: "reserved",
      input_tokens: null,
      output_tokens: null,
      tokens_charged: 0,
      tokens_reserved: 2000,
      context_chars: 0,
      reservation_bound: "tiktoken/o200k_base",
    });
  });
});

describe("parseUploadResponse", () => {
  it("accepts the documented shape and drops unknown fields", () => {
    const parsed = parseUploadResponse({ ...UPLOAD, extra: "leak" });
    expect(parsed).toEqual(UPLOAD);
    expect(parsed).not.toHaveProperty("extra");
  });

  it.each([
    ["a missing paper_id", { ...UPLOAD, paper_id: undefined }],
    ["a non-string filename", { ...UPLOAD, filename: null }],
    ["an array", [UPLOAD]],
    ["null", null],
  ])("returns null for %s", (_label, input) => {
    expect(parseUploadResponse(input)).toBeNull();
  });

  it("coerces invalid numbers to 0 and a missing message to an empty string", () => {
    expect(parseUploadResponse({ paper_id: DIGEST, filename: "p.pdf", pages: "15", chunks: null })).toEqual({
      paper_id: DIGEST,
      filename: "p.pdf",
      pages: 0,
      chunks: 0,
      processing_time_ms: 0,
      message: "",
    });
  });
});

describe("parseReadiness", () => {
  it("projects /ready onto the browser contract and keeps limits and ledger identity server-side", () => {
    const status = parseReadiness(READY, true);
    expect(status).toEqual({
      ready: true,
      configured_model: "gpt-4o-mini",
      effective_generation: "openai",
      effective_retrieval: "chroma",
      init_error: null,
      budget_state: "ok",
      usage: BUDGET_USAGE,
    });
    const serialized = JSON.stringify(status);
    for (const hidden of ["limits", "ledger_created_at", "token_bound", "access_configured", "pending_cleanup_ids", "configured_provider"]) {
      expect(serialized, hidden).not.toContain(hidden);
    }
  });

  it("uses the body's ready flag when present and the HTTP status otherwise", () => {
    expect(parseReadiness({ ...READY, ready: false }, true)?.ready).toBe(false);
    expect(parseReadiness({ ...READY, ready: undefined }, true)?.ready).toBe(true);
    expect(parseReadiness({ ...READY, ready: "yes" }, false)?.ready).toBe(false);
  });

  it("nulls the budget when model_budget or its usage is missing or malformed", () => {
    expect(parseReadiness({ ...READY, model_budget: undefined }, true)).toMatchObject({ budget_state: null, usage: null });
    expect(parseReadiness({ ...READY, model_budget: { state: "unavailable" } }, true)).toMatchObject({
      budget_state: "unavailable",
      usage: null,
    });
    expect(parseReadiness({ ...READY, model_budget: { state: 7, usage: "n/a" } }, true)).toMatchObject({
      budget_state: null,
      usage: null,
    });
  });

  it("keeps only integer usage counters and drops everything else in the usage object", () => {
    const status = parseReadiness(
      { ...READY, model_budget: { state: "ok", usage: { calls_today: 1.5, calls_total: "7", tokens_charged_today: 900, extra: "leak" } } },
      true,
    );
    expect(status?.usage).toEqual({
      calls_today: null,
      calls_total: null,
      tokens_charged_today: 900,
      tokens_charged_total: null,
      calls_unsettled: null,
      daily_call_allowance: null,
      total_call_allowance: null,
      daily_token_allowance: null,
      total_token_allowance: null,
    });
    expect(status?.usage).not.toHaveProperty("extra");
  });

  it("falls back to 'unavailable' modes and an empty model when the fields are absent", () => {
    expect(parseReadiness({ ready: false, init_error: { nested: true } }, false)).toEqual({
      ready: false,
      configured_model: "",
      effective_generation: "unavailable",
      effective_retrieval: "unavailable",
      init_error: null,
      budget_state: null,
      usage: null,
    });
  });

  it.each([null, "ready", 42, [READY]])("returns null for a non-object body (%j)", (input) => {
    expect(parseReadiness(input, true)).toBeNull();
  });
});
