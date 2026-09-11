#!/usr/bin/env node
/**
 * Local stand-in for the Research Paper RAG API, for developing the frontend
 * without a Render instance, credentials or paid model calls.
 *
 * It mimics the documented contract of app/main.py (routes, status codes,
 * error categories, response fields). Answers are OBVIOUSLY canned and marked
 * as such in the text; this script is never part of the deployed app.
 *
 *   node scripts/fake-backend.mjs            # listens on 127.0.0.1:8765
 *   RAG_API_URL=http://127.0.0.1:8765 RAG_API_TOKEN=fake-token-... npm run dev
 */
import { createHash } from "node:crypto";
import http from "node:http";

const PORT = Number(process.env.FAKE_PORT ?? 8765);
const TOKEN = process.env.FAKE_TOKEN ?? "fake-token-for-local-development-0123456789";
const MODE = process.env.FAKE_MODE ?? "normal"; // normal | busy | budget | down | slow | empty

const papers = new Map(); // paper_id -> PaperInfo
let calls = 0;
const ledgerCreated = new Date().toISOString();

function json(res, status, body, headers = {}) {
  res.writeHead(status, { "Content-Type": "application/json", ...headers });
  res.end(JSON.stringify(body));
}
function error(res, status, category, message, headers = {}, extra = {}) {
  json(res, status, { detail: { message, category, request_id: Math.random().toString(16).slice(2, 10), ...extra } }, headers);
}
function readiness() {
  return {
    ready: MODE !== "down",
    configured_provider: "openai",
    configured_model: "gpt-4o-mini",
    effective_retrieval: "chroma",
    effective_generation: MODE === "down" ? "unavailable" : "openai",
    init_error: MODE === "down" ? "ledger_unavailable" : null,
    pending_cleanup_ids: [],
    provider_connection_verified: false,
    access_configured: true,
    model_budget: {
      state: MODE === "budget" ? "exhausted" : "ok",
      configured: true,
      usage: {
        calls_today: 5 + calls, calls_total: 5 + calls,
        tokens_charged_today: 4119 + calls * 900, tokens_charged_total: 4119 + calls * 900,
        tokens_measured_total: 4119, tokens_reserved_total: 5787, calls_unsettled: 0,
        daily_call_allowance: 20, total_call_allowance: 20,
        daily_token_allowance: 100000, total_token_allowance: 100000,
        ledger_created_at: ledgerCreated,
      },
      token_bound: "tiktoken/o200k_base",
    },
    limits: { max_file_bytes: 10485760, max_pdf_pages: 60, max_question_chars: 2000, max_top_k: 5 },
  };
}

function readBody(req) {
  return new Promise((resolve) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
  });
}

function extractFile(buffer, contentType) {
  const boundary = /boundary=([^;]+)/.exec(contentType ?? "")?.[1];
  if (!boundary) return null;
  const marker = Buffer.from(`--${boundary}`);
  let start = buffer.indexOf(marker);
  while (start !== -1) {
    const headerEnd = buffer.indexOf("\r\n\r\n", start);
    if (headerEnd === -1) break;
    const header = buffer.subarray(start, headerEnd).toString("latin1");
    const next = buffer.indexOf(marker, headerEnd);
    const body = buffer.subarray(headerEnd + 4, next === -1 ? buffer.length : next - 2);
    if (/name="file"/.test(header)) {
      const filename = /filename="([^"]*)"/.exec(header)?.[1] ?? "upload.pdf";
      return { filename, body };
    }
    start = next;
  }
  return null;
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", "http://localhost");
  const publicRoute = url.pathname === "/ready" || url.pathname === "/health";
  if (!publicRoute) {
    const auth = req.headers.authorization ?? "";
    if (auth !== `Bearer ${TOKEN}`) {
      return error(res, 401, "unauthorized", "A valid bearer token is required.", { "WWW-Authenticate": "Bearer" });
    }
  }
  if (MODE === "slow") await new Promise((r) => setTimeout(r, 4000));

  if (url.pathname === "/ready" || url.pathname === "/health") {
    const body = readiness();
    return json(res, body.ready ? 200 : 503, body);
  }
  if (url.pathname === "/papers" && req.method === "GET") {
    return json(res, 200, [...papers.values()]);
  }
  if (url.pathname === "/papers/upload" && req.method === "POST") {
    if (MODE === "busy") return error(res, 429, "busy", "The demo is busy with another request. Retry shortly.", { "Retry-After": "5" });
    if (MODE === "down") return error(res, 503, "ledger_unavailable", "The configured backend is not ready for uploads.");
    const buffer = await readBody(req);
    const file = extractFile(buffer, req.headers["content-type"]);
    if (!file) return error(res, 400, "invalid_request", "The request could not be parsed.");
    if (!file.body.subarray(0, 1024).includes("%PDF-")) return error(res, 400, "invalid_pdf", "The file could not be parsed as a text PDF.");
    await new Promise((r) => setTimeout(r, 1500));
    const paper_id = createHash("sha256").update(file.body).digest("hex");
    const pages = Math.max(1, Math.round(file.body.length / 150000));
    const chunks = pages * 8;
    papers.set(paper_id, { paper_id, filename: file.filename, pages, chunks, uploaded_at: new Date().toISOString(), status: "ready" });
    return json(res, 200, { paper_id, filename: file.filename, pages, chunks, processing_time_ms: 1500, message: `Paper '${file.filename}' is indexed. ${chunks} chunks available.` });
  }
  if (url.pathname === "/query" && req.method === "POST") {
    const body = JSON.parse((await readBody(req)).toString("utf8") || "{}");
    if (MODE === "empty" || papers.size === 0) return error(res, 400, "empty_corpus", "No papers uploaded yet.");
    if (MODE === "busy") return error(res, 429, "busy", "The demo is busy with another request. Retry shortly.", { "Retry-After": "5" });
    if (MODE === "budget") return error(res, 429, "budget_exhausted", "The model-call allowance is exhausted; no model request was made.", {}, { scope: "day", kind: "calls" });
    if (MODE === "down") return error(res, 503, "ledger_unavailable", "The configured backend is not ready for queries.");
    const scoped = body.paper_id ? [...papers.values()].filter((p) => p.paper_id === body.paper_id) : [...papers.values()];
    if (scoped.length === 0) {
      return json(res, 200, { request_id: "fake-abstain", question: body.question, answer: "The retrieved sources contain insufficient evidence to answer this question.", citations: [], papers_searched: 0, retrieval_time_ms: 12.1, generation_time_ms: 0, total_time_ms: 12.1, model_used: "not-invoked", model_usage: null });
    }
    await new Promise((r) => setTimeout(r, 2500));
    calls += 1;
    const topK = Math.min(5, Math.max(1, Number(body.top_k ?? 5)));
    const citations = Array.from({ length: topK }, (_, i) => {
      const paper = scoped[i % scoped.length];
      return {
        text: `[FAKE BACKEND passage ${i + 1}] Placeholder text standing in for a retrieved chunk of ${paper.filename}. This is not a real model output.`,
        page: ((i * 3) % (paper.pages || 1)) + 1,
        paper: paper.filename,
        relevance_score: Number((0.5 - i * 0.04).toFixed(4)),
        paper_id: paper.paper_id,
        chunk_id: `${paper.paper_id.slice(0, 8)}-${i}`,
      };
    });
    const first = citations[0];
    const answer = `**Fake backend answer** for local development only; no model was called.\n\nThe question was: ${body.question}\n\nA canned claim with a matching reference (Source: ${first.paper}, Page ${first.page}). A second canned claim [Source: ${citations[citations.length - 1].paper}, Page ${citations[citations.length - 1].page}]. A reference that matches nothing (Source: missing.pdf, Page 99) stays plain text.`;
    return json(res, 200, { request_id: `fake-${calls}`, question: body.question, answer, citations, papers_searched: new Set(citations.map((c) => c.paper)).size, retrieval_time_ms: 41.3, generation_time_ms: 2400.2, total_time_ms: 2441.5, model_used: "fake-backend", model_usage: { accounting: "measured", input_tokens: 812, output_tokens: 96, tokens_charged: 908, tokens_reserved: 1450, context_chars: 2400, reservation_bound: "tiktoken/o200k_base" } });
  }
  return error(res, 404, "not_found", "Not found.");
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`fake RAG backend on http://127.0.0.1:${PORT} (mode: ${MODE}); token: ${TOKEN}`);
});
