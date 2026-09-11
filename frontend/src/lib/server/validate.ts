import type {
  BudgetUsage,
  Citation,
  ModelUsage,
  ObservatoryStatus,
  PaperInfo,
  QueryResponse,
  UploadResponse,
} from "@/lib/shared/types";

/**
 * Runtime checks for upstream payloads. Unknown fields are dropped so that
 * the browser only ever sees the documented contract.
 */

type Obj = Record<string, unknown>;

function obj(value: unknown): Obj | null {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Obj) : null;
}
function str(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}
function strOrNull(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}
function num(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}
function numOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
function intOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) ? value : null;
}

export function parsePaperList(value: unknown): PaperInfo[] | null {
  if (!Array.isArray(value)) return null;
  const papers: PaperInfo[] = [];
  for (const item of value) {
    const o = obj(item);
    if (!o || typeof o.paper_id !== "string") return null;
    papers.push({
      paper_id: o.paper_id,
      filename: strOrNull(o.filename),
      pages: intOrNull(o.pages),
      chunks: num(o.chunks),
      uploaded_at: strOrNull(o.uploaded_at),
      status: str(o.status, "ready"),
    });
  }
  return papers;
}

function parseCitation(value: unknown): Citation | null {
  const o = obj(value);
  if (!o || typeof o.text !== "string" || typeof o.paper !== "string") return null;
  return {
    text: o.text,
    page: intOrNull(o.page),
    paper: o.paper,
    relevance_score: num(o.relevance_score),
    paper_id: strOrNull(o.paper_id),
    chunk_id: strOrNull(o.chunk_id),
  };
}

function parseUsage(value: unknown): ModelUsage | null {
  const o = obj(value);
  if (!o || typeof o.accounting !== "string") return null;
  return {
    accounting: o.accounting,
    input_tokens: intOrNull(o.input_tokens),
    output_tokens: intOrNull(o.output_tokens),
    tokens_charged: num(o.tokens_charged),
    tokens_reserved: num(o.tokens_reserved),
    context_chars: num(o.context_chars),
    reservation_bound: strOrNull(o.reservation_bound),
  };
}

export function parseQueryResponse(value: unknown): QueryResponse | null {
  const o = obj(value);
  if (!o || typeof o.answer !== "string" || typeof o.question !== "string" || !Array.isArray(o.citations)) return null;
  const citations: Citation[] = [];
  for (const c of o.citations) {
    const parsed = parseCitation(c);
    if (!parsed) return null;
    citations.push(parsed);
  }
  return {
    request_id: str(o.request_id, "unknown"),
    question: o.question,
    answer: o.answer,
    citations,
    papers_searched: num(o.papers_searched),
    retrieval_time_ms: num(o.retrieval_time_ms),
    generation_time_ms: num(o.generation_time_ms),
    total_time_ms: num(o.total_time_ms),
    model_used: str(o.model_used, "unknown"),
    model_usage: parseUsage(o.model_usage),
  };
}

export function parseUploadResponse(value: unknown): UploadResponse | null {
  const o = obj(value);
  if (!o || typeof o.paper_id !== "string" || typeof o.filename !== "string") return null;
  return {
    paper_id: o.paper_id,
    filename: o.filename,
    pages: num(o.pages),
    chunks: num(o.chunks),
    processing_time_ms: num(o.processing_time_ms),
    message: str(o.message),
  };
}

function parseUsageSummary(value: unknown): BudgetUsage | null {
  const o = obj(value);
  if (!o) return null;
  return {
    calls_today: intOrNull(o.calls_today),
    calls_total: intOrNull(o.calls_total),
    tokens_charged_today: intOrNull(o.tokens_charged_today),
    tokens_charged_total: intOrNull(o.tokens_charged_total),
    calls_unsettled: intOrNull(o.calls_unsettled),
    daily_call_allowance: intOrNull(o.daily_call_allowance),
    total_call_allowance: intOrNull(o.total_call_allowance),
    daily_token_allowance: intOrNull(o.daily_token_allowance),
    total_token_allowance: intOrNull(o.total_token_allowance),
  };
}

/** Project GET /ready onto what the browser needs; the limits object and ledger identity stay server-side. */
export function parseReadiness(value: unknown, httpOk: boolean): ObservatoryStatus | null {
  const o = obj(value);
  if (!o) return null;
  const budget = obj(o.model_budget);
  return {
    ready: typeof o.ready === "boolean" ? o.ready : httpOk,
    configured_model: str(o.configured_model),
    effective_generation: str(o.effective_generation, "unavailable"),
    effective_retrieval: str(o.effective_retrieval, "unavailable"),
    init_error: strOrNull(o.init_error),
    budget_state: budget ? strOrNull(budget.state) : null,
    usage: budget ? parseUsageSummary(budget.usage) : null,
  };
}

export { numOrNull };
