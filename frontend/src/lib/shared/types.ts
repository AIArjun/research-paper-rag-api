/**
 * Types mirroring the Research Paper RAG API's documented response models
 * (app/main.py) and the narrow BFF contract this frontend exposes to the browser.
 * Nothing here is invented: every field exists in the backend's Pydantic models.
 */

export interface PaperInfo {
  paper_id: string;
  filename: string | null;
  pages: number | null;
  chunks: number;
  uploaded_at: string | null;
  status: string; // "ready" | "pending_cleanup"
}

export interface Citation {
  text: string;
  page: number | null;
  paper: string;
  relevance_score: number;
  paper_id: string | null;
  chunk_id: string | null;
}

export interface ModelUsage {
  accounting: string; // "measured" | "reserved"
  input_tokens: number | null;
  output_tokens: number | null;
  tokens_charged: number;
  tokens_reserved: number;
  context_chars: number;
  reservation_bound: string | null;
}

export interface QueryResponse {
  request_id: string;
  question: string;
  answer: string;
  citations: Citation[];
  papers_searched: number;
  retrieval_time_ms: number;
  generation_time_ms: number;
  total_time_ms: number;
  model_used: string; // e.g. "gpt-4o-mini" | "not-invoked" | "demo-mode"
  model_usage: ModelUsage | null;
}

export interface UploadResponse {
  paper_id: string;
  filename: string;
  pages: number;
  chunks: number;
  processing_time_ms: number;
  message: string;
}

export interface BudgetUsage {
  calls_today: number | null;
  calls_total: number | null;
  tokens_charged_today: number | null;
  tokens_charged_total: number | null;
  calls_unsettled: number | null;
  daily_call_allowance: number | null;
  total_call_allowance: number | null;
  daily_token_allowance: number | null;
  total_token_allowance: number | null;
}

/** Projection of GET /ready that the browser is allowed to see. */
export interface ObservatoryStatus {
  ready: boolean;
  configured_model: string;
  effective_generation: string;
  effective_retrieval: string;
  init_error: string | null;
  budget_state: string | null; // "ok" | "exhausted" | "not_applicable" | "unavailable" | ...
  usage: BudgetUsage | null;
}

/** Error categories this frontend surfaces. Upstream categories are mapped onto these. */
export type ErrorCategory =
  | "not_configured"
  | "unauthenticated"
  | "invalid_passcode"
  | "too_many_attempts"
  | "forbidden_origin"
  | "invalid_request"
  | "request_too_large"
  | "invalid_pdf"
  | "file_too_large"
  | "pdf_limit"
  | "corpus_full"
  | "empty_corpus"
  | "busy"
  | "budget_exhausted"
  | "backend_unavailable"
  | "backend_credential"
  | "storage"
  | "generation_failed"
  | "not_found"
  | "timeout"
  | "unreachable"
  | "backend_error";

export interface ApiError {
  category: ErrorCategory;
  message: string;
  request_id?: string;
  retry_after?: number;
  upstream_category?: string;
}

export interface ApiErrorBody {
  error: ApiError;
}

export interface PapersResponse {
  papers: PaperInfo[];
}
