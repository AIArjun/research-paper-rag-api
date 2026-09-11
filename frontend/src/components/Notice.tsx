import type { ReactNode } from "react";
import type { ApiError, ErrorCategory } from "@/lib/shared/types";

const TITLES: Partial<Record<ErrorCategory, string>> = {
  busy: "The backend is busy",
  budget_exhausted: "Allowance used up",
  empty_corpus: "Nothing to search yet",
  backend_unavailable: "Backend not ready",
  backend_credential: "Backend credential refused",
  file_too_large: "PDF too large",
  invalid_pdf: "Not a readable PDF",
  pdf_limit: "PDF over the demo ceiling",
  corpus_full: "Corpus is full",
  timeout: "No answer in time",
  unreachable: "Backend unreachable",
  generation_failed: "Generation failed",
  unauthenticated: "Session ended",
  invalid_request: "Request not valid",
  forbidden_origin: "Request refused",
  storage: "Storage needs cleanup",
  not_found: "Paper not found",
  request_too_large: "Request too large",
  backend_error: "Unexpected backend answer",
  not_configured: "Not configured",
  invalid_passcode: "Passcode not accepted",
  too_many_attempts: "Too many attempts",
};

interface Props {
  error: ApiError;
  onDismiss?: () => void;
  action?: ReactNode;
  tone?: "error" | "info";
}

export function Notice({ error, onDismiss, action, tone = "error" }: Props) {
  return (
    <div className={`notice notice--${tone}`} role="alert">
      <div className="notice__body">
        <p className="notice__title">{TITLES[error.category] ?? "Something went wrong"}</p>
        <p className="notice__message">{error.message}</p>
        {(error.request_id || error.retry_after) && (
          <p className="notice__meta">
            {error.retry_after ? `Suggested wait: ${error.retry_after} s. ` : ""}
            {error.request_id ? `Request ${error.request_id}` : ""}
          </p>
        )}
      </div>
      <div className="notice__actions">
        {action}
        {onDismiss && (
          <button type="button" className="button button--quiet button--small" onClick={onDismiss} aria-label="Dismiss message">
            Dismiss
          </button>
        )}
      </div>
    </div>
  );
}
