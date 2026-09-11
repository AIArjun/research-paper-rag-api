import {
  DEFAULT_TOP_K,
  PAPER_ID_PATTERN,
  QUESTION_MAX_CHARS,
  QUESTION_MIN_CHARS,
  TOP_K_MAX,
  TOP_K_MIN,
} from "@/lib/shared/limits";

export interface ValidQuery {
  question: string;
  paper_id?: string;
  top_k: number;
}

/** Mirror of the backend's QueryRequest bounds, applied before anything is forwarded. */
export function validateQuery(input: Record<string, unknown>): ValidQuery | null {
  const question = typeof input.question === "string" ? input.question.trim() : "";
  if (question.length < QUESTION_MIN_CHARS || question.length > QUESTION_MAX_CHARS) return null;
  const top_k = input.top_k === undefined || input.top_k === null ? DEFAULT_TOP_K : input.top_k;
  if (typeof top_k !== "number" || !Number.isInteger(top_k) || top_k < TOP_K_MIN || top_k > TOP_K_MAX) return null;
  const valid: ValidQuery = { question, top_k };
  if (input.paper_id !== undefined && input.paper_id !== null) {
    if (typeof input.paper_id !== "string" || !PAPER_ID_PATTERN.test(input.paper_id)) return null;
    valid.paper_id = input.paper_id;
  }
  return valid;
}
