import type { BudgetUsage, ObservatoryStatus } from "@/lib/shared/types";

export type BudgetScope = "today" | "lifetime";

export interface BudgetSummary {
  /** True when any daily or lifetime call/token allowance is met or exceeded, or the backend says so. */
  exhausted: boolean;
  exhaustedScope: BudgetScope | null;
  /** Short line for the status pill, e.g. "calls 5/20 today · 5/20 lifetime". */
  short: string | null;
  /** Longer explanation for the tooltip / screen readers. */
  detail: string;
}

function met(used: number | null, allowance: number | null): boolean {
  return used !== null && allowance !== null && allowance > 0 && used >= allowance;
}

function pair(used: number | null, allowance: number | null): string | null {
  if (used === null) return null;
  return allowance !== null ? `${used.toLocaleString()}/${allowance.toLocaleString()}` : used.toLocaleString();
}

/**
 * The backend's `model_budget.state` reports whether the ledger is readable,
 * not whether the allowance is spent, so exhaustion is derived here from the
 * four allowances the backend enforces (calls and tokens, per day and lifetime).
 */
export function summarizeBudget(status: Pick<ObservatoryStatus, "budget_state" | "usage"> | null): BudgetSummary {
  const usage: BudgetUsage | null = status?.usage ?? null;
  if (!usage) {
    return { exhausted: status?.budget_state === "exhausted", exhaustedScope: null, short: null, detail: "The backend did not report allowance usage." };
  }
  const lifetime = met(usage.calls_total, usage.total_call_allowance) || met(usage.tokens_charged_total, usage.total_token_allowance);
  const today = met(usage.calls_today, usage.daily_call_allowance) || met(usage.tokens_charged_today, usage.daily_token_allowance);
  const exhausted = lifetime || today || status?.budget_state === "exhausted";
  const exhaustedScope: BudgetScope | null = lifetime ? "lifetime" : today ? "today" : null;

  const callsToday = pair(usage.calls_today, usage.daily_call_allowance);
  const callsTotal = pair(usage.calls_total, usage.total_call_allowance);
  const tokensToday = pair(usage.tokens_charged_today, usage.daily_token_allowance);
  const tokensTotal = pair(usage.tokens_charged_total, usage.total_token_allowance);

  const short = exhausted
    ? `allowance used up${exhaustedScope ? ` (${exhaustedScope})` : ""}`
    : callsTotal
      ? `calls ${callsToday ?? "?"} today · ${callsTotal} lifetime`
      : null;
  const detail = [
    "Shared model-call allowance, read from the backend ledger.",
    callsToday ? `Calls today ${callsToday}.` : "",
    callsTotal ? `Calls lifetime ${callsTotal}.` : "",
    tokensToday ? `Tokens today ${tokensToday}.` : "",
    tokensTotal ? `Tokens lifetime ${tokensTotal}.` : "",
    exhausted ? "No further model call will be accepted until the allowance changes." : "",
  ]
    .filter(Boolean)
    .join(" ");
  return { exhausted, exhaustedScope, short, detail };
}
