import { describe, expect, it } from "vitest";
import { summarizeBudget } from "./budget";

const usage = {
  calls_today: 5, calls_total: 5, tokens_charged_today: 4119, tokens_charged_total: 4119, calls_unsettled: 0,
  daily_call_allowance: 20, total_call_allowance: 20, daily_token_allowance: 100000, total_token_allowance: 100000,
};

describe("summarizeBudget", () => {
  it("shows today and lifetime calls when nothing is exhausted", () => {
    const s = summarizeBudget({ budget_state: "ok", usage });
    expect(s.exhausted).toBe(false);
    expect(s.short).toBe("calls 5/20 today · 5/20 lifetime");
    expect(s.detail).toContain("Tokens lifetime 4,119/100,000.");
  });

  it("derives exhaustion from each allowance even when the backend state says ok", () => {
    expect(summarizeBudget({ budget_state: "ok", usage: { ...usage, calls_total: 20 } })).toMatchObject({ exhausted: true, exhaustedScope: "lifetime", short: "allowance used up (lifetime)" });
    expect(summarizeBudget({ budget_state: "ok", usage: { ...usage, calls_today: 20, calls_total: 7 } })).toMatchObject({ exhausted: true, exhaustedScope: "today" });
    expect(summarizeBudget({ budget_state: "ok", usage: { ...usage, tokens_charged_total: 100000 } }).exhaustedScope).toBe("lifetime");
    expect(summarizeBudget({ budget_state: "ok", usage: { ...usage, tokens_charged_today: 100000 } }).exhaustedScope).toBe("today");
    expect(summarizeBudget({ budget_state: "exhausted", usage }).exhausted).toBe(true);
  });

  it("copes with missing usage or allowances", () => {
    expect(summarizeBudget(null)).toMatchObject({ exhausted: false, short: null });
    expect(summarizeBudget({ budget_state: "ok", usage: { ...usage, daily_call_allowance: null, total_call_allowance: null } }).short).toBe("calls 5 today · 5 lifetime");
    expect(summarizeBudget({ budget_state: "ok", usage: { ...usage, calls_total: null } }).short).toBeNull();
  });
});
