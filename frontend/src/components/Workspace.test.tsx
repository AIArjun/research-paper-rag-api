// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { PaperInfo, QueryResponse } from "@/lib/shared/types";
import { SAMPLE_PAPERS } from "@/lib/shared/samples";

const router = vi.hoisted(() => ({ refresh: vi.fn() }));
vi.mock("next/navigation", () => ({ useRouter: () => router }));

import { Workspace } from "./Workspace";

const ATTENTION = SAMPLE_PAPERS[0]!;
const papers: PaperInfo[] = [
  { paper_id: ATTENTION.sha256, filename: ATTENTION.file, pages: 15, chunks: 110, uploaded_at: null, status: "ready" },
  { paper_id: "a".repeat(64), filename: "other.pdf", pages: 3, chunks: 12, uploaded_at: null, status: "ready" },
];
const status = {
  ready: true, configured_model: "gpt-4o-mini", effective_generation: "openai", effective_retrieval: "chroma",
  init_error: null, budget_state: "ok",
  usage: { calls_today: 5, calls_total: 5, tokens_charged_today: 1, tokens_charged_total: 1, calls_unsettled: 0, daily_call_allowance: 20, total_call_allowance: 20, daily_token_allowance: 100000, total_token_allowance: 100000 },
};
const answer: QueryResponse = {
  request_id: "req-1", question: "Why does scaled dot-product attention divide by the square root of d_k?",
  answer: "Because large dot products push softmax into tiny gradients (Source: attention-is-all-you-need.pdf, Page 4). <b>not html</b>",
  citations: [
    { text: "We suspect that for large values of d_k…", page: 4, paper: ATTENTION.file, relevance_score: 0.44, paper_id: ATTENTION.sha256, chunk_id: "c1" },
    { text: "Another passage", page: 2, paper: "other.pdf", relevance_score: 0.31, paper_id: "a".repeat(64), chunk_id: "c2" },
  ],
  papers_searched: 2, retrieval_time_ms: 40, generation_time_ms: 1200, total_time_ms: 1240, model_used: "gpt-4o-mini",
  model_usage: { accounting: "measured", input_tokens: 800, output_tokens: 90, tokens_charged: 890, tokens_reserved: 1400, context_chars: 2400, reservation_bound: "tiktoken/o200k_base" },
};

type Handler = (init: RequestInit | undefined) => Promise<Response>;
const json = (body: unknown, init: ResponseInit = {}) =>
  new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" }, ...init });

let queryHandler: Handler;
const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
  const url = String(input);
  if (url.endsWith("/api/papers")) return json({ papers });
  if (url.endsWith("/api/status")) return json(status);
  if (url.endsWith("/api/query")) return queryHandler(init);
  if (url.endsWith("/api/auth/logout")) return new Response(null, { status: 204 });
  throw new Error(`unexpected fetch ${url}`);
});

const queryCalls = () => fetchMock.mock.calls.filter(([input]) => String(input).endsWith("/api/query"));

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
  Element.prototype.scrollIntoView = vi.fn();
  queryHandler = async () => json(answer);
});
afterEach(() => {
  cleanup();
  fetchMock.mockClear();
});

async function openWorkspace() {
  const view = render(<Workspace />);
  await screen.findByRole("radio", { name: /Attention Is All You Need/ });
  return view;
}

describe("Workspace never queries the model without an explicit submit", () => {
  it("loads the library and status on mount without touching /api/query", async () => {
    await openWorkspace();
    const urls = fetchMock.mock.calls.map(([input]) => String(input));
    expect(urls.some((u) => u.endsWith("/api/papers"))).toBe(true);
    expect(urls.some((u) => u.endsWith("/api/status"))).toBe(true);
    expect(queryCalls()).toHaveLength(0);
  });

  it("does not query on paper selection, prompt fill, or remount", async () => {
    const view = await openWorkspace();
    fireEvent.click(screen.getByRole("radio", { name: /Attention Is All You Need/ }));
    fireEvent.click(screen.getAllByRole("button", { name: /scaled dot-product attention/ })[0]!);
    expect((screen.getByLabelText("Your question") as HTMLTextAreaElement).value).toMatch(/scaled dot-product/);
    view.unmount();
    await openWorkspace();
    await new Promise((r) => setTimeout(r, 50));
    expect(queryCalls()).toHaveLength(0);
  });

  it("sends exactly one query on Ask, scoped to the selected paper, and disables duplicate submits", async () => {
    let release: (value: Response) => void = () => undefined;
    queryHandler = () => new Promise<Response>((resolve) => { release = resolve; });
    await openWorkspace();
    fireEvent.click(screen.getByRole("radio", { name: /Attention Is All You Need/ }));
    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "Why divide by sqrt(d_k)?" } });
    const ask = screen.getByRole("button", { name: "Ask" });
    fireEvent.click(ask);
    fireEvent.click(ask);
    fireEvent.submit(ask.closest("form")!);
    await waitFor(() => expect((screen.getByRole("button", { name: "Asking…" }) as HTMLButtonElement).disabled).toBe(true));
    expect(queryCalls()).toHaveLength(1);
    const body = JSON.parse(String(queryCalls()[0]![1]?.body));
    expect(body).toEqual({ question: "Why divide by sqrt(d_k)?", top_k: 5, paper_id: ATTENTION.sha256 });
    release(json(answer));
    await screen.findByText("Research preview · Check the cited sources");
    expect(queryCalls()).toHaveLength(1);
  });

  it("renders the answer safely with citation chips only for returned citations", async () => {
    await openWorkspace();
    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "Why divide by sqrt(d_k)?" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));
    await screen.findByText("Research preview · Check the cited sources");
    expect(document.querySelector("b")).toBeNull();
    expect(screen.getByText(/<b>not html<\/b>/)).toBeTruthy();
    const chips = document.querySelectorAll(".cite");
    expect(chips).toHaveLength(1);
    expect(chips[0]?.getAttribute("aria-label")).toBe("Show evidence: Attention, page 4");
    fireEvent.click(chips[0]!);
    expect(screen.getByRole("heading", { name: /Page 4/ })).toBeTruthy();
    expect(screen.getByText(/similarity 0.440/)).toBeTruthy();
    expect(screen.getByText("gpt-4o-mini")).toBeTruthy();
    expect(screen.getByText(/800 in · 90 out/)).toBeTruthy();
    const frame = document.querySelector("iframe");
    expect(frame?.getAttribute("src")).toContain(`/samples/${ATTENTION.file}#page=4`);
  });

  it("shows a categorized error and never retries on its own", async () => {
    queryHandler = async () => json({ error: { category: "busy", message: "The backend is busy.", retry_after: 5, request_id: "r9" } }, { status: 429 });
    await openWorkspace();
    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "Why divide by sqrt(d_k)?" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));
    await screen.findByText("The backend is busy");
    expect(screen.getByText(/Suggested wait: 5 s/)).toBeTruthy();
    await new Promise((r) => setTimeout(r, 100));
    expect(queryCalls()).toHaveLength(1);
    queryHandler = async () => json(answer);
    fireEvent.click(screen.getByRole("button", { name: "Ask again" }));
    await screen.findByText("Research preview · Check the cited sources");
    expect(queryCalls()).toHaveLength(2);
  });

  it("explains an abstention as no model call", async () => {
    queryHandler = async () => json({ ...answer, citations: [], papers_searched: 0, model_used: "not-invoked", model_usage: null, generation_time_ms: 0 });
    await openWorkspace();
    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "Anything at all?" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));
    await screen.findByText(/no model call was made/);
    expect(screen.queryByRole("heading", { name: "Evidence" })).toBeNull();
  });
});
