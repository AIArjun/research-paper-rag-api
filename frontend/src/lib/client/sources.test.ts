import { describe, expect, it } from "vitest";
import { embeddedPageUrl, pageUrl, paperTitle, resolveSource, sampleInCorpus, sha256Hex } from "./sources";
import { SAMPLE_PAPERS } from "@/lib/shared/samples";
import type { PaperInfo } from "@/lib/shared/types";

const sample = SAMPLE_PAPERS[0]!;
const papers: PaperInfo[] = [
  { paper_id: sample.sha256, filename: sample.file, pages: 15, chunks: 110, uploaded_at: null, status: "ready" },
  { paper_id: "b".repeat(64), filename: "mine.pdf", pages: 9, chunks: 30, uploaded_at: null, status: "ready" },
];

describe("resolveSource", () => {
  it("maps a citation to the bundled sample only when the digest matches", () => {
    const hit = resolveSource({ paper_id: sample.sha256, paper: "renamed.pdf" }, papers, new Map());
    expect(hit).toMatchObject({ kind: "sample", url: `/samples/${sample.file}`, pages: sample.pages });
    expect(resolveSource({ paper_id: sample.sha256.toUpperCase(), paper: sample.file }, papers, new Map())?.kind).toBe("sample");
    expect(resolveSource({ paper_id: "c".repeat(64), paper: sample.file }, papers, new Map())).toBeNull();
  });

  it("uses a session-held file only for the exact paper id, never by filename", () => {
    const local = new Map([["b".repeat(64), { url: "blob:abc", name: "mine.pdf", bytes: 10 }]]);
    expect(resolveSource({ paper_id: "b".repeat(64), paper: "mine.pdf" }, papers, local)).toMatchObject({ kind: "local", url: "blob:abc", pages: 9 });
    expect(resolveSource({ paper_id: "d".repeat(64), paper: "mine.pdf" }, papers, local)).toBeNull();
    expect(resolveSource({ paper_id: null, paper: "mine.pdf" }, papers, local)).toBeNull();
  });

  it("builds viewer URLs with the physical page", () => {
    expect(pageUrl("/samples/a.pdf", 4)).toBe("/samples/a.pdf#page=4");
    expect(pageUrl("/samples/a.pdf", null)).toBe("/samples/a.pdf");
    expect(embeddedPageUrl("blob:x", 7)).toBe("blob:x#page=7&view=FitH&navpanes=0");
  });

  it("titles and corpus membership follow the digest", () => {
    expect(paperTitle(sample.sha256, "whatever.pdf")).toBe(sample.title);
    expect(paperTitle("e".repeat(64), "whatever.pdf")).toBe("whatever.pdf");
    expect(sampleInCorpus(sample, papers)).toBe(true);
    expect(sampleInCorpus(SAMPLE_PAPERS[1]!, papers)).toBe(false);
    expect(sampleInCorpus(sample, [{ ...papers[0]!, status: "pending_cleanup" }])).toBe(false);
  });

  it("hashes with Web Crypto when available", async () => {
    const digest = await sha256Hex(new TextEncoder().encode("abc").buffer as ArrayBuffer);
    expect(digest).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  });
});
