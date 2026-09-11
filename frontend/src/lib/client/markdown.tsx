import { Fragment, type ReactNode } from "react";
import type { Citation } from "@/lib/shared/types";

/**
 * A deliberately small Markdown renderer that emits React elements only.
 * No HTML is ever parsed or injected, so model output cannot smuggle markup.
 * Supported: paragraphs, headings (demoted to h3/h4), bullet and numbered
 * lists, fenced code, inline code, bold, italic, and "(Source: X, Page N)"
 * references that match a returned citation.
 */

export interface CitationRef {
  index: number; // position in the citations array
  citation: Citation;
}

export type CitationRenderer = (ref: CitationRef, key: string) => ReactNode;

const CITATION_PATTERN =
  /[([]\s*(?:Source:\s*)?([^,\])]*?)\s*,\s*(?:Page|p\.|pp\.)\s*(\d{1,4})\s*[)\]]/gi;

function normalizeName(name: string): string {
  return name.trim().toLowerCase().replace(/^['"“”]+|['"“”]+$/g, "");
}

/** Map a textual reference to the first returned citation with the same paper and page. */
export function findCitation(citations: readonly Citation[], paper: string, page: number): CitationRef | null {
  const wanted = normalizeName(paper);
  const index = citations.findIndex((c) => {
    const name = normalizeName(c.paper);
    const stem = name.replace(/\.pdf$/, "");
    return c.page === page && (name === wanted || stem === wanted || stem === wanted.replace(/\.pdf$/, ""));
  });
  if (index < 0) return null;
  const citation = citations[index];
  return citation ? { index, citation } : null;
}

type Inline = ReactNode;

function renderInline(text: string, citations: readonly Citation[], renderCitation: CitationRenderer, keyPrefix: string): Inline[] {
  const out: Inline[] = [];
  let cursor = 0;
  let n = 0;
  const pushText = (chunk: string) => {
    if (chunk.length > 0) out.push(...renderEmphasis(chunk, `${keyPrefix}-t${n++}`));
  };
  for (const match of text.matchAll(CITATION_PATTERN)) {
    const [whole, paper = "", pageText = ""] = match;
    const start = match.index ?? 0;
    const ref = findCitation(citations, paper, Number(pageText));
    if (!ref) continue;
    pushText(text.slice(cursor, start));
    out.push(renderCitation(ref, `${keyPrefix}-c${n++}`));
    cursor = start + whole.length;
  }
  pushText(text.slice(cursor));
  return out;
}

const EMPHASIS_PATTERN = /(`[^`\n]+`|\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|_[^_\n]+_)/g;

function renderEmphasis(text: string, keyPrefix: string): Inline[] {
  const parts = text.split(EMPHASIS_PATTERN);
  return parts.map((part, i) => {
    const key = `${keyPrefix}-${i}`;
    if (part.length === 0) return null;
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) return <code key={key}>{part.slice(1, -1)}</code>;
    if ((part.startsWith("**") && part.endsWith("**")) || (part.startsWith("__") && part.endsWith("__"))) {
      return part.length > 4 ? <strong key={key}>{part.slice(2, -2)}</strong> : part;
    }
    if ((part.startsWith("*") && part.endsWith("*")) || (part.startsWith("_") && part.endsWith("_"))) {
      return part.length > 2 ? <em key={key}>{part.slice(1, -1)}</em> : part;
    }
    return <Fragment key={key}>{part}</Fragment>;
  });
}

type Block =
  | { kind: "p"; lines: string[] }
  | { kind: "h"; level: number; text: string }
  | { kind: "ul"; items: string[] }
  | { kind: "ol"; items: string[] }
  | { kind: "code"; text: string };

function parseBlocks(source: string): Block[] {
  const lines = source.replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i] ?? "";
    if (line.trim() === "") { i += 1; continue; }
    if (/^\s*```/.test(line)) {
      const buf: string[] = [];
      i += 1;
      while (i < lines.length && !/^\s*```/.test(lines[i] ?? "")) { buf.push(lines[i] ?? ""); i += 1; }
      i += 1;
      blocks.push({ kind: "code", text: buf.join("\n") });
      continue;
    }
    const heading = /^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line);
    if (heading) { blocks.push({ kind: "h", level: heading[1]?.length ?? 1, text: heading[2] ?? "" }); i += 1; continue; }
    if (/^\s*[-*•]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*•]\s+/.test(lines[i] ?? "")) { items.push((lines[i] ?? "").replace(/^\s*[-*•]\s+/, "")); i += 1; }
      blocks.push({ kind: "ul", items });
      continue;
    }
    if (/^\s*\d{1,3}[.)]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d{1,3}[.)]\s+/.test(lines[i] ?? "")) { items.push((lines[i] ?? "").replace(/^\s*\d{1,3}[.)]\s+/, "")); i += 1; }
      blocks.push({ kind: "ol", items });
      continue;
    }
    const buf: string[] = [];
    while (i < lines.length && (lines[i] ?? "").trim() !== "" && !/^\s*(```|#{1,6}\s|[-*•]\s|\d{1,3}[.)]\s)/.test(lines[i] ?? "")) {
      buf.push((lines[i] ?? "").trim());
      i += 1;
    }
    blocks.push({ kind: "p", lines: buf });
  }
  return blocks;
}

export function renderMarkdown(source: string, citations: readonly Citation[], renderCitation: CitationRenderer): ReactNode[] {
  return parseBlocks(source).map((block, b) => {
    const key = `b${b}`;
    switch (block.kind) {
      case "code":
        return <pre key={key}><code>{block.text}</code></pre>;
      case "h": {
        const Tag = block.level <= 2 ? "h3" : "h4";
        return <Tag key={key}>{renderInline(block.text, citations, renderCitation, key)}</Tag>;
      }
      case "ul":
        return <ul key={key}>{block.items.map((item, j) => <li key={`${key}-${j}`}>{renderInline(item, citations, renderCitation, `${key}-${j}`)}</li>)}</ul>;
      case "ol":
        return <ol key={key}>{block.items.map((item, j) => <li key={`${key}-${j}`}>{renderInline(item, citations, renderCitation, `${key}-${j}`)}</li>)}</ol>;
      case "p":
        return <p key={key}>{renderInline(block.lines.join(" "), citations, renderCitation, key)}</p>;
    }
  });
}
