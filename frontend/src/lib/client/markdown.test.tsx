import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { findCitation, renderMarkdown } from "./markdown";
import type { Citation } from "@/lib/shared/types";

const citations: Citation[] = [
  { text: "a", page: 4, paper: "attention-is-all-you-need.pdf", relevance_score: 0.4, paper_id: "x", chunk_id: "1" },
  { text: "b", page: 3, paper: "retrieval-augmented-generation.pdf", relevance_score: 0.3, paper_id: "y", chunk_id: "2" },
];
const chip = (ref: { index: number }, key: string) => <span key={key} data-cite={ref.index} />;
const html = (source: string) => renderToStaticMarkup(<>{renderMarkdown(source, citations, chip)}</>);

describe("renderMarkdown", () => {
  it("never emits raw HTML from the source", () => {
    const out = html('<script>alert(1)</script> <img src=x onerror=y> **bold** `code`');
    expect(out).not.toContain("<script>");
    expect(out).not.toContain("<img");
    expect(out).toContain("&lt;script&gt;");
    expect(out).toContain("<strong>bold</strong>");
    expect(out).toContain("<code>code</code>");
  });

  it("renders block structure", () => {
    const out = html("# Title\n\nPara one\ncontinued\n\n- a\n- b\n\n1. x\n2) y\n\n```\nraw <b>\n```");
    expect(out).toContain("<h3>Title</h3>");
    expect(out).toContain("<p>Para one continued</p>");
    expect(out).toContain("<ul><li>a</li><li>b</li></ul>");
    expect(out).toContain("<ol><li>x</li><li>y</li></ol>");
    expect(out).toContain("<pre><code>raw &lt;b&gt;</code></pre>");
  });

  it("turns only references that match a returned citation into chips", () => {
    const out = html("Claim (Source: attention-is-all-you-need.pdf, Page 4). Other [Source: retrieval-augmented-generation, p. 3]. Missing (Source: nothing.pdf, Page 9). Wrong page (Source: attention-is-all-you-need.pdf, Page 12).");
    expect(out).toContain('data-cite="0"');
    expect(out).toContain('data-cite="1"');
    expect(out).toContain("(Source: nothing.pdf, Page 9)");
    expect(out).toContain("(Source: attention-is-all-you-need.pdf, Page 12)");
    expect((out.match(/data-cite/g) ?? []).length).toBe(2);
  });

  it("findCitation matches by name with or without the .pdf suffix, case-insensitively", () => {
    expect(findCitation(citations, "Attention-Is-All-You-Need.PDF", 4)?.index).toBe(0);
    expect(findCitation(citations, "retrieval-augmented-generation", 3)?.index).toBe(1);
    expect(findCitation(citations, "retrieval-augmented-generation", 4)).toBeNull();
  });

  it("always makes progress, including on indented headings and stray fences", () => {
    expect(html("    # indented heading")).toBe("<p># indented heading</p>");
    expect(html("text\n    # not a heading\nmore")).toBe("<p>text # not a heading more</p>");
    expect(html("```\nunterminated")).toBe("<pre><code>unterminated</code></pre>");
    expect(html("   ## ok")).toBe("<h3>ok</h3>");
  });

  it("keeps mathematical identifiers and intraword underscores literal", () => {
    const out = html("Divide by \\( \\sqrt{d_k} \\) because \\( d_k \\) grows; see snake_case_name and other_name.");
    expect(out).not.toContain("<em>");
    expect(out).toContain('<code class="math">\\sqrt{d_k}</code>');
    expect(out).toContain('<code class="math">d_k</code>');
    expect(out).toContain("snake_case_name and other_name");
    expect(html("$5 and $10 are prices")).toBe("<p>$5 and $10 are prices</p>");
    expect(html("inline $x_i$ math")).toContain('<code class="math">x_i</code>');
    expect(html("_emphasis_ and *also* and **strong** and __strong__")).toBe("<p><em>emphasis</em> and <em>also</em> and <strong>strong</strong> and <strong>strong</strong></p>");
    expect(html("a * b * c")).toBe("<p>a * b * c</p>");
  });
});
