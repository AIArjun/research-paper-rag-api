import { sampleByPaperId, samplePublicPath, type SamplePaper } from "@/lib/shared/samples";
import type { Citation, PaperInfo } from "@/lib/shared/types";

/** A PDF held in this browser session only (object URL), keyed by the backend paper id. */
export interface LocalFile {
  url: string;
  name: string;
  bytes: number;
}

export type SourceKind = "sample" | "local";

export interface ResolvedSource {
  kind: SourceKind;
  url: string;
  title: string;
  filename: string;
  pages: number | null;
}

export type LocalFileMap = ReadonlyMap<string, LocalFile>;

/** Hex SHA-256 of a blob when the Web Crypto API is available (secure contexts only). */
export async function sha256Hex(bytes: ArrayBuffer): Promise<string | null> {
  if (typeof crypto === "undefined" || !crypto.subtle) return null;
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Display title for a paper: the sample title when the digest matches, else the stored filename. */
export function paperTitle(paperId: string | null | undefined, filename: string | null | undefined): string {
  const sample = sampleByPaperId(paperId);
  if (sample) return sample.title;
  return filename?.trim() || "Untitled paper";
}

export function paperShortTitle(paperId: string | null | undefined, filename: string | null | undefined): string {
  const sample = sampleByPaperId(paperId);
  if (sample) return sample.short;
  const name = filename?.trim() || "Untitled";
  return name.length > 28 ? `${name.slice(0, 26)}…` : name;
}

/** Find the PDF behind a citation: a bundled sample (same-origin) or a file added in this session. */
export function resolveSource(
  citation: Pick<Citation, "paper_id" | "paper">,
  papers: readonly PaperInfo[] | null,
  local: LocalFileMap,
): ResolvedSource | null {
  const sample = sampleByPaperId(citation.paper_id);
  if (sample) {
    return { kind: "sample", url: samplePublicPath(sample), title: sample.title, filename: sample.file, pages: sample.pages };
  }
  if (citation.paper_id) {
    const file = local.get(citation.paper_id);
    if (file) {
      const paper = papers?.find((p) => p.paper_id === citation.paper_id);
      return { kind: "local", url: file.url, title: paperTitle(citation.paper_id, file.name), filename: file.name, pages: paper?.pages ?? null };
    }
  }
  return null;
}

export function sampleInCorpus(sample: SamplePaper, papers: readonly PaperInfo[] | null): boolean {
  return Boolean(papers?.some((p) => p.paper_id === sample.sha256 && p.status === "ready"));
}

/** Page fragment understood by the built-in PDF viewers of Chromium and Firefox. */
export function pageUrl(url: string, page: number | null): string {
  return page && page > 0 ? `${url}#page=${page}` : url;
}

/** Same, for the embedded inspector: no thumbnail pane, fit the page width. */
export function embeddedPageUrl(url: string, page: number | null): string {
  const base = page && page > 0 ? `${url}#page=${page}&` : `${url}#`;
  return `${base}view=FitH&navpanes=0`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
