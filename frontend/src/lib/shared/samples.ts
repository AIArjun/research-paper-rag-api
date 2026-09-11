import manifest from "../../../public/samples/manifest.json";

export interface SamplePaper {
  file: string; // path under /samples/
  title: string;
  short: string;
  authors: string;
  source_url: string;
  sha256: string;
  pages: number;
  bytes: number;
  license_note: string;
}

export const SAMPLE_PAPERS: readonly SamplePaper[] = manifest.papers;

/** Look a backend paper id (a SHA-256 hex digest) up against the bundled samples. */
export function sampleByPaperId(paperId: string | null | undefined): SamplePaper | null {
  if (!paperId) return null;
  const id = paperId.toLowerCase();
  return SAMPLE_PAPERS.find((p) => p.sha256 === id) ?? null;
}

export function samplePublicPath(sample: SamplePaper): string {
  return `/samples/${sample.file}`;
}
