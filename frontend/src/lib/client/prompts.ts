import { SAMPLE_PAPERS } from "@/lib/shared/samples";
import type { PaperInfo } from "@/lib/shared/types";

export interface SuggestedPrompt {
  text: string;
  /** Digest of the sample the prompt is about; omitted for generic prompts. */
  sha256?: string;
}

const ATTENTION = SAMPLE_PAPERS[0]?.sha256;
const RAG = SAMPLE_PAPERS[1]?.sha256;

/**
 * Suggestions only fill the composer; nothing is sent until the visitor presses Ask.
 * Paper-specific prompts appear only while that paper is in the corpus.
 */
export const SUGGESTED_PROMPTS: readonly SuggestedPrompt[] = [
  { text: "Why does scaled dot-product attention divide by the square root of d_k?", sha256: ATTENTION },
  { text: "How many identical layers does the Transformer encoder stack use, and what are the two sub-layers in each?", sha256: ATTENTION },
  { text: "What distinguishes RAG-Sequence from RAG-Token in how retrieved documents are used?", sha256: RAG },
  { text: "What BLEU score did the big Transformer model reach on WMT 2014 English-to-German?", sha256: ATTENTION },
  { text: "Which knowledge-intensive tasks does the RAG paper evaluate on?", sha256: RAG },
  { text: "What is the main contribution of this paper, in three sentences?" },
];

export function promptsFor(papers: readonly PaperInfo[] | null, selectedPaperId: string | null): SuggestedPrompt[] {
  const present = new Set((papers ?? []).filter((p) => p.status === "ready").map((p) => p.paper_id));
  return SUGGESTED_PROMPTS.filter((prompt) => {
    if (!prompt.sha256) return true;
    if (selectedPaperId) return prompt.sha256 === selectedPaperId;
    return present.has(prompt.sha256);
  }).slice(0, 4);
}
