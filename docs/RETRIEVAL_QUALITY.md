# Retrieval quality

Real retrieval combines the first 20 Chroma semantic candidates with up to 20
BM25 keyword candidates, alternating keyword and semantic hits with duplicate
chunk IDs removed. This keeps exact-identifier passages represented even when
they are absent from the semantic candidates. Both lists
are restricted to the selected paper when one is supplied. At most five hits
are returned, as before. The API field `relevance_score` is an uncalibrated
ranking score, calculated as 1 / (60 + output rank); it is not a similarity,
probability or a measure of answer correctness.

The generator receives a bounded window around each hit on the same physical
PDF page. Each window shares the existing context budget, including source
headers and separators; the final prompt still passes through the existing
context and token limits. Citation previews remain the first 300 characters of
the original retrieved chunk. Open the cited PDF page for the surrounding text.

The lexical chunks and original extracted page text are process-local, published
only after indexing succeeds, and removed after deletion succeeds. Failed storage
mutations continue to block queries until cleanup. This does not add persistent
paper storage or separate user workspaces. The demo provider retains its original
keyword/template behavior.

This improves access to exact identifiers and nearby sentences. It does not
guarantee completeness, factual correctness, table extraction, or abstention.
No provider calls are required to rank or inspect evidence. Quality evidence and
live verification must be recorded separately.

## Evidence regression (12 September 2026)

`scripts/evaluate_retrieval.py` runs the actual engine against the two bundled
public PDFs, verifying their SHA-256 hashes. It compares the former five semantic
chunks with the new generation context. On Windows/Python 3.12 with the pinned
MiniLM model snapshot, all expected markers were present for 5/8 old contexts
and 8/8 new contexts. These are development cases, not a blind benchmark or an
answer-accuracy score. The cases cover attention scaling, sequence/token document
use, retriever/generator definitions, encoder layers, BLEU, training hardware,
retriever fine-tuning and the Wikipedia index. The model-definition check requires
the defining BERT/DPR/BART passages, not just isolated acronym mentions.

An absent-topic case still retrieves unrelated text. This change does not implement
a relevance threshold; the generation step must abstain when the evidence is
insufficient. The real-profile CI runs the same check without network or provider
calls, under the existing 2 GiB / 1 CPU resource limit.
