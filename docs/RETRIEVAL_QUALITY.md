# Retrieval quality

Real retrieval combines the first 20 Chroma semantic candidates with up to 20
BM25 keyword candidates, using reciprocal rank fusion (constant 60). Both lists
are restricted to the selected paper when one is supplied. At most five hits
are returned, as before. The API field `relevance_score` is an uncalibrated
ranking score; it is not a probability or a measure of answer correctness.

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
live verification will be recorded after the candidate is evaluated.
