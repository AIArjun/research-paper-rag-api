# Research Observatory (frontend)

A private-demo workspace for the Research Paper RAG API, by Arjunworks. Next.js App Router + TypeScript, plain CSS, no client-side secrets.

Requires Node.js 22.22.2+ or 24.15.0+ (the jsdom test environment sets that floor); Vercel's Node 22.x runtime and the CI's Node 22 satisfy it.

- **Entry**: a passcode gate. A successful passcode sets a signed, HttpOnly, `SameSite=Lax` session cookie (12 h).
- **Workspace**: paper library and selection, bounded PDF upload (up to 4 MB), question composer, real answers with citations, a source constellation tied to the returned citations, and physical-page inspection of the cited PDF page.
- **BFF**: every backend call goes through server-side route handlers under `src/app/api/`. The browser never sees the Render bearer token. Routes are allowlisted (`/ready`, `/papers`, `/papers/upload`, `/query`), bodies are bounded, responses are `no-store`, and upstream error text is replaced by categorized messages.

## Run locally

```sh
cd frontend
npm ci
cp .env.example .env.local   # fill in values; nothing here is committed
npm run dev                  # http://localhost:3000
```

Checks:

```sh
npm run typecheck
npm test          # vitest: 16 files, server + client suites, no network
npm run build
```

### Developing without the Render backend

`scripts/fake-backend.mjs` is a dev-only stand-in that speaks the documented API contract (routes, status codes, error categories, response fields) and never calls a model. Its answers are labelled as fake in the text. It is not part of the deployed app.

```sh
node scripts/fake-backend.mjs                      # 127.0.0.1:8765; FAKE_MODE=normal|busy|budget|down|slow|empty
RAG_API_URL=http://127.0.0.1:8765 RAG_API_TOKEN=fake-token-for-local-development-0123456789 \
DEMO_PASSCODE=local-dev-passcode-abcdefghijklmnop SESSION_SECRET=local-dev-session-secret-0123456789abcdef \
npm run dev
```

## Environment (server-only)

| Variable | Purpose |
|---|---|
| `RAG_API_URL` | Base URL of the protected API, e.g. `https://research-paper-rag-api.onrender.com` |
| `RAG_API_TOKEN` | The API's shared bearer token (Render `DEMO_ACCESS_TOKEN`) |
| `DEMO_PASSCODE` | Passcode visitors type on the entry screen (24+ random characters recommended, 16 minimum) |
| `SESSION_SECRET` | HMAC key for the session cookie (32+ characters) |
| `APP_ORIGIN` | Optional comma-separated list of exact origins allowed to make state-changing requests. Vercel system URLs (`VERCEL_URL`, `VERCEL_BRANCH_URL`, `VERCEL_PROJECT_PRODUCTION_URL`) are trusted automatically; `http://localhost:3000` is trusted outside production. |

Missing or malformed configuration fails closed: the entry screen explains that the operator must configure the server, and every API route answers `503 not_configured`. In production at least one trusted origin must exist (Vercel provides them; set `APP_ORIGIN` for a custom domain).

## Vercel

- Root Directory: `frontend`
- Framework: Next.js (auto-detected); Build `npm run build`; Install `npm ci`
- Node.js: 22.x
- Environment variables: the four required names above (plus `APP_ORIGIN` for a custom domain), Production and Preview.

Route `maxDuration` is 120 s for uploads and 90 s for queries; upstream fetches time out earlier (100 s / 75 s) and are never retried.

## What the tests cover

| Area | Files |
|---|---|
| Session cookie signing, expiry, tampering; constant-time passcode; capped login throttle; fail-closed config that never logs values and never trusts Host headers | `src/lib/server/{session,auth,env}.test.ts` |
| Origin validation, streamed body bounds (over-limit bodies are never read past the ceiling), multipart/PDF checks, question/top_k/paper_id validation | `src/lib/server/{origin,body,upload-check,query-validation}.test.ts` |
| Upstream client: bearer only on protected routes, exactly one fetch per call (no retry), timeout/network/invalid-JSON mapping, sanitized error categories with upstream text never forwarded, payload validation | `src/lib/server/{upstream,errors,validate}.test.ts` |
| Route handlers end to end with a fake fetch: 401 before any body byte is read or any upstream call, 403 on foreign origin, 422 on bad input, throttle, cookie attributes | `src/app/api/routes.test.ts` |
| Workspace never sends `/api/query` on mount, selection, prompt fill, remount or error; exactly one request per Ask with the selected scope; duplicate submits blocked; safe rendering; categorized errors without automatic retry; abstention shown as no model call | `src/components/Workspace.test.tsx` |
| Markdown renderer emits React elements only, guarantees progress, keeps math and snake_case literal, chips only for returned citations; source resolution by digest; budget exhaustion derived from every allowance; bundled sample checksums | `src/lib/client/*.test.ts(x)`, `src/lib/shared/samples.test.ts` |

## Bounds and honesty

- PDF ceiling 4 MiB on both client and server (below Vercel's 4.5 MB function payload limit); the server counts streamed bytes before parsing the multipart body and checks the `%PDF-` header and `.pdf` name.
- Questions 3–2000 characters, `top_k` 1–5, `paper_id` restricted to the backend's id alphabet.
- Nothing queries the model automatically: not on load, login, paper selection, remount, or after an error. Only the explicit submit does, and the submit button is disabled while a request is pending.
- Answers are labelled "Research preview · Check the cited sources". Retrieval scores are shown as Chroma similarity, never as confidence. The demo's 20-call / 100 000-token allowance (per day and lifetime) is the backend's; the status pill shows today's and lifetime call counts from `/ready` and reports "allowance used up" when any of the four allowances is met. Nothing is enforced here; Render's ledger is the backstop.
- Known backend limitations are preserved, not hidden: one of the rubric questions (which models the RAG paper uses as retriever and generator) does not retrieve its defining passages, and the 400-token output cap can truncate an answer.
- The corpus clears when the backend restarts; the bundled sample PDFs can be re-added through the same bounded upload flow.

## Not included

No Content-Security-Policy header yet (Next.js inline scripts would need a nonce pipeline); the response headers do set `X-Frame-Options`, `nosniff` and a strict referrer policy. No per-user storage, no analytics, no persistent uploads: this is a shared public-paper demo whose corpus clears on backend restart.

## Sample PDFs

`public/samples/` holds the two public arXiv fixtures with a `manifest.json` (source URL, SHA-256, page count). They are served same-origin for page inspection and can be added to the corpus from the library panel. Backend paper ids are SHA-256 digests, so a citation's `paper_id` maps to a bundled file only when the digests match exactly. User-uploaded PDFs are kept as object URLs in the current browser session only, keyed by the same digest; if the bytes are not available, the UI asks for a re-upload rather than inventing a link.
