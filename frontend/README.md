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
npm test
npm run build
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

## Bounds and honesty

- PDF ceiling 4 MiB on both client and server (below Vercel's 4.5 MB function payload limit); the server counts streamed bytes before parsing the multipart body and checks the `%PDF-` header and `.pdf` name.
- Questions 3–2000 characters, `top_k` 1–5, `paper_id` restricted to the backend's id alphabet.
- Nothing queries the model automatically: not on load, login, paper selection, remount, or after an error. Only the explicit submit does, and the submit button is disabled while a request is pending.
- Answers are labelled "Research preview · Check the cited sources". Retrieval scores are shown as Chroma similarity, never as confidence. The demo's 20-call / 100 000-token allowance is the backend's, displayed from `/ready`, not enforced here.
- The corpus clears when the backend restarts; the bundled sample PDFs can be re-added through the same bounded upload flow.

## Sample PDFs

`public/samples/` holds the two public arXiv fixtures with a `manifest.json` (source URL, SHA-256, page count). They are served same-origin for page inspection and can be added to the corpus from the library panel. Backend paper ids are SHA-256 digests, so a citation's `paper_id` maps to a bundled file only when the digests match exactly. User-uploaded PDFs are kept as object URLs in the current browser session only, keyed by the same digest; if the bytes are not available, the UI asks for a re-upload rather than inventing a link.
