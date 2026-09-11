/**
 * Bounds shared by the browser UI and the server-side BFF routes.
 * The Render backend enforces its own (larger) ceilings; these are the
 * stricter limits this frontend applies before anything is forwarded.
 */

/** Hard PDF ceiling for this frontend: 4 MiB, below Vercel's 4.5 MB function payload limit. */
export const MAX_PDF_BYTES = 4 * 1024 * 1024;

/** Multipart framing allowance on top of the PDF ceiling. */
export const MULTIPART_ALLOWANCE_BYTES = 16 * 1024;

/** Largest upload request body the server will read before rejecting. */
export const MAX_UPLOAD_REQUEST_BYTES = MAX_PDF_BYTES + MULTIPART_ALLOWANCE_BYTES;

/** JSON body ceiling for /api/query (a 2000-character question fits in any escaping). */
export const MAX_QUERY_BODY_BYTES = 16 * 1024;

/** JSON body ceiling for /api/auth/login. */
export const MAX_LOGIN_BODY_BYTES = 4 * 1024;

export const QUESTION_MIN_CHARS = 3;
export const QUESTION_MAX_CHARS = 2000;

export const TOP_K_MIN = 1;
export const TOP_K_MAX = 5;
export const DEFAULT_TOP_K = 5;

/** Backend paper ids are SHA-256 hex digests; the backend caps the field at 128 characters. */
export const MAX_PAPER_ID_CHARS = 128;
export const PAPER_ID_PATTERN = /^[A-Za-z0-9_-]{1,128}$/;

export const MAX_FILENAME_CHARS = 255;

/** Session lifetime for the private demo cookie. */
export const SESSION_TTL_SECONDS = 12 * 60 * 60;

/** Human-readable size label used in the UI and error copy. */
export const MAX_PDF_LABEL = "4 MB";
