import { MAX_FILENAME_CHARS, MAX_PDF_BYTES } from "@/lib/shared/limits";

export type UploadCheck =
  | { ok: true; name: string; bytes: Uint8Array<ArrayBuffer> }
  | { ok: false; status: number; category: "invalid_pdf" | "file_too_large" | "invalid_request" };

/** Basename only, .pdf suffix, printable ASCII, bounded length. */
export function safePdfName(raw: string): string | null {
  const base = raw.split(/[\\/]/).pop()?.trim() ?? "";
  if (!base || base.length > MAX_FILENAME_CHARS) return null;
  if (!/^[\x20-\x7e]+$/.test(base) || !base.toLowerCase().endsWith(".pdf")) return null;
  if (base.startsWith(".")) return null;
  return base;
}

export function looksLikePdf(bytes: Uint8Array<ArrayBuffer>): boolean {
  const head = new TextDecoder("latin1").decode(bytes.subarray(0, 1024));
  return head.includes("%PDF-");
}

/** Parse an already-bounded multipart body and validate the single `file` part. */
export async function checkUploadForm(bytes: Uint8Array<ArrayBuffer>, contentType: string | null): Promise<UploadCheck> {
  if (!contentType || !contentType.toLowerCase().startsWith("multipart/form-data")) {
    return { ok: false, status: 400, category: "invalid_request" };
  }
  let form: FormData;
  try {
    form = await new Response(bytes, { headers: { "Content-Type": contentType } }).formData();
  } catch {
    return { ok: false, status: 400, category: "invalid_request" };
  }
  const file = form.get("file");
  if (!(file instanceof File)) return { ok: false, status: 400, category: "invalid_request" };
  if (file.size > MAX_PDF_BYTES) return { ok: false, status: 413, category: "file_too_large" };
  if (file.size === 0) return { ok: false, status: 400, category: "invalid_pdf" };
  // Multipart parsing reports an untyped part as application/octet-stream; the
  // %PDF- header check below is what actually vouches for the content.
  if (file.type && file.type !== "application/pdf" && file.type !== "application/octet-stream") {
    return { ok: false, status: 400, category: "invalid_pdf" };
  }
  const name = safePdfName(file.name);
  if (!name) return { ok: false, status: 400, category: "invalid_pdf" };
  const fileBytes = new Uint8Array(await file.arrayBuffer());
  if (!looksLikePdf(fileBytes)) return { ok: false, status: 400, category: "invalid_pdf" };
  return { ok: true, name, bytes: fileBytes };
}
