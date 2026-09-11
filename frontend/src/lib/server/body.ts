/** Bounded request-body reading: counts streamed bytes and stops past the ceiling. */
export type BoundedBody =
  | { ok: true; bytes: Uint8Array<ArrayBuffer> }
  | { ok: false; reason: "too_large" | "unreadable" };

export async function readBoundedBody(request: Request, maxBytes: number): Promise<BoundedBody> {
  const declared = request.headers.get("content-length");
  if (declared !== null) {
    const length = Number(declared);
    if (!Number.isFinite(length) || length < 0) return { ok: false, reason: "unreadable" };
    if (length > maxBytes) return { ok: false, reason: "too_large" };
  }
  const body = request.body;
  if (!body) return { ok: true, bytes: new Uint8Array(0) };
  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      total += value.byteLength;
      if (total > maxBytes) {
        await reader.cancel().catch(() => undefined);
        return { ok: false, reason: "too_large" };
      }
      chunks.push(value);
    }
  } catch {
    return { ok: false, reason: "unreadable" };
  }
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return { ok: true, bytes: out };
}

export function parseJsonObject(bytes: Uint8Array<ArrayBuffer>): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return null;
    return parsed as Record<string, unknown>;
  } catch {
    return null;
  }
}
