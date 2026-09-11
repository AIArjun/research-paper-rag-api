import { describe, expect, it } from "vitest";
import { checkUploadForm, looksLikePdf, safePdfName, type UploadCheck } from "@/lib/server/upload-check";
import { MAX_FILENAME_CHARS, MAX_PDF_BYTES } from "@/lib/shared/limits";

const encoder = new TextEncoder();
const PDF_BYTES = encoder.encode("%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF\n");

const INVALID_REQUEST = { ok: false, status: 400, category: "invalid_request" } as const;
const INVALID_PDF = { ok: false, status: 400, category: "invalid_pdf" } as const;
const FILE_TOO_LARGE = { ok: false, status: 413, category: "file_too_large" } as const;

/** Serialize a FormData the way a browser would: raw multipart bytes plus the boundary-bearing content type. */
async function encodeForm(form: FormData): Promise<{ bytes: Uint8Array<ArrayBuffer>; contentType: string }> {
  const response = new Response(form);
  const contentType = response.headers.get("content-type");
  if (!contentType) throw new Error("Response did not derive a multipart content type from the FormData");
  return { bytes: new Uint8Array(await response.arrayBuffer()), contentType };
}

async function checkFile(file: File): Promise<UploadCheck> {
  const form = new FormData();
  form.append("file", file);
  const { bytes, contentType } = await encodeForm(form);
  return checkUploadForm(bytes, contentType);
}

describe("safePdfName", () => {
  it("keeps only the basename, whichever separator was used", () => {
    expect(safePdfName("/tmp/uploads/paper.pdf")).toBe("paper.pdf");
    expect(safePdfName("C:\\Users\\me\\Documents\\paper.pdf")).toBe("paper.pdf");
    expect(safePdfName("../../etc/paper.pdf")).toBe("paper.pdf");
  });

  it("requires a .pdf extension, case-insensitively", () => {
    expect(safePdfName("Paper.PDF")).toBe("Paper.PDF");
    expect(safePdfName("paper.txt")).toBeNull();
    expect(safePdfName("paper.pdf.exe")).toBeNull();
    expect(safePdfName("paper")).toBeNull();
  });

  it("bounds the basename at MAX_FILENAME_CHARS", () => {
    const longest = "a".repeat(MAX_FILENAME_CHARS - ".pdf".length) + ".pdf";
    expect(longest).toHaveLength(MAX_FILENAME_CHARS);
    expect(safePdfName(longest)).toBe(longest);
    expect(safePdfName(`a${longest}`)).toBeNull();
  });

  it("rejects control characters and non-ASCII", () => {
    expect(safePdfName("pa\tper.pdf")).toBeNull();
    expect(safePdfName("paper\u0000.pdf")).toBeNull();
    expect(safePdfName("paper\u007f.pdf")).toBeNull();
    expect(safePdfName("résumé.pdf")).toBeNull();
  });

  it("rejects dotfiles and empty names", () => {
    expect(safePdfName(".hidden.pdf")).toBeNull();
    expect(safePdfName("/tmp/.hidden.pdf")).toBeNull();
    expect(safePdfName("")).toBeNull();
    expect(safePdfName("dir/")).toBeNull();
  });

  it("trims surrounding whitespace from the basename", () => {
    expect(safePdfName("  paper.pdf  ")).toBe("paper.pdf");
  });
});

describe("looksLikePdf", () => {
  it("recognizes a file that starts with the PDF header", () => {
    expect(looksLikePdf(encoder.encode("%PDF-1.7\n"))).toBe(true);
  });

  it("is binary-safe: high bytes before the header do not hide it", () => {
    const bytes = new Uint8Array([0xff, 0xfe, 0x00, 0xc3, ...encoder.encode("%PDF-1.4")]);
    expect(looksLikePdf(bytes)).toBe(true);
  });

  it("accepts a header that appears within the first 1024 bytes", () => {
    const bytes = new Uint8Array(1024).fill(0x20);
    bytes.set(encoder.encode("%PDF-1.5"), 1000);
    expect(looksLikePdf(bytes)).toBe(true);
  });

  it("ignores a header that only appears after the first 1024 bytes", () => {
    const bytes = new Uint8Array(2048).fill(0x20);
    bytes.set(encoder.encode("%PDF-1.5"), 1024);
    expect(looksLikePdf(bytes)).toBe(false);
  });

  it("rejects bytes without a PDF header", () => {
    expect(looksLikePdf(encoder.encode("hello"))).toBe(false);
    expect(looksLikePdf(new Uint8Array(0))).toBe(false);
  });
});

describe("checkUploadForm", () => {
  it("rejects a body whose content type is not multipart/form-data", async () => {
    expect(await checkUploadForm(encoder.encode('{"file":"x"}'), "application/json")).toEqual(INVALID_REQUEST);
    expect(await checkUploadForm(PDF_BYTES, null)).toEqual(INVALID_REQUEST);
  });

  it("rejects a multipart body that cannot be parsed", async () => {
    expect(await checkUploadForm(encoder.encode("garbage"), "multipart/form-data; boundary=nope")).toEqual(INVALID_REQUEST);
  });

  it("rejects a multipart body without a file part", async () => {
    const form = new FormData();
    form.append("note", "no file here");
    const { bytes, contentType } = await encodeForm(form);
    expect(await checkUploadForm(bytes, contentType)).toEqual(INVALID_REQUEST);
  });

  it("rejects a 'file' field that is a plain string rather than a file", async () => {
    const form = new FormData();
    form.append("file", "not a file");
    const { bytes, contentType } = await encodeForm(form);
    expect(await checkUploadForm(bytes, contentType)).toEqual(INVALID_REQUEST);
  });

  it("rejects an empty file as invalid_pdf", async () => {
    expect(await checkFile(new File([], "empty.pdf", { type: "application/pdf" }))).toEqual(INVALID_PDF);
  });

  it("accepts a file exactly at MAX_PDF_BYTES and rejects one byte more with 413 file_too_large", async () => {
    const atLimit = new Uint8Array(MAX_PDF_BYTES);
    atLimit.set(PDF_BYTES);
    const accepted = await checkFile(new File([atLimit], "limit.pdf", { type: "application/pdf" }));
    expect(accepted.ok).toBe(true);

    const oneOver = new Uint8Array(MAX_PDF_BYTES + 1);
    oneOver.set(PDF_BYTES);
    expect(await checkFile(new File([oneOver], "big.pdf", { type: "application/pdf" }))).toEqual(FILE_TOO_LARGE);
  });

  it("rejects a file whose declared type is not application/pdf", async () => {
    expect(await checkFile(new File([PDF_BYTES], "paper.pdf", { type: "text/plain" }))).toEqual(INVALID_PDF);
  });

  it("rejects a file whose name does not sanitize to a .pdf", async () => {
    expect(await checkFile(new File([PDF_BYTES], "notes.txt", { type: "application/pdf" }))).toEqual(INVALID_PDF);
    expect(await checkFile(new File([PDF_BYTES], ".hidden.pdf", { type: "application/pdf" }))).toEqual(INVALID_PDF);
  });

  it("rejects a file whose bytes carry no PDF header", async () => {
    const notPdf = new File([encoder.encode("hello, definitely not a pdf")], "paper.pdf", { type: "application/pdf" });
    expect(await checkFile(notPdf)).toEqual(INVALID_PDF);
  });

  it("accepts a small valid PDF and returns the sanitized name with the exact bytes", async () => {
    const result = await checkFile(new File([PDF_BYTES], "uploads/2026/My Paper.pdf", { type: "application/pdf" }));
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.name).toBe("My Paper.pdf");
    expect(result.bytes.byteLength).toBe(PDF_BYTES.byteLength);
    expect(result.bytes).toEqual(PDF_BYTES);
  });
});
