import { describe, expect, it, vi } from "vitest";
import { parseJsonObject, readBoundedBody, type BoundedBody } from "@/lib/server/body";

const encoder = new TextEncoder();
const decoder = new TextDecoder();

interface StreamHooks {
  pull?: () => void;
  cancel?: (reason?: unknown) => void;
}

/**
 * A byte stream that yields `chunks` one per read. `highWaterMark: 0` keeps the
 * stream machinery from pulling on its own, so every pull is attributable to
 * the code under test.
 */
function chunkStream(chunks: readonly Uint8Array[], hooks: StreamHooks = {}): ReadableStream<Uint8Array> {
  let index = 0;
  return new ReadableStream<Uint8Array>(
    {
      pull(controller) {
        hooks.pull?.();
        const next = chunks[index++];
        if (next === undefined) controller.close();
        else controller.enqueue(next);
      },
      cancel(reason) {
        hooks.cancel?.(reason);
      },
    },
    { highWaterMark: 0 },
  );
}

function postRequest(body: ReadableStream<Uint8Array> | null, headers: Record<string, string> = {}): Request {
  // Node's undici requires duplex: "half" for stream bodies; lib.dom's RequestInit does not declare it.
  const init: RequestInit & { duplex: "half" } = { method: "POST", body, headers, duplex: "half" };
  return new Request("http://bff.test/api/query", init);
}

function bytesOf(result: BoundedBody): Uint8Array {
  if (!result.ok) throw new Error(`expected an ok body, got ${result.reason}`);
  return result.bytes;
}

describe("readBoundedBody", () => {
  it("returns a body within the limit intact", async () => {
    const request = postRequest(chunkStream([encoder.encode("hello, "), encoder.encode("world")]));
    const bytes = bytesOf(await readBoundedBody(request, 64));
    expect(decoder.decode(bytes)).toBe("hello, world");
  });

  it("concatenates chunk boundaries correctly and accepts a body exactly at the limit", async () => {
    const request = postRequest(chunkStream([encoder.encode("ab"), encoder.encode("cd"), encoder.encode("e")]));
    const bytes = bytesOf(await readBoundedBody(request, 5));
    expect(bytes.byteLength).toBe(5);
    expect(decoder.decode(bytes)).toBe("abcde");
  });

  it("rejects a declared Content-Length over the limit without ever pulling the stream", async () => {
    const pull = vi.fn();
    const request = postRequest(chunkStream([new Uint8Array(1)], { pull }), { "content-length": "101" });
    await expect(readBoundedBody(request, 100)).resolves.toEqual({ ok: false, reason: "too_large" });
    expect(pull).not.toHaveBeenCalled();
    expect(request.bodyUsed).toBe(false);
  });

  it("accepts a declared Content-Length exactly at the limit", async () => {
    const request = postRequest(chunkStream([encoder.encode("12345")]), { "content-length": "5" });
    expect(decoder.decode(bytesOf(await readBoundedBody(request, 5)))).toBe("12345");
  });

  it("treats a non-numeric Content-Length as unreadable", async () => {
    const request = postRequest(chunkStream([new Uint8Array(1)]), { "content-length": "abc" });
    await expect(readBoundedBody(request, 100)).resolves.toEqual({ ok: false, reason: "unreadable" });
  });

  it("stops reading an undeclared stream once it exceeds the limit and cancels the reader", async () => {
    const pull = vi.fn();
    const cancel = vi.fn();
    const chunks = [new Uint8Array(60), new Uint8Array(60), new Uint8Array(60)];
    const request = postRequest(chunkStream(chunks, { pull, cancel }));
    await expect(readBoundedBody(request, 100)).resolves.toEqual({ ok: false, reason: "too_large" });
    expect(cancel).toHaveBeenCalledTimes(1);
    // 60 fits, 120 does not: the third chunk was never requested.
    expect(pull).toHaveBeenCalledTimes(2);
  });

  it("returns zero bytes for a request without a body", async () => {
    const bytes = bytesOf(await readBoundedBody(postRequest(null), 100));
    expect(bytes.byteLength).toBe(0);
  });

  it("returns zero bytes for a stream that closes immediately", async () => {
    const bytes = bytesOf(await readBoundedBody(postRequest(chunkStream([])), 100));
    expect(bytes.byteLength).toBe(0);
  });

  it("reports a stream that errors mid-read as unreadable", async () => {
    const broken = new ReadableStream<Uint8Array>(
      {
        pull() {
          throw new Error("connection reset");
        },
      },
      { highWaterMark: 0 },
    );
    await expect(readBoundedBody(postRequest(broken), 100)).resolves.toEqual({ ok: false, reason: "unreadable" });
  });
});

describe("parseJsonObject", () => {
  it("accepts a JSON object", () => {
    expect(parseJsonObject(encoder.encode('{"question":"hi","top_k":3}'))).toEqual({ question: "hi", top_k: 3 });
    expect(parseJsonObject(encoder.encode("{}"))).toEqual({});
  });

  it("rejects arrays and primitives", () => {
    for (const doc of ["[1,2]", '"text"', "42", "true", "null"]) {
      expect(parseJsonObject(encoder.encode(doc)), doc).toBeNull();
    }
  });

  it("rejects invalid JSON", () => {
    for (const doc of ["{", "", "{'a':1}", '{"a":1,}']) {
      expect(parseJsonObject(encoder.encode(doc)), JSON.stringify(doc)).toBeNull();
    }
  });

  it("rejects bytes that are not valid UTF-8", () => {
    // { 0xFF }
    expect(parseJsonObject(new Uint8Array([0x7b, 0xff, 0x7d]))).toBeNull();
    // {"a":"<lone continuation byte>"} - a lenient decoder would turn this into U+FFFD and parse fine
    expect(parseJsonObject(new Uint8Array([0x7b, 0x22, 0x61, 0x22, 0x3a, 0x22, 0x80, 0x22, 0x7d]))).toBeNull();
  });
});
