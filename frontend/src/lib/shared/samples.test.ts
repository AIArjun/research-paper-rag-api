import { createHash } from "node:crypto";
import { readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { SAMPLE_PAPERS } from "./samples";

const dir = join(process.cwd(), "public", "samples");

describe("bundled sample fixtures", () => {
  it("match the manifest's SHA-256, size and count", () => {
    expect(SAMPLE_PAPERS).toHaveLength(2);
    for (const sample of SAMPLE_PAPERS) {
      const path = join(dir, sample.file);
      expect(statSync(path).size).toBe(sample.bytes);
      const digest = createHash("sha256").update(readFileSync(path)).digest("hex");
      expect(digest).toBe(sample.sha256);
      expect(sample.source_url.startsWith("https://arxiv.org/pdf/")).toBe(true);
      expect(sample.pages).toBeGreaterThan(0);
    }
  });
});
