import { describe, expect, it } from "vitest";
import { isTrustedOrigin } from "@/lib/server/origin";

const ALLOWED: readonly string[] = ["https://observatory.example.com", "http://localhost:3000"];

describe("isTrustedOrigin", () => {
  describe("with an Origin header", () => {
    it("accepts an origin on the allowlist", () => {
      expect(isTrustedOrigin(new Headers({ origin: "https://observatory.example.com" }), ALLOWED)).toBe(true);
      expect(isTrustedOrigin(new Headers({ origin: "http://localhost:3000" }), ALLOWED)).toBe(true);
    });

    it("compares the normalized origin, so host case and the default port do not matter", () => {
      expect(isTrustedOrigin(new Headers({ origin: "HTTPS://Observatory.Example.COM:443" }), ALLOWED)).toBe(true);
    });

    it("rejects an origin that is not on the allowlist", () => {
      expect(isTrustedOrigin(new Headers({ origin: "https://attacker.example.net" }), ALLOWED)).toBe(false);
    });

    it("rejects an allowed host on a different port", () => {
      expect(isTrustedOrigin(new Headers({ origin: "http://localhost:3001" }), ALLOWED)).toBe(false);
      expect(isTrustedOrigin(new Headers({ origin: "https://observatory.example.com:8443" }), ALLOWED)).toBe(false);
    });

    it("rejects an allowed host on a different scheme", () => {
      expect(isTrustedOrigin(new Headers({ origin: "http://observatory.example.com" }), ALLOWED)).toBe(false);
    });

    it("rejects the opaque 'null' origin", () => {
      expect(isTrustedOrigin(new Headers({ origin: "null" }), ALLOWED)).toBe(false);
    });

    it("rejects a malformed origin", () => {
      expect(isTrustedOrigin(new Headers({ origin: "not a url" }), ALLOWED)).toBe(false);
      expect(isTrustedOrigin(new Headers({ origin: "observatory.example.com" }), ALLOWED)).toBe(false);
    });

    it("never consults Host or X-Forwarded-Host: a foreign Origin loses even when they match an allowed host", () => {
      const headers = new Headers({
        origin: "https://attacker.example.net",
        host: "observatory.example.com",
        "x-forwarded-host": "observatory.example.com",
        "sec-fetch-site": "same-origin",
      });
      expect(isTrustedOrigin(headers, ALLOWED)).toBe(false);
    });

    it("fails closed when the allowlist is empty", () => {
      expect(isTrustedOrigin(new Headers({ origin: "https://observatory.example.com" }), [])).toBe(false);
    });
  });

  describe("without an Origin header", () => {
    it("accepts a same-origin Sec-Fetch-Site", () => {
      expect(isTrustedOrigin(new Headers({ "sec-fetch-site": "same-origin" }), ALLOWED)).toBe(true);
    });

    it("rejects a cross-site Sec-Fetch-Site", () => {
      expect(isTrustedOrigin(new Headers({ "sec-fetch-site": "cross-site" }), ALLOWED)).toBe(false);
    });

    it("rejects a same-site (sibling subdomain) Sec-Fetch-Site", () => {
      expect(isTrustedOrigin(new Headers({ "sec-fetch-site": "same-site" }), ALLOWED)).toBe(false);
    });

    it("rejects a request with neither Origin nor Sec-Fetch-Site", () => {
      expect(isTrustedOrigin(new Headers(), ALLOWED)).toBe(false);
    });

    it("does not let Host or X-Forwarded-Host stand in for a missing Origin", () => {
      const headers = new Headers({
        host: "observatory.example.com",
        "x-forwarded-host": "observatory.example.com",
      });
      expect(isTrustedOrigin(headers, ALLOWED)).toBe(false);
    });
  });
});
