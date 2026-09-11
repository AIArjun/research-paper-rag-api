import { describe, it, expect, vi } from "vitest";
import {
  MIN_PASSCODE_CHARS,
  MIN_SESSION_SECRET_CHARS,
  loadConfig,
  readConfig,
  trustedOrigins,
} from "@/lib/server/env";

const SECRET = "test-secret-0123456789abcdef0123456789abcdef";
const TOKEN = "test-token-0123456789abcdef";
const PASSCODE = "test-passcode-0123456789abcdef";

const VALID: Record<string, string | undefined> = {
  RAG_API_URL: "https://api.example.test",
  RAG_API_TOKEN: TOKEN,
  DEMO_PASSCODE: PASSCODE,
  SESSION_SECRET: SECRET,
  APP_ORIGIN: "https://demo.example.test",
  NODE_ENV: "test",
};

const REQUIRED = ["RAG_API_URL", "RAG_API_TOKEN", "DEMO_PASSCODE", "SESSION_SECRET"] as const;

function problemsOf(env: Record<string, string | undefined>): string[] {
  const result = readConfig(env);
  if (result.ok) throw new Error("expected readConfig to fail closed");
  return result.problems;
}

function configOf(env: Record<string, string | undefined>) {
  const result = readConfig(env);
  if (!result.ok) throw new Error(`expected readConfig to succeed, got: ${result.problems.join("; ")}`);
  return result.config;
}

describe("readConfig", () => {
  it("accepts a complete configuration and exposes it verbatim", () => {
    const config = configOf(VALID);
    expect(config.ragApiUrl).toBe("https://api.example.test");
    expect(config.ragApiToken).toBe(TOKEN);
    expect(config.demoPasscode).toBe(PASSCODE);
    expect(config.sessionSecret).toBe(SECRET);
    expect(config.isProduction).toBe(false);
    expect(config.allowedOrigins).toContain("https://demo.example.test");
  });

  it.each(REQUIRED)("fails closed and names the variable when %s is missing", (name) => {
    const problems = problemsOf({ ...VALID, [name]: undefined });
    expect(problems).toHaveLength(1);
    expect(problems[0]).toContain(name);
  });

  it("lists every missing variable at once", () => {
    const problems = problemsOf({ NODE_ENV: "test", APP_ORIGIN: VALID.APP_ORIGIN });
    for (const name of REQUIRED) {
      expect(problems.some((p) => p.includes(name))).toBe(true);
    }
  });

  it("rejects a passcode or session secret that is too short", () => {
    expect(MIN_PASSCODE_CHARS).toBe(16);
    expect(MIN_SESSION_SECRET_CHARS).toBe(32);

    const shortPass = problemsOf({ ...VALID, DEMO_PASSCODE: "p".repeat(MIN_PASSCODE_CHARS - 1) });
    expect(shortPass).toHaveLength(1);
    expect(shortPass[0]).toContain("DEMO_PASSCODE");
    expect(readConfig({ ...VALID, DEMO_PASSCODE: "p".repeat(MIN_PASSCODE_CHARS) }).ok).toBe(true);

    const shortSecret = problemsOf({ ...VALID, SESSION_SECRET: "s".repeat(MIN_SESSION_SECRET_CHARS - 1) });
    expect(shortSecret).toHaveLength(1);
    expect(shortSecret[0]).toContain("SESSION_SECRET");
    expect(readConfig({ ...VALID, SESSION_SECRET: "s".repeat(MIN_SESSION_SECRET_CHARS) }).ok).toBe(true);
  });

  it("strips trailing slashes from RAG_API_URL but keeps a base path", () => {
    expect(configOf({ ...VALID, RAG_API_URL: "https://api.example.test/" }).ragApiUrl).toBe("https://api.example.test");
    expect(configOf({ ...VALID, RAG_API_URL: "https://api.example.test/v1///" }).ragApiUrl).toBe(
      "https://api.example.test/v1",
    );
    expect(configOf({ ...VALID, RAG_API_URL: "  http://localhost:8000/  " }).ragApiUrl).toBe("http://localhost:8000");
  });

  it.each([
    ["non-http scheme", "ftp://api.example.test"],
    ["javascript scheme", "javascript:alert(1)"],
    ["relative path", "/api"],
    ["not a URL", "api.example.test"],
    ["query string", "https://api.example.test/?debug=1"],
    ["embedded credentials", "https://user:pass@api.example.test"],
  ])("rejects RAG_API_URL that is %s", (_label, url) => {
    const problems = problemsOf({ ...VALID, RAG_API_URL: url });
    expect(problems).toHaveLength(1);
    expect(problems[0]).toContain("RAG_API_URL");
  });

  it.each([
    ["an inner space", "abc def"],
    ["a trailing newline", "abcdef\n"],
    ["a leading space", " abcdef"],
    ["a tab", "abc\tdef"],
    ["non-ASCII", "abcédef"],
  ])("rejects RAG_API_TOKEN containing %s", (_label, token) => {
    const problems = problemsOf({ ...VALID, RAG_API_TOKEN: token });
    expect(problems).toHaveLength(1);
    expect(problems[0]).toContain("RAG_API_TOKEN");
  });

  it("never echoes environment values in the problem list, only variable names", () => {
    const leakyToken = "leak-token-with space-0123456789";
    const leakyPass = "leak-passcode";
    const leakySecret = "leak-session-secret-too-short";
    const leakyUrl = "ftp://leak-url.example.test";
    const leakyOrigin = "https://leak-origin.example.test/path";
    const joined = problemsOf({
      NODE_ENV: "production",
      RAG_API_URL: leakyUrl,
      RAG_API_TOKEN: leakyToken,
      DEMO_PASSCODE: leakyPass,
      SESSION_SECRET: leakySecret,
      APP_ORIGIN: leakyOrigin,
    }).join("\n");
    expect(joined).not.toContain(leakyToken);
    expect(joined).not.toContain("leak-token");
    expect(joined).not.toContain(leakyPass);
    expect(joined).not.toContain(leakySecret);
    expect(joined).not.toContain("leak-session");
    expect(joined).not.toContain("leak-url");
    expect(joined).not.toContain("leak-origin");
    expect(joined).not.toContain("leak");
    for (const name of [...REQUIRED, "APP_ORIGIN"]) expect(joined).toContain(name);
  });
});

describe("trustedOrigins", () => {
  it("normalizes an APP_ORIGIN comma list to exact origins", () => {
    const { origins, problems } = trustedOrigins({
      APP_ORIGIN: " https://a.example.test , HTTPS://B.Example.test:8443/ ,, http://c.example.test:80",
      NODE_ENV: "production",
    });
    expect(problems).toEqual([]);
    expect(origins).toEqual(["https://a.example.test", "https://b.example.test:8443", "http://c.example.test"]);
  });

  it.each([
    ["a path", "https://a.example.test/app"],
    ["a query string", "https://a.example.test/?x=1"],
    ["a fragment", "https://a.example.test/#top"],
    ["credentials", "https://user:pw@a.example.test"],
    ["a bare host", "a.example.test"],
    ["a non-http scheme", "wss://a.example.test"],
  ])("rejects an APP_ORIGIN entry with %s and lists a problem that fails readConfig", (_label, entry) => {
    const { origins, problems } = trustedOrigins({ APP_ORIGIN: entry, NODE_ENV: "production" });
    expect(origins).toEqual([]);
    expect(problems).toHaveLength(1);
    expect(problems[0]).toContain("APP_ORIGIN");

    const result = readConfig({ ...VALID, APP_ORIGIN: entry });
    expect(result.ok).toBe(false);
    if (result.ok) throw new Error("unreachable");
    expect(result.problems.some((p) => p.includes("APP_ORIGIN"))).toBe(true);
  });

  it("keeps the valid entries of a mixed list while reporting the bad one", () => {
    const { origins, problems } = trustedOrigins({
      APP_ORIGIN: "https://ok.example.test,https://bad.example.test/path",
      NODE_ENV: "production",
    });
    expect(origins).toEqual(["https://ok.example.test"]);
    expect(problems).toHaveLength(1);
  });

  it("turns Vercel system hosts into https origins", () => {
    const { origins, problems } = trustedOrigins({
      NODE_ENV: "production",
      VERCEL_URL: "my-app-git-abc123.vercel.app",
      VERCEL_BRANCH_URL: "my-app-git-feature.vercel.app",
      VERCEL_PROJECT_PRODUCTION_URL: "my-app.vercel.app",
    });
    expect(problems).toEqual([]);
    expect(origins).toEqual(
      expect.arrayContaining([
        "https://my-app-git-abc123.vercel.app",
        "https://my-app-git-feature.vercel.app",
        "https://my-app.vercel.app",
      ]),
    );
    expect(origins).toHaveLength(3);
  });

  it("de-duplicates the same origin arriving from several sources", () => {
    const { origins } = trustedOrigins({
      NODE_ENV: "production",
      APP_ORIGIN: "https://my-app.vercel.app",
      VERCEL_URL: "my-app.vercel.app",
      VERCEL_PROJECT_PRODUCTION_URL: "my-app.vercel.app",
    });
    expect(origins).toEqual(["https://my-app.vercel.app"]);
  });

  it.each([
    ["development", "development"],
    ["test", "test"],
    ["unset", undefined],
  ])("includes the localhost origins when NODE_ENV is %s", (_label, nodeEnv) => {
    const { origins } = trustedOrigins({ NODE_ENV: nodeEnv });
    expect(origins).toEqual(expect.arrayContaining(["http://localhost:3000", "http://127.0.0.1:3000"]));
  });

  it("omits the localhost origins in production", () => {
    const { origins } = trustedOrigins({ NODE_ENV: "production", APP_ORIGIN: "https://demo.example.test" });
    expect(origins).toEqual(["https://demo.example.test"]);
    expect(configOf({ ...VALID, NODE_ENV: "production" }).allowedOrigins).toEqual(["https://demo.example.test"]);
  });

  it("fails readConfig in production when neither APP_ORIGIN nor a Vercel URL is present", () => {
    expect(trustedOrigins({ NODE_ENV: "production" }).origins).toEqual([]);
    const problems = problemsOf({ ...VALID, NODE_ENV: "production", APP_ORIGIN: undefined });
    expect(problems).toHaveLength(1);
    expect(problems[0]).toMatch(/no trusted origin/i);
    expect(problems[0]).toContain("APP_ORIGIN");
  });

  it("never consults Host-style environment values", () => {
    const hostile = {
      NODE_ENV: "production",
      HOST: "evil.example",
      HOSTNAME: "evil.example",
      X_FORWARDED_HOST: "evil.example",
      HTTP_HOST: "evil.example",
      HTTP_X_FORWARDED_HOST: "evil.example",
      NEXT_PUBLIC_VERCEL_URL: "evil.example",
    };
    expect(trustedOrigins(hostile).origins).toEqual([]);

    const { origins } = trustedOrigins({ ...hostile, VERCEL_URL: "my-app.vercel.app" });
    expect(origins).toEqual(["https://my-app.vercel.app"]);
    expect(origins.join(" ")).not.toContain("evil.example");

    const config = configOf({ ...VALID, ...hostile, NODE_ENV: "test" });
    expect(config.allowedOrigins.join(" ")).not.toContain("evil.example");
  });
});

describe("loadConfig", () => {
  it("reads process.env, logs variable names only, and does not repeat an identical warning", () => {
    const leakySecret = "leak-session-secret-too-short";
    const leakyToken = "leak token with space";
    vi.stubEnv("NODE_ENV", "test");
    vi.stubEnv("RAG_API_URL", VALID.RAG_API_URL);
    vi.stubEnv("RAG_API_TOKEN", leakyToken);
    vi.stubEnv("DEMO_PASSCODE", PASSCODE);
    vi.stubEnv("SESSION_SECRET", leakySecret);
    vi.stubEnv("APP_ORIGIN", undefined);
    for (const key of ["VERCEL_URL", "VERCEL_BRANCH_URL", "VERCEL_PROJECT_PRODUCTION_URL"]) vi.stubEnv(key, undefined);
    const error = vi.spyOn(console, "error").mockImplementation(() => {});

    const first = loadConfig();
    expect(first.ok).toBe(false);
    if (first.ok) throw new Error("unreachable");
    expect(first.problems.some((p) => p.includes("RAG_API_TOKEN"))).toBe(true);
    expect(first.problems.some((p) => p.includes("SESSION_SECRET"))).toBe(true);
    expect(error).toHaveBeenCalledTimes(1);
    const logged = error.mock.calls.map((call) => call.map(String).join(" ")).join("\n");
    expect(logged).toContain("RAG_API_TOKEN");
    expect(logged).toContain("SESSION_SECRET");
    expect(logged).not.toContain(leakySecret);
    expect(logged).not.toContain(leakyToken);

    loadConfig();
    expect(error).toHaveBeenCalledTimes(1);

    vi.stubEnv("SESSION_SECRET", SECRET);
    const second = loadConfig();
    expect(second.ok).toBe(false);
    expect(error).toHaveBeenCalledTimes(2);

    vi.stubEnv("RAG_API_TOKEN", TOKEN);
    expect(loadConfig().ok).toBe(true);
    expect(error).toHaveBeenCalledTimes(2);
  });
});
