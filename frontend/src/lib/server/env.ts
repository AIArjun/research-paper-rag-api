/**
 * Server-only configuration. Values are read from process.env inside route
 * handlers and server components; none of them is ever sent to the browser.
 * Missing or malformed configuration fails closed (the caller answers 503).
 */

export interface ServerConfig {
  ragApiUrl: string; // origin + optional base path, no trailing slash
  ragApiToken: string;
  demoPasscode: string;
  sessionSecret: string;
  allowedOrigins: readonly string[]; // exact origins, e.g. https://demo.example
  isProduction: boolean;
}

export type ConfigResult =
  | { ok: true; config: ServerConfig }
  | { ok: false; problems: string[] };

export const MIN_PASSCODE_CHARS = 16;
export const MIN_SESSION_SECRET_CHARS = 32;

type Env = Record<string, string | undefined>;

function normalizeOrigin(value: string): string | null {
  try {
    const url = new URL(value.trim());
    if (url.protocol !== "https:" && url.protocol !== "http:") return null;
    if (url.pathname !== "/" || url.search || url.hash || url.username || url.password) return null;
    return url.origin;
  } catch {
    return null;
  }
}

/**
 * Trusted origins come from explicit configuration and Vercel's system
 * environment (set at deploy time), never from request headers such as Host
 * or X-Forwarded-Host.
 */
export function trustedOrigins(env: Env): { origins: string[]; problems: string[] } {
  const problems: string[] = [];
  const origins = new Set<string>();
  const configured = (env.APP_ORIGIN ?? "").split(",").map((s) => s.trim()).filter(Boolean);
  for (const entry of configured) {
    const origin = normalizeOrigin(entry);
    if (origin) origins.add(origin);
    else problems.push("APP_ORIGIN contains an entry that is not a plain origin (scheme://host[:port])");
  }
  for (const key of ["VERCEL_PROJECT_PRODUCTION_URL", "VERCEL_BRANCH_URL", "VERCEL_URL"] as const) {
    const host = env[key]?.trim();
    if (host) {
      const origin = normalizeOrigin(`https://${host}`);
      if (origin) origins.add(origin);
    }
  }
  if (env.NODE_ENV !== "production") {
    origins.add("http://localhost:3000");
    origins.add("http://127.0.0.1:3000");
  }
  return { origins: [...origins], problems };
}

function isPrintableAsciiNoSpace(value: string): boolean {
  return /^[\x21-\x7e]+$/.test(value);
}

export function readConfig(env: Env = process.env): ConfigResult {
  const problems: string[] = [];

  const rawUrl = (env.RAG_API_URL ?? "").trim().replace(/\/+$/, "");
  let ragApiUrl = "";
  try {
    const parsed = new URL(rawUrl);
    if (parsed.protocol !== "https:" && parsed.protocol !== "http:") throw new Error("scheme");
    if (parsed.search || parsed.hash || parsed.username || parsed.password) throw new Error("extras");
    ragApiUrl = rawUrl;
  } catch {
    problems.push("RAG_API_URL must be an absolute http(s) URL");
  }

  const ragApiToken = env.RAG_API_TOKEN ?? "";
  if (!ragApiToken || !isPrintableAsciiNoSpace(ragApiToken) || ragApiToken.length > 512) {
    problems.push("RAG_API_TOKEN must be set (printable ASCII, no whitespace)");
  }

  const demoPasscode = env.DEMO_PASSCODE ?? "";
  if (demoPasscode.length < MIN_PASSCODE_CHARS || demoPasscode.length > 512) {
    problems.push(`DEMO_PASSCODE must be ${MIN_PASSCODE_CHARS}-512 characters`);
  }

  const sessionSecret = env.SESSION_SECRET ?? "";
  if (sessionSecret.length < MIN_SESSION_SECRET_CHARS || sessionSecret.length > 1024) {
    problems.push(`SESSION_SECRET must be at least ${MIN_SESSION_SECRET_CHARS} characters`);
  }

  const { origins, problems: originProblems } = trustedOrigins(env);
  problems.push(...originProblems);
  if (origins.length === 0) {
    problems.push("No trusted origin: set APP_ORIGIN or deploy on Vercel");
  }

  if (problems.length > 0) return { ok: false, problems };
  return {
    ok: true,
    config: {
      ragApiUrl,
      ragApiToken,
      demoPasscode,
      sessionSecret,
      allowedOrigins: origins,
      isProduction: env.NODE_ENV === "production",
    },
  };
}

let warnedProblems: string | null = null;

/** Read configuration once per request, logging problems (names only, never values) once per process. */
export function loadConfig(): ConfigResult {
  const result = readConfig(process.env);
  if (!result.ok) {
    const key = result.problems.join("|");
    if (warnedProblems !== key) {
      warnedProblems = key;
      console.error("[observatory] configuration incomplete:", result.problems.join("; "));
    }
  }
  return result;
}
