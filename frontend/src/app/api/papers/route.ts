import { apiError, errorResponse, jsonResponse } from "@/lib/server/errors";
import { guardRead } from "@/lib/server/guard";
import { callUpstream } from "@/lib/server/upstream";
import { parsePaperList } from "@/lib/server/validate";
import type { PapersResponse } from "@/lib/shared/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 30;

export async function GET(request: Request): Promise<Response> {
  const guarded = guardRead(request);
  if (!guarded.ok) return guarded.response;

  const result = await callUpstream(guarded.config, { route: "/papers", method: "GET", timeoutMs: 20_000 });
  if (!result.ok) return errorResponse(result.status, result.error);

  const papers = parsePaperList(result.body);
  if (!papers) return errorResponse(502, apiError("backend_error"));
  const body: PapersResponse = { papers };
  return jsonResponse(body);
}
