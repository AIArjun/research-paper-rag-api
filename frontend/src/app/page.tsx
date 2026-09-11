import { cookies } from "next/headers";
import { Entry } from "@/components/Entry";
import { Workspace } from "@/components/Workspace";
import { loadConfig } from "@/lib/server/env";
import { hasValidSessionCookie } from "@/lib/server/guard";
import { SESSION_COOKIE } from "@/lib/server/session";

export const dynamic = "force-dynamic";

export default async function Page() {
  const configured = loadConfig();
  if (!configured.ok) return <Entry configured={false} />;
  const jar = await cookies();
  const signedIn = hasValidSessionCookie(jar.get(SESSION_COOKIE)?.value, configured.config);
  return signedIn ? <Workspace /> : <Entry configured />;
}
