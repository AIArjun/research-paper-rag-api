"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { callApi } from "@/lib/client/api";

/** Placeholder shell; the full workspace lands in the next commits. */
export function Workspace() {
  const router = useRouter();
  const [pending, setPending] = useState(false);

  async function logout() {
    setPending(true);
    try {
      await callApi<void>("/api/auth/logout", { method: "POST" });
    } finally {
      router.refresh();
    }
  }

  return (
    <main className="deck">
      <header className="deck__bar">
        <p className="deck__wordmark">Research Observatory</p>
        <button className="button" type="button" onClick={logout} disabled={pending}>
          Leave
        </button>
      </header>
      <p className="deck__placeholder">Workspace under construction.</p>
    </main>
  );
}
