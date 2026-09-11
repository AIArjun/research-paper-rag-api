"use client";

import { useRouter } from "next/navigation";
import { useCallback, useRef, useState } from "react";
import { useElapsed } from "@/lib/client/useElapsed";
import { useObservatory } from "@/lib/client/useObservatory";
import { AnswerSurface } from "./AnswerSurface";
import { Composer } from "./Composer";
import { DeckBar } from "./DeckBar";
import { Evidence } from "./Evidence";
import { Library } from "./Library";

export function Workspace() {
  const router = useRouter();
  const [leaving, setLeaving] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const libraryRef = useRef<HTMLElement>(null);

  const onLoggedOut = useCallback(() => {
    router.refresh();
  }, [router]);

  const { state, actions, localFiles } = useObservatory(onLoggedOut);
  const busy = state.pending.kind !== "idle";
  const pendingStart = state.pending.kind === "idle" ? null : state.pending.startedAt;
  const elapsed = useElapsed(pendingStart);

  const readyPapers = (state.papers ?? []).filter((p) => p.status === "ready");
  const corpusEmpty = state.papers !== null && readyPapers.length === 0;
  const papersInScope = state.selectedPaperId ? 1 : readyPapers.length;

  const focusUpload = useCallback(() => {
    setLibraryOpen(true);
    window.requestAnimationFrame(() => {
      libraryRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      libraryRef.current?.querySelector<HTMLElement>(".dropzone__browse")?.focus();
    });
  }, []);

  async function leave() {
    setLeaving(true);
    await actions.logout();
  }

  return (
    <div className="observatory">
      <DeckBar status={state.status} statusKnown={state.statusChecked} onLogout={leave} logoutPending={leaving} />

      <div className="observatory__grid">
        <aside ref={libraryRef} className={`observatory__library${libraryOpen ? " observatory__library--open" : ""}`} aria-label="Paper library">
          <button
            type="button"
            className="library-toggle"
            aria-expanded={libraryOpen}
            aria-controls="library-panel"
            onClick={() => setLibraryOpen((open) => !open)}
          >
            <span>Library</span>
            <span className="library-toggle__meta">
              {state.papers === null ? "…" : `${readyPapers.length} ${readyPapers.length === 1 ? "paper" : "papers"}`}
            </span>
          </button>
          <div id="library-panel" className="observatory__library-panel">
            <Library
              papers={state.papers}
              loading={state.papersLoading}
              error={state.papersError}
              selectedPaperId={state.selectedPaperId}
              localFiles={localFiles}
              busy={busy}
              uploadingName={state.pending.kind === "uploading" ? state.pending.name : null}
              uploadElapsed={state.pending.kind === "uploading" ? elapsed : 0}
              uploadError={state.uploadError}
              uploadNotice={state.uploadNotice}
              onSelect={actions.selectPaper}
              onRefresh={() => void actions.refreshPapers()}
              onFile={(file) => void actions.upload(file)}
              onAddSample={(sample) => void actions.addSample(sample)}
              onDismissUpload={actions.dismissUpload}
            />
          </div>
        </aside>

        <main className="observatory__deck" id="main">
          <Composer
            papers={state.papers}
            selectedPaperId={state.selectedPaperId}
            busy={busy}
            querying={state.pending.kind === "querying"}
            corpusEmpty={corpusEmpty}
            onClearScope={() => actions.selectPaper(null)}
            onAsk={(question, topK) => void actions.ask(question, topK)}
          />

          <AnswerSurface
            pending={state.pending}
            elapsedSeconds={state.pending.kind === "querying" ? elapsed : 0}
            result={state.result}
            error={state.queryError}
            modelName={state.status?.configured_model ?? ""}
            papersInScope={papersInScope}
            onCitation={actions.selectCitation}
            onAskAgain={() => void actions.askAgain()}
            onDismissError={actions.dismissQuery}
          />

          {state.result && state.result.citations.length > 0 && state.pending.kind !== "querying" && (
            <Evidence
              key={state.result.request_id}
              citations={state.result.citations}
              papersSearched={state.result.papers_searched}
              selected={state.selectedCitation}
              papers={state.papers}
              localFiles={localFiles}
              onSelect={actions.selectCitation}
              onRequestUpload={focusUpload}
            />
          )}
        </main>
      </div>
    </div>
  );
}
