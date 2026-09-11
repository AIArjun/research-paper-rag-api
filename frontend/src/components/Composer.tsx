"use client";

import { useEffect, useId, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { promptsFor } from "@/lib/client/prompts";
import { paperTitle } from "@/lib/client/sources";
import { validQuestion } from "@/lib/client/useObservatory";
import { DEFAULT_TOP_K, QUESTION_MAX_CHARS, QUESTION_MIN_CHARS, TOP_K_MAX, TOP_K_MIN } from "@/lib/shared/limits";
import type { PaperInfo } from "@/lib/shared/types";

interface Props {
  papers: PaperInfo[] | null;
  selectedPaperId: string | null;
  busy: boolean;
  querying: boolean;
  corpusEmpty: boolean;
  onClearScope: () => void;
  onAsk: (question: string, topK: number) => void;
}

export function Composer({ papers, selectedPaperId, busy, querying, corpusEmpty, onClearScope, onAsk }: Props) {
  const textId = useId();
  const topKId = useId();
  const countId = useId();
  const textarea = useRef<HTMLTextAreaElement>(null);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(DEFAULT_TOP_K);

  const selected = selectedPaperId ? papers?.find((p) => p.paper_id === selectedPaperId) ?? null : null;
  const prompts = promptsFor(papers, selectedPaperId);
  const length = question.trim().length;
  const canAsk = !busy && !corpusEmpty && validQuestion(question);

  useEffect(() => {
    const el = textarea.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 320)}px`;
  }, [question]);

  function submit(event?: FormEvent) {
    event?.preventDefault();
    if (!canAsk) return;
    onAsk(question, topK);
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      submit();
    }
  }

  function fill(text: string) {
    setQuestion(text);
    textarea.current?.focus();
  }

  return (
    <form className="composer" onSubmit={submit} aria-labelledby="composer-title">
      <div className="composer__head">
        <h2 id="composer-title" className="composer__title">
          Ask the papers
        </h2>
        <div className="scope" aria-live="polite">
          <span className="scope__label">Scope</span>
          <span className="scope__value">{selected ? paperTitle(selected.paper_id, selected.filename) : `All papers${papers ? ` (${papers.filter((p) => p.status === "ready").length})` : ""}`}</span>
          {selected && (
            <button type="button" className="link" onClick={onClearScope}>
              Search all instead
            </button>
          )}
        </div>
      </div>

      <label htmlFor={textId} className="visually-hidden">
        Your question
      </label>
      <textarea
        id={textId}
        ref={textarea}
        className="composer__input"
        rows={3}
        placeholder={corpusEmpty ? "Add a paper first, then ask something a cited page can answer." : "Ask something a cited page can answer…"}
        value={question}
        maxLength={QUESTION_MAX_CHARS}
        onChange={(event) => setQuestion(event.target.value)}
        onKeyDown={onKeyDown}
        disabled={busy}
        aria-describedby={countId}
      />

      {prompts.length > 0 && (
        <div className="prompts" aria-label="Suggested questions">
          <span className="prompts__label">Try</span>
          {prompts.map((prompt) => (
            <button key={prompt.text} type="button" className="chip" onClick={() => fill(prompt.text)} disabled={busy}>
              {prompt.text}
            </button>
          ))}
        </div>
      )}

      <div className="composer__foot">
        <p id={countId} className={`composer__count${length > QUESTION_MAX_CHARS - 100 ? " composer__count--near" : ""}`}>
          {length} / {QUESTION_MAX_CHARS}
          {length > 0 && length < QUESTION_MIN_CHARS ? " · at least 3 characters" : ""}
        </p>
        <div className="composer__controls">
          <label className="topk" htmlFor={topKId}>
            <span>Passages</span>
            <select id={topKId} value={topK} onChange={(event) => setTopK(Number(event.target.value))} disabled={busy}>
              {Array.from({ length: TOP_K_MAX - TOP_K_MIN + 1 }, (_, i) => TOP_K_MIN + i).map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </label>
          <button type="submit" className="button button--gold composer__ask" disabled={!canAsk} aria-keyshortcuts="Control+Enter Meta+Enter">
            {querying ? "Asking…" : "Ask"}
          </button>
        </div>
      </div>
      <p className="composer__note">Up to one model call per question; nothing is sent until you press Ask (Ctrl/⌘ + Enter).</p>
    </form>
  );
}
