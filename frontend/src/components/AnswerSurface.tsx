"use client";

import { type ReactNode } from "react";
import { renderMarkdown } from "@/lib/client/markdown";
import { paperShortTitle } from "@/lib/client/sources";
import type { Pending } from "@/lib/client/useObservatory";
import type { ApiError, QueryResponse } from "@/lib/shared/types";
import { Notice } from "./Notice";
import { Pulse } from "./Pulse";

interface Props {
  pending: Pending;
  elapsedSeconds: number;
  result: QueryResponse | null;
  error: ApiError | null;
  modelName: string;
  papersInScope: number;
  onCitation: (index: number) => void;
  onAskAgain: () => void;
  onDismissError: () => void;
}

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

export function AnswerSurface({ pending, elapsedSeconds, result, error, modelName, papersInScope, onCitation, onAskAgain, onDismissError }: Props) {
  const querying = pending.kind === "querying";

  if (querying) {
    return (
      <section className="answer answer--pending" aria-live="polite" aria-busy="true">
        <div className="answer__pending">
          <Pulse label="Working" />
          <div>
            <p className="answer__pending-title">
              Retrieving passages{papersInScope > 0 ? ` from ${papersInScope} ${papersInScope === 1 ? "paper" : "papers"}` : ""}
              {modelName ? ` and generating with ${modelName}` : " and generating"}
            </p>
            <p className="answer__pending-question">“{pending.question}”</p>
            <p className="answer__pending-meta">{elapsedSeconds} s elapsed · up to one model call · no automatic retry</p>
          </div>
        </div>
      </section>
    );
  }

  const errorBlock: ReactNode = error ? (
    <Notice
      error={error}
      onDismiss={onDismissError}
      action={
        error.category === "busy" || error.category === "timeout" || error.category === "unreachable" || error.category === "generation_failed" ? (
          <button type="button" className="button button--small" onClick={onAskAgain}>
            Ask again
          </button>
        ) : null
      }
    />
  ) : null;

  if (!result) {
    return (
      <section className="answer answer--empty" aria-live="polite">
        {errorBlock}
        {!error && (
          <div className="answer__empty">
            <svg viewBox="0 0 120 120" aria-hidden="true" focusable="false" className="answer__lens">
              <circle cx="60" cy="60" r="44" />
              <circle cx="60" cy="60" r="28" strokeDasharray="3 7" />
              <circle cx="60" cy="16" r="3.5" fill="currentColor" stroke="none" />
              <circle cx="92" cy="80" r="2.5" fill="currentColor" stroke="none" />
            </svg>
            <p className="answer__empty-title">Answers appear here with the passages they cite.</p>
            <p className="answer__empty-text">Choose a scope in the library, write a question, and press Ask. Every point in the evidence view opens the physical page it came from.</p>
          </div>
        )}
      </section>
    );
  }

  const abstained = result.model_used === "not-invoked";
  const usage = result.model_usage;

  return (
    <section className="answer" aria-live="polite" aria-labelledby="answer-title">
      {errorBlock}
      <div className="answer__paper">
        <div className="answer__kicker">
          <span id="answer-title" className="answer__label">
            Answer
          </span>
          <span className="preview-tag">Research preview · Check the cited sources</span>
        </div>
        <p className="answer__question">{result.question}</p>

        {abstained ? (
          <div className="answer__abstain">
            <p className="answer__body">{result.answer}</p>
            <p className="answer__abstain-note">Nothing was retrieved for this scope, so no model call was made and nothing was charged to the allowance.</p>
          </div>
        ) : (
          <div className="answer__body">
            {renderMarkdown(result.answer, result.citations, (ref, key) => (
              <button
                key={key}
                type="button"
                className="cite"
                onClick={() => onCitation(ref.index)}
                aria-label={`Show evidence: ${paperShortTitle(ref.citation.paper_id, ref.citation.paper)}, page ${ref.citation.page ?? "unknown"}`}
              >
                {paperShortTitle(ref.citation.paper_id, ref.citation.paper)} · p. {ref.citation.page ?? "?"}
              </button>
            ))}
          </div>
        )}

        {!abstained && (
          <dl className="answer__meta">
            <div>
              <dt>Model</dt>
              <dd>{result.model_used}</dd>
            </div>
            <div>
              <dt>Retrieval</dt>
              <dd>{formatMs(result.retrieval_time_ms)}</dd>
            </div>
            <div>
              <dt>Generation</dt>
              <dd>{formatMs(result.generation_time_ms)}</dd>
            </div>
            {usage && (
              <div>
                <dt>Tokens</dt>
                <dd>
                  {usage.input_tokens !== null && usage.output_tokens !== null
                    ? `${usage.input_tokens.toLocaleString()} in · ${usage.output_tokens.toLocaleString()} out`
                    : `${usage.tokens_charged.toLocaleString()} charged`}
                  <span className="answer__meta-note"> ({usage.accounting})</span>
                </dd>
              </div>
            )}
            <div>
              <dt>Request</dt>
              <dd className="answer__meta-mono">{result.request_id}</dd>
            </div>
          </dl>
        )}
      </div>
    </section>
  );
}
