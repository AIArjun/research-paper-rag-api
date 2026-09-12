"use client";

import { useEffect, useRef } from "react";
import { paperTitle, resolveSource, type LocalFileMap } from "@/lib/client/sources";
import type { Citation, PaperInfo } from "@/lib/shared/types";
import { Constellation, lanesFor } from "./Constellation";
import { PageInspector } from "./PageInspector";

interface Props {
  citations: Citation[];
  papersSearched: number;
  selected: number | null;
  papers: PaperInfo[] | null;
  localFiles: LocalFileMap;
  onSelect: (index: number | null) => void;
  onRequestUpload: () => void;
}

export function Evidence({ citations, papersSearched, selected, papers, localFiles, onSelect, onRequestUpload }: Props) {
  const cardRefs = useRef<Array<HTMLLIElement | null>>([]);
  const lanes = lanesFor(citations);

  useEffect(() => {
    if (selected === null) return;
    cardRefs.current[selected]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selected]);

  return (
    <section className="evidence" aria-labelledby="evidence-title">
      <div className="evidence__head">
        <h2 id="evidence-title" className="evidence__title">
          Evidence
        </h2>
        <p className="evidence__summary">
          {citations.length} {citations.length === 1 ? "passage" : "passages"} from {papersSearched} {papersSearched === 1 ? "paper" : "papers"}
        </p>
      </div>

      <div className="evidence__grid">
        <div className="evidence__map">
          <Constellation citations={citations} selected={selected} onSelect={(i) => onSelect(i === selected ? null : i)} />
        </div>

        <ol className="sources evidence__sources" aria-label="Retrieved passages">
          {citations.map((c, i) => {
            const lane = lanes.find((l) => l.key === (c.paper_id ?? c.paper));
            const source = resolveSource(c, papers, localFiles);
            const isSelected = selected === i;
            return (
              <li
                key={`${c.chunk_id ?? i}`}
                ref={(el) => {
                  cardRefs.current[i] = el;
                }}
                className={`source source--${lane?.hue ?? "glass"}${isSelected ? " source--selected" : ""}`}
              >
                <button type="button" className="source__select" aria-pressed={isSelected} onClick={() => onSelect(isSelected ? null : i)}>
                  <span className="source__rank" aria-hidden="true">
                    {i + 1}
                  </span>
                  <span className="source__body">
                    <span className="source__paper">{paperTitle(c.paper_id, c.paper)}</span>
                    <span className="source__where">
                      {c.page !== null ? `Page ${c.page}` : "Page unknown"} · retrieval score {c.relevance_score.toFixed(3)}
                      <span className="source__uncal"> (uncalibrated)</span>
                    </span>
                    <span className="source__text">{c.text}</span>
                    <span className="source__status">
                      {isSelected ? "Shown in the page inspector" : source ? "Select to open the page" : "Select for the passage; PDF not held here"}
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ol>

        <div className="evidence__inspector">
          <PageInspector
            citation={selected === null ? null : citations[selected] ?? null}
            index={selected}
            papers={papers}
            localFiles={localFiles}
            onRequestUpload={onRequestUpload}
          />
        </div>
      </div>
    </section>
  );
}
