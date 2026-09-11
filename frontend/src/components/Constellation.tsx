"use client";

import { paperShortTitle } from "@/lib/client/sources";
import type { Citation } from "@/lib/shared/types";

interface Props {
  citations: Citation[];
  selected: number | null;
  onSelect: (index: number) => void;
}

const SIZE = 520;
const CENTER = SIZE / 2;
export const HUES = ["glass", "gold", "ivory", "lilac", "coral"] as const;

export interface PaperLane {
  key: string;
  label: string;
  hue: (typeof HUES)[number];
  radius: number;
}

/** One lane per distinct paper, in order of first appearance among the citations. */
export function lanesFor(citations: readonly Citation[]): PaperLane[] {
  const lanes: PaperLane[] = [];
  for (const c of citations) {
    const key = c.paper_id ?? c.paper;
    if (lanes.some((l) => l.key === key)) continue;
    const i = lanes.length;
    lanes.push({ key, label: paperShortTitle(c.paper_id, c.paper), hue: HUES[i % HUES.length] ?? "glass", radius: 150 + i * 24 });
  }
  return lanes;
}

export interface NodePosition {
  x: number;
  y: number;
  lane: PaperLane;
}

/** Nodes sit on their paper's orbit, spaced evenly by rank, rank 1 at the top. */
export function positionsFor(citations: readonly Citation[], lanes: readonly PaperLane[]): NodePosition[] {
  const n = Math.max(citations.length, 1);
  return citations.map((c, i) => {
    const lane = lanes.find((l) => l.key === (c.paper_id ?? c.paper)) ?? lanes[0]!;
    const angle = -Math.PI / 2 + (i * 2 * Math.PI) / n;
    return { x: CENTER + Math.cos(angle) * lane.radius, y: CENTER + Math.sin(angle) * lane.radius, lane };
  });
}

export function Constellation({ citations, selected, onSelect }: Props) {
  const lanes = lanesFor(citations);
  const positions = positionsFor(citations, lanes);

  return (
    <figure className="constellation" aria-labelledby="constellation-caption">
      <div className="constellation__stage">
        <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="constellation__svg" aria-hidden="true" focusable="false">
          <g className="constellation__rings">
            {lanes.map((lane) => (
              <circle key={lane.key} cx={CENTER} cy={CENTER} r={lane.radius} className={`ring ring--${lane.hue}`} />
            ))}
          </g>
          <g className="constellation__lines">
            {positions.map((p, i) => (
              <line
                key={i}
                x1={CENTER}
                y1={CENTER}
                x2={p.x}
                y2={p.y}
                className={`beam beam--${p.lane.hue}${selected === i ? " beam--selected" : ""}`}
                style={{ animationDelay: `${i * 90}ms` }}
              />
            ))}
          </g>
          <g className="constellation__core">
            <circle cx={CENTER} cy={CENTER} r="26" className="core__halo" />
            <circle cx={CENTER} cy={CENTER} r="9" className="core__dot" />
          </g>
        </svg>
        {positions.map((p, i) => {
          const c = citations[i]!;
          const label = `${p.lane.label}, page ${c.page ?? "unknown"}, passage ${i + 1} of ${citations.length}`;
          return (
            <button
              key={i}
              type="button"
              className={`node node--${p.lane.hue}${selected === i ? " node--selected" : ""}`}
              style={{ left: `${(p.x / SIZE) * 100}%`, top: `${(p.y / SIZE) * 100}%`, animationDelay: `${120 + i * 90}ms` }}
              aria-pressed={selected === i}
              aria-label={`Show evidence: ${label}`}
              title={label}
              onClick={() => onSelect(i)}
            >
              <span className="node__page">{c.page ?? "?"}</span>
            </button>
          );
        })}
      </div>
      <figcaption id="constellation-caption" className="constellation__caption">
        <span className="constellation__legend">
          {lanes.map((lane) => (
            <span key={lane.key} className="legend">
              <span className={`legend__swatch legend__swatch--${lane.hue}`} aria-hidden="true" />
              {lane.label}
            </span>
          ))}
        </span>
        <span className="constellation__note">Each point is one returned passage, labelled with its physical PDF page. Order follows retrieval rank.</span>
      </figcaption>
    </figure>
  );
}
