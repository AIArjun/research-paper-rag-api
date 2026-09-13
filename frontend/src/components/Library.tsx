"use client";

import { formatBytes, paperTitle, sampleInCorpus, type LocalFileMap } from "@/lib/client/sources";
import { SAMPLE_PAPERS, sampleByPaperId, type SamplePaper } from "@/lib/shared/samples";
import type { ApiError, PaperInfo } from "@/lib/shared/types";
import { Notice } from "./Notice";
import { UploadZone } from "./UploadZone";

interface Props {
  papers: PaperInfo[] | null;
  loading: boolean;
  error: ApiError | null;
  selectedPaperId: string | null;
  localFiles: LocalFileMap;
  busy: boolean;
  uploadingName: string | null;
  uploadElapsed: number;
  uploadError: ApiError | null;
  uploadNotice: string | null;
  onSelect: (paperId: string | null) => void;
  onRefresh: () => void;
  onFile: (file: File) => void;
  onAddSample: (sample: SamplePaper) => void;
  onDismissUpload: () => void;
}

export function Library(props: Props) {
  const { papers, loading, error, selectedPaperId, localFiles, busy, onSelect } = props;
  const ready = (papers ?? []).filter((p) => p.status === "ready");
  const missingSamples = SAMPLE_PAPERS.filter((s) => !sampleInCorpus(s, papers));

  return (
    <div className="library">
      <div className="library__head">
        <h2 className="library__title">Library</h2>
        <button
          type="button"
          className="button button--quiet button--small"
          onClick={props.onRefresh}
          disabled={loading}
          aria-label="Refresh the paper list"
        >
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {error && <Notice error={error} action={<button type="button" className="button button--small" onClick={props.onRefresh}>Try again</button>} />}

      {papers && papers.length === 0 && !error && (
        <p className="library__empty">
          The library is empty. Add a sample or your own public PDF to begin.
        </p>
      )}

      {papers && papers.length > 0 && (
        <div className="shelf" role="radiogroup" aria-label="Scope of the next question">
          <button
            type="button"
            role="radio"
            aria-checked={selectedPaperId === null}
            className={`plate plate--all${selectedPaperId === null ? " plate--selected" : ""}`}
            onClick={() => onSelect(null)}
          >
            <span className="plate__title">All papers</span>
            <span className="plate__meta">
              {ready.length} indexed · {ready.reduce((sum, p) => sum + p.chunks, 0)} passages
            </span>
          </button>
          {papers.map((paper) => {
            const sample = sampleByPaperId(paper.paper_id);
            const local = localFiles.get(paper.paper_id);
            const pending = paper.status !== "ready";
            const selected = selectedPaperId === paper.paper_id;
            return (
              <button
                key={paper.paper_id}
                type="button"
                role="radio"
                aria-checked={selected}
                className={`plate${selected ? " plate--selected" : ""}${pending ? " plate--pending" : ""}`}
                onClick={() => onSelect(paper.paper_id)}
                disabled={pending}
              >
                <span className="plate__title">{paperTitle(paper.paper_id, paper.filename)}</span>
                <span className="plate__meta">
                  {paper.pages !== null ? `${paper.pages} pages · ` : ""}
                  {paper.chunks} passages
                  {pending ? " · pending cleanup" : ""}
                </span>
                <span className="plate__tags">
                  {sample && <span className="tag tag--glass">bundled sample</span>}
                  {!sample && local && <span className="tag tag--gold">PDF in this session</span>}
                  {!sample && !local && <span className="tag">PDF not held here</span>}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {missingSamples.length > 0 && (
        <div className="samples">
          <p className="samples__label">{papers && papers.length === 0 ? "Start with a bundled paper" : "Bundled papers not in the corpus"}</p>
          {missingSamples.map((sample) => (
            <button
              key={sample.sha256}
              type="button"
              className="sample"
              disabled={busy}
              onClick={() => props.onAddSample(sample)}
            >
              <span className="sample__title">{sample.title}</span>
              <span className="sample__meta">
                {sample.authors} · {sample.pages} pages · {formatBytes(sample.bytes)}
              </span>
              <span className="sample__cta">Add to corpus</span>
            </button>
          ))}
        </div>
      )}

      <UploadZone disabled={busy} uploadingName={props.uploadingName} elapsedSeconds={props.uploadElapsed} onFile={props.onFile} />

      {props.uploadError && <Notice error={props.uploadError} onDismiss={props.onDismissUpload} />}
      {props.uploadNotice && (
        <p className="library__notice" role="status">
          {props.uploadNotice}{" "}
          <button type="button" className="link" onClick={props.onDismissUpload}>
            Dismiss
          </button>
        </p>
      )}

      <p className="library__foot">
        Shared public-paper demo. Papers you add are visible to other visitors. Upload public documents only.
      </p>
    </div>
  );
}
