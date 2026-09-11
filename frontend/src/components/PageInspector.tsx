"use client";

import { embeddedPageUrl, pageUrl, paperTitle, resolveSource, type LocalFileMap } from "@/lib/client/sources";
import type { Citation, PaperInfo } from "@/lib/shared/types";

interface Props {
  citation: Citation | null;
  index: number | null;
  papers: PaperInfo[] | null;
  localFiles: LocalFileMap;
  onRequestUpload: () => void;
}

export function PageInspector({ citation, index, papers, localFiles, onRequestUpload }: Props) {
  if (!citation || index === null) {
    return (
      <aside className="inspector inspector--idle" aria-label="Page inspector">
        <div className="inspector__frame inspector__frame--empty">
          <p className="inspector__empty-title">Page inspector</p>
          <p className="inspector__empty-text">Select a passage or a point in the constellation to open the physical PDF page it came from.</p>
        </div>
      </aside>
    );
  }

  const source = resolveSource(citation, papers, localFiles);
  const title = source?.title ?? paperTitle(citation.paper_id, citation.paper);
  const page = citation.page;

  return (
    <aside className="inspector" aria-label={`Page inspector: ${title}`}>
      <div className="inspector__head">
        <p className="inspector__kicker">Passage {index + 1} · physical page</p>
        <h3 className="inspector__title">
          {page !== null ? `Page ${page}` : "Unknown page"}
          {source?.pages ? <span className="inspector__of"> of {source.pages}</span> : null}
        </h3>
        <p className="inspector__paper">{title}</p>
      </div>

      <blockquote className="inspector__quote">
        <p>{citation.text}</p>
        <footer>Retrieved passage preview (up to 300 characters); the full passage may continue on the page.</footer>
      </blockquote>

      {source ? (
        <>
          <div className="inspector__frame">
            <iframe
              key={`${source.url}#${page ?? 0}`}
              className="inspector__pdf"
              src={embeddedPageUrl(source.url, page)}
              title={`${title}, page ${page ?? "unknown"}`}
              loading="lazy"
            />
          </div>
          <p className="inspector__links">
            <a className="button button--small" href={pageUrl(source.url, page)} target="_blank" rel="noopener noreferrer">
              Open page {page ?? ""} in a new tab
            </a>
            <span className="inspector__hint">
              {source.kind === "sample" ? "Bundled copy, served from this site." : "Your copy, held in this browser session only."}
              {" "}If the embedded viewer shows page 1, use the tab link.
            </span>
          </p>
        </>
      ) : (
        <div className="inspector__frame inspector__frame--missing">
          <p className="inspector__empty-title">PDF not available here</p>
          <p className="inspector__empty-text">
            The original file for <strong>{citation.paper}</strong> is not held in this browser session, so page {page ?? "?"} cannot be opened. Re-add the same PDF to inspect it; its digest must match this paper.
          </p>
          <button type="button" className="button button--small" onClick={onRequestUpload}>
            Re-add the PDF
          </button>
        </div>
      )}
      <p className="inspector__foot">Page numbers are 1-based physical PDF pages, not printed page labels.</p>
    </aside>
  );
}
