"use client";

import { useId, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { MAX_PDF_LABEL } from "@/lib/shared/limits";
import { Pulse } from "./Pulse";

interface Props {
  disabled: boolean;
  uploadingName: string | null;
  elapsedSeconds: number;
  onFile: (file: File) => void;
}

export function UploadZone({ disabled, uploadingName, elapsedSeconds, onFile }: Props) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  function pick(files: FileList | null) {
    const file = files?.[0];
    if (file) onFile(file);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setOver(false);
    if (disabled) return;
    pick(event.dataTransfer.files);
  }

  function onChange(event: ChangeEvent<HTMLInputElement>) {
    pick(event.target.files);
    event.target.value = "";
  }

  if (uploadingName) {
    return (
      <div className="dropzone dropzone--busy" role="status" aria-live="polite">
        <Pulse label="Indexing" />
        <div>
          <p className="dropzone__title">Indexing {uploadingName}</p>
          <p className="dropzone__hint">Extracting text, chunking and embedding on the backend · {elapsedSeconds} s</p>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`dropzone${over ? " dropzone--over" : ""}${disabled ? " dropzone--disabled" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
    >
      <input
        ref={inputRef}
        id={inputId}
        className="visually-hidden"
        type="file"
        accept="application/pdf,.pdf"
        disabled={disabled}
        onChange={onChange}
      />
      <p className="dropzone__title">Add a paper</p>
      <p className="dropzone__hint">Drop a public text PDF here, up to {MAX_PDF_LABEL}. It joins the library shared by everyone using this demo.</p>
      <label htmlFor={inputId} className={`button button--small dropzone__browse${disabled ? " button--disabled" : ""}`}>
        Choose a PDF
      </label>
    </div>
  );
}
