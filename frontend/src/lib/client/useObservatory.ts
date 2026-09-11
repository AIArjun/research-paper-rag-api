"use client";

import { useCallback, useEffect, useReducer, useRef } from "react";
import { callApi, postJson, toApiError } from "./api";
import { sha256Hex, type LocalFile, type LocalFileMap } from "./sources";
import { MAX_PDF_BYTES, QUESTION_MAX_CHARS, QUESTION_MIN_CHARS, TOP_K_MAX, TOP_K_MIN } from "@/lib/shared/limits";
import { samplePublicPath, type SamplePaper } from "@/lib/shared/samples";
import type {
  ApiError,
  ObservatoryStatus,
  PaperInfo,
  PapersResponse,
  QueryResponse,
  UploadResponse,
} from "@/lib/shared/types";

export type Pending =
  | { kind: "idle" }
  | { kind: "uploading"; name: string; startedAt: number }
  | { kind: "querying"; question: string; startedAt: number };

export interface AskParams {
  question: string;
  topK: number;
  paperId: string | null;
}

export interface ObservatoryState {
  papers: PaperInfo[] | null;
  papersLoading: boolean;
  papersError: ApiError | null;
  status: ObservatoryStatus | null;
  statusChecked: boolean;
  selectedPaperId: string | null;
  pending: Pending;
  result: QueryResponse | null;
  resultAt: number | null;
  lastAsk: AskParams | null;
  queryError: ApiError | null;
  uploadError: ApiError | null;
  uploadNotice: string | null;
  selectedCitation: number | null;
  localVersion: number;
}

type Action =
  | { type: "papers/loading" }
  | { type: "papers/loaded"; papers: PaperInfo[] }
  | { type: "papers/failed"; error: ApiError }
  | { type: "status/loaded"; status: ObservatoryStatus | null }
  | { type: "paper/select"; paperId: string | null }
  | { type: "upload/started"; name: string; startedAt: number }
  | { type: "upload/succeeded"; notice: string }
  | { type: "upload/failed"; error: ApiError }
  | { type: "upload/dismiss" }
  | { type: "query/started"; ask: AskParams; startedAt: number }
  | { type: "query/succeeded"; result: QueryResponse; at: number }
  | { type: "query/failed"; error: ApiError }
  | { type: "query/dismiss" }
  | { type: "citation/select"; index: number | null }
  | { type: "local/changed" };

const initialState: ObservatoryState = {
  papers: null,
  papersLoading: false,
  papersError: null,
  status: null,
  statusChecked: false,
  selectedPaperId: null,
  pending: { kind: "idle" },
  result: null,
  resultAt: null,
  lastAsk: null,
  queryError: null,
  uploadError: null,
  uploadNotice: null,
  selectedCitation: null,
  localVersion: 0,
};

export function reducer(state: ObservatoryState, action: Action): ObservatoryState {
  switch (action.type) {
    case "papers/loading":
      return { ...state, papersLoading: true, papersError: null };
    case "papers/loaded": {
      const stillThere = state.selectedPaperId && action.papers.some((p) => p.paper_id === state.selectedPaperId);
      return { ...state, papers: action.papers, papersLoading: false, papersError: null, selectedPaperId: stillThere ? state.selectedPaperId : null };
    }
    case "papers/failed":
      return { ...state, papersLoading: false, papersError: action.error };
    case "status/loaded":
      return { ...state, status: action.status, statusChecked: true };
    case "paper/select":
      return { ...state, selectedPaperId: action.paperId };
    case "upload/started":
      return { ...state, pending: { kind: "uploading", name: action.name, startedAt: action.startedAt }, uploadError: null, uploadNotice: null };
    case "upload/succeeded":
      return { ...state, pending: { kind: "idle" }, uploadNotice: action.notice };
    case "upload/failed":
      return { ...state, pending: { kind: "idle" }, uploadError: action.error };
    case "upload/dismiss":
      return { ...state, uploadError: null, uploadNotice: null };
    case "query/started":
      return {
        ...state,
        pending: { kind: "querying", question: action.ask.question, startedAt: action.startedAt },
        lastAsk: action.ask,
        queryError: null,
      };
    case "query/succeeded":
      return { ...state, pending: { kind: "idle" }, result: action.result, resultAt: action.at, selectedCitation: null };
    case "query/failed":
      return { ...state, pending: { kind: "idle" }, queryError: action.error };
    case "query/dismiss":
      return { ...state, queryError: null };
    case "citation/select":
      return { ...state, selectedCitation: action.index };
    case "local/changed":
      return { ...state, localVersion: state.localVersion + 1 };
  }
}

export interface ObservatoryActions {
  refreshPapers(): Promise<void>;
  refreshStatus(): Promise<void>;
  selectPaper(paperId: string | null): void;
  upload(file: File): Promise<void>;
  addSample(sample: SamplePaper): Promise<void>;
  ask(question: string, topK: number): Promise<void>;
  askAgain(): Promise<void>;
  selectCitation(index: number | null): void;
  dismissUpload(): void;
  dismissQuery(): void;
  logout(): Promise<ApiError | null>;
}

export interface Observatory {
  state: ObservatoryState;
  actions: ObservatoryActions;
  localFiles: LocalFileMap;
}

export function validQuestion(question: string): boolean {
  const length = question.trim().length;
  return length >= QUESTION_MIN_CHARS && length <= QUESTION_MAX_CHARS;
}

const LOCAL_UPLOAD_ERROR = (message: string): ApiError => ({ category: "invalid_pdf", message });

/**
 * All observatory state and the only code paths that talk to the BFF.
 * Model calls happen in `ask` alone, and only when the visitor invokes it.
 */
export function useObservatory(onLoggedOut: () => void): Observatory {
  const [state, dispatch] = useReducer(reducer, initialState);
  const pendingRef = useRef<Pending>({ kind: "idle" });
  const localRef = useRef<Map<string, LocalFile>>(new Map());
  const stateRef = useRef(state);
  stateRef.current = state;
  // Kept in a ref so an unstable callback identity can never re-trigger the load effect.
  const loggedOutRef = useRef(onLoggedOut);
  loggedOutRef.current = onLoggedOut;
  const signalLoggedOut = useCallback(() => loggedOutRef.current(), []);

  const setPending = (pending: Pending) => {
    pendingRef.current = pending;
  };

  const refreshStatus = useCallback(async () => {
    try {
      const status = await callApi<ObservatoryStatus>("/api/status");
      dispatch({ type: "status/loaded", status });
    } catch (failure) {
      const error = toApiError(failure);
      if (error.category === "unauthenticated") signalLoggedOut();
      dispatch({ type: "status/loaded", status: null });
    }
  }, [signalLoggedOut]);

  const refreshPapers = useCallback(async () => {
    dispatch({ type: "papers/loading" });
    try {
      const response = await callApi<PapersResponse>("/api/papers");
      dispatch({ type: "papers/loaded", papers: response.papers });
    } catch (failure) {
      const error = toApiError(failure);
      if (error.category === "unauthenticated") signalLoggedOut();
      dispatch({ type: "papers/failed", error });
    }
  }, [signalLoggedOut]);

  // The library and the backend status load once when the workspace opens.
  // Neither touches the model; questions are sent only from `ask`.
  useEffect(() => {
    void refreshPapers();
    void refreshStatus();
  }, [refreshPapers, refreshStatus]);

  useEffect(() => {
    const files = localRef.current;
    return () => {
      for (const file of files.values()) URL.revokeObjectURL(file.url);
      files.clear();
    };
  }, []);

  const rememberLocal = useCallback(async (paperId: string, file: File, bytes: ArrayBuffer) => {
    // The backend id is the SHA-256 of the exact bytes sent; when the browser can
    // hash, it double-checks before tying this file to that id.
    const digest = await sha256Hex(bytes);
    if (digest && digest !== paperId.toLowerCase()) return;
    const existing = localRef.current.get(paperId);
    if (existing) URL.revokeObjectURL(existing.url);
    const url = URL.createObjectURL(new Blob([bytes], { type: "application/pdf" }));
    localRef.current.set(paperId, { url, name: file.name, bytes: file.size });
    dispatch({ type: "local/changed" });
  }, []);

  const sendUpload = useCallback(
    async (file: File, label: string) => {
      if (pendingRef.current.kind !== "idle") return;
      if (file.size > MAX_PDF_BYTES) {
        dispatch({ type: "upload/failed", error: { category: "file_too_large", message: "That PDF is larger than 4 MB, the ceiling for this demo." } });
        return;
      }
      if (file.size === 0) {
        dispatch({ type: "upload/failed", error: LOCAL_UPLOAD_ERROR("That file is empty.") });
        return;
      }
      if (!file.name.toLowerCase().endsWith(".pdf") || (file.type && file.type !== "application/pdf")) {
        dispatch({ type: "upload/failed", error: LOCAL_UPLOAD_ERROR("Only PDF files can be added.") });
        return;
      }
      const startedAt = Date.now();
      setPending({ kind: "uploading", name: label, startedAt });
      dispatch({ type: "upload/started", name: label, startedAt });
      let bytes: ArrayBuffer;
      try {
        bytes = await file.arrayBuffer();
      } catch {
        setPending({ kind: "idle" });
        dispatch({ type: "upload/failed", error: LOCAL_UPLOAD_ERROR("The file could not be read from this device.") });
        return;
      }
      const form = new FormData();
      form.append("file", new File([bytes], file.name, { type: "application/pdf" }), file.name);
      try {
        const uploaded = await callApi<UploadResponse>("/api/papers/upload", { method: "POST", body: form });
        await rememberLocal(uploaded.paper_id, file, bytes);
        setPending({ kind: "idle" });
        dispatch({ type: "upload/succeeded", notice: `${label} is indexed: ${uploaded.pages} pages, ${uploaded.chunks} chunks.` });
        await refreshPapers();
        await refreshStatus();
      } catch (failure) {
        const error = toApiError(failure);
        setPending({ kind: "idle" });
        if (error.category === "unauthenticated") signalLoggedOut();
        dispatch({ type: "upload/failed", error });
      }
    },
    [signalLoggedOut, refreshPapers, refreshStatus, rememberLocal],
  );

  const upload = useCallback((file: File) => sendUpload(file, file.name), [sendUpload]);

  const addSample = useCallback(
    async (sample: SamplePaper) => {
      if (pendingRef.current.kind !== "idle") return;
      let blob: Blob;
      try {
        const response = await fetch(samplePublicPath(sample), { cache: "force-cache" });
        if (!response.ok) throw new Error("sample missing");
        blob = await response.blob();
      } catch {
        dispatch({ type: "upload/failed", error: { category: "unreachable", message: "The bundled sample could not be loaded from this site." } });
        return;
      }
      const bytes = await blob.arrayBuffer();
      const digest = await sha256Hex(bytes);
      if (digest && digest !== sample.sha256) {
        dispatch({ type: "upload/failed", error: { category: "invalid_pdf", message: "The bundled sample did not match its recorded checksum, so it was not sent." } });
        return;
      }
      await sendUpload(new File([bytes], sample.file, { type: "application/pdf" }), sample.title);
    },
    [sendUpload],
  );

  const runAsk = useCallback(
    async (ask: AskParams) => {
      if (pendingRef.current.kind !== "idle") return;
      if (!validQuestion(ask.question)) return;
      const topK = Math.min(TOP_K_MAX, Math.max(TOP_K_MIN, Math.round(ask.topK)));
      const startedAt = Date.now();
      setPending({ kind: "querying", question: ask.question, startedAt });
      dispatch({ type: "query/started", ask: { ...ask, topK }, startedAt });
      try {
        const payload: { question: string; top_k: number; paper_id?: string } = { question: ask.question.trim(), top_k: topK };
        if (ask.paperId) payload.paper_id = ask.paperId;
        const result = await postJson<QueryResponse>("/api/query", payload);
        setPending({ kind: "idle" });
        dispatch({ type: "query/succeeded", result, at: Date.now() });
        void refreshStatus();
      } catch (failure) {
        const error = toApiError(failure);
        setPending({ kind: "idle" });
        if (error.category === "unauthenticated") signalLoggedOut();
        dispatch({ type: "query/failed", error });
        if (error.category === "budget_exhausted" || error.category === "empty_corpus") void refreshStatus();
      }
    },
    [signalLoggedOut, refreshStatus],
  );

  const ask = useCallback(
    (question: string, topK: number) => runAsk({ question, topK, paperId: stateRef.current.selectedPaperId }),
    [runAsk],
  );

  const askAgain = useCallback(async () => {
    const last = stateRef.current.lastAsk;
    if (last) await runAsk(last);
  }, [runAsk]);

  const selectPaper = useCallback((paperId: string | null) => dispatch({ type: "paper/select", paperId }), []);
  const selectCitation = useCallback((index: number | null) => dispatch({ type: "citation/select", index }), []);
  const dismissUpload = useCallback(() => dispatch({ type: "upload/dismiss" }), []);
  const dismissQuery = useCallback(() => dispatch({ type: "query/dismiss" }), []);

  /** Resolves to null once the server has cleared the cookie; otherwise the error to show, and the session stays. */
  const logout = useCallback(async (): Promise<ApiError | null> => {
    try {
      await callApi<void>("/api/auth/logout", { method: "POST" });
    } catch (failure) {
      return toApiError(failure);
    }
    for (const file of localRef.current.values()) URL.revokeObjectURL(file.url);
    localRef.current.clear();
    signalLoggedOut();
    return null;
  }, [signalLoggedOut]);

  return {
    state,
    actions: { refreshPapers, refreshStatus, selectPaper, upload, addSample, ask, askAgain, selectCitation, dismissUpload, dismissQuery, logout },
    localFiles: localRef.current,
  };
}
