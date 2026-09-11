import { summarizeBudget } from "@/lib/client/budget";
import type { ObservatoryStatus } from "@/lib/shared/types";

interface Props {
  status: ObservatoryStatus | null;
  statusKnown: boolean;
  onLogout: () => void;
  logoutPending: boolean;
}

function describe(status: ObservatoryStatus | null, known: boolean): { tone: "ok" | "warn" | "off" | "unknown"; text: string; detail: string } {
  if (!known) return { tone: "unknown", text: "Checking backend", detail: "Reading the backend's readiness endpoint." };
  if (!status) return { tone: "off", text: "Backend status unavailable", detail: "The readiness endpoint did not answer." };
  if (!status.ready) {
    return { tone: "off", text: `Backend not ready${status.init_error ? ` · ${status.init_error}` : ""}`, detail: "Uploads and questions will be refused until it is ready." };
  }
  const budget = summarizeBudget(status);
  const model = status.configured_model || status.effective_generation;
  const tone = budget.exhausted || (status.budget_state !== null && status.budget_state !== "ok") ? "warn" : "ok";
  const text = [model, budget.short].filter(Boolean).join(" · ");
  const detail = `Retrieval: ${status.effective_retrieval}. Generation: ${status.effective_generation}. ${budget.detail}`;
  return { tone, text: text || "Backend ready", detail };
}

export function DeckBar({ status, statusKnown, onLogout, logoutPending }: Props) {
  const s = describe(status, statusKnown);
  return (
    <header className="bar">
      <div className="bar__brand">
        <span className="bar__wordmark">
          Research <em>Observatory</em>
        </span>
        <span className="bar__by">Arjunworks · private demo</span>
      </div>
      <div className="bar__side">
        <p className={`status status--${s.tone}`} title={s.detail}>
          <span className="status__dot" aria-hidden="true" />
          <span className="status__text">{s.text}</span>
          <span className="visually-hidden">. {s.detail}</span>
        </p>
        <button type="button" className="button button--small" onClick={onLogout} disabled={logoutPending}>
          Leave
        </button>
      </div>
    </header>
  );
}
