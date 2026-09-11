"use client";

import { useRouter } from "next/navigation";
import { useId, useState, type FormEvent } from "react";
import { postJson, toApiError } from "@/lib/client/api";
import type { ApiError } from "@/lib/shared/types";
import { OrbitField } from "./OrbitField";

interface Props {
  configured: boolean;
}

export function Entry({ configured }: Props) {
  const router = useRouter();
  const inputId = useId();
  const [passcode, setPasscode] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (pending || passcode.length === 0) return;
    setPending(true);
    setError(null);
    try {
      await postJson<void>("/api/auth/login", { passcode });
      setPasscode("");
      router.refresh();
    } catch (failure) {
      setError(toApiError(failure));
      setPending(false);
    }
  }

  return (
    <main className="entry">
      <OrbitField className="entry__field" />
      <section className="entry__card" aria-labelledby="entry-title">
        <p className="entry__kicker">Arjunworks · private demo</p>
        <h1 id="entry-title" className="entry__title">
          Research <span className="entry__title-accent">Observatory</span>
        </h1>
        <p className="entry__lede">
          Ask questions of public papers. Read the answer, inspect the evidence, and open the cited page.
        </p>
        {configured ? (
          <form className="entry__form" onSubmit={submit} noValidate>
            <label className="entry__label" htmlFor={inputId}>
              Passcode
            </label>
            <div className="entry__row">
              <input
                id={inputId}
                className="entry__input"
                type="password"
                name="passcode"
                autoComplete="current-password"
                inputMode="text"
                required
                maxLength={512}
                value={passcode}
                onChange={(event) => setPasscode(event.target.value)}
                aria-describedby={error ? "entry-error" : undefined}
                aria-invalid={error ? true : undefined}
                disabled={pending}
              />
              <button className="button button--primary" type="submit" disabled={pending || passcode.length === 0}>
                {pending ? "Opening…" : "Enter"}
              </button>
            </div>
            <p id="entry-error" className="entry__status" role="status" aria-live="polite">
              {error ? error.message : " "}
            </p>
          </form>
        ) : (
          <p className="entry__status entry__status--block" role="status">
            The observatory is not open yet. The operator needs to finish the server setup before anyone can enter.
          </p>
        )}
        <p className="entry__footnote">
          Shared public-paper demo. Answers are a research preview: check the cited sources.
        </p>
      </section>
    </main>
  );
}
