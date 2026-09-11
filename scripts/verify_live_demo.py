"""Bounded, explicit verification of one deployed protected demo (Stage 3).

Credential: the shared access token is read from the DEMO_ACCESS_TOKEN
environment variable only. It is never accepted on the command line, never
printed, and never written to the evidence files (the files are refused if
the token would appear in them).

Default run (no paid model call, nothing deleted):
  1. GET /ready before anything else (ledger identity and counts).
  2. Missing and wrong bearer tokens on /papers must answer 401.
  3. Authenticated /papers listing.
  4. Upload the manifest fixtures after verifying their SHA-256 digests
     (identical bytes are idempotent server-side; nothing is ever deleted).
  5. One empty-filter query (a paper_id that cannot exist) that must abstain
     with model_used=not-invoked and no model_usage.
  6. GET /ready again; the ledger call count must be unchanged.

Live generation runs only with --live, is capped at --max-live-calls (never
more than 5 per invocation), makes exactly one attempt per question with no
retry, and stops at the first non-200 answer. Every call is bracketed by two
/ready snapshots so the measured ledger delta is recorded next to the
provider-reported model_usage.

--compare-ledger PREVIOUS.json checks, on the very first /ready snapshot and
before any upload or paid call, that the ledger identity and counts in an
earlier evidence file survived a deploy or restart; a replaced ledger stops
the run.

Exit status: 0 when every recorded check passed (safe checks, the ledger
comparison when requested, and with --live every call answered 200 from the
configured model with measured usage confirmed by the ledger delta); 1 when a
check failed or a precondition aborted the run; 2 when the token was missing
or would have appeared in the evidence. Semantic answer quality and citation
support are judged manually from the evidence files (docs/STAGE3.md).
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

MAX_LIVE_CALLS = 5
WRONG_TOKEN = "wrong-verification-token-0123456789abcdef0123"  # deliberately invalid, not a secret
EVIDENCE_FILE_PREFIX = "stage3-evidence-"

DEFAULT_MANIFEST = [
    {
        "file": "attention-is-all-you-need.pdf",
        "source_url": "https://arxiv.org/pdf/1706.03762",
        "sha256": "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697",
        "pages": 15,
        "chunks": 110,
    },
    {
        "file": "retrieval-augmented-generation.pdf",
        "source_url": "https://arxiv.org/pdf/2005.11401",
        "sha256": "23e3249e9a1e75418d82efecab0ea8c4d033b89c93742f63208d47ce01f21233",
        "pages": 19,
        "chunks": 188,
    },
]

# The evidence rubric (docs/STAGE3.md): specific, checkable against the PDFs.
DEFAULT_QUESTIONS = [
    {
        "id": "transformer-encoder-layers",
        "question": (
            "How many identical layers does the Transformer encoder stack use, and "
            "what are the two sub-layers in each encoder layer?"
        ),
        "expected_support": "Attention Is All You Need, Section 3.1: N = 6 layers; multi-head "
                            "self-attention and a position-wise feed-forward network.",
    },
    {
        "id": "transformer-bleu-en-de",
        "question": "What BLEU score did the big Transformer model achieve on the WMT 2014 "
                    "English-to-German translation task?",
        "expected_support": "Attention Is All You Need, abstract and Table 2: 28.4 BLEU.",
    },
    {
        "id": "rag-retriever-generator",
        "question": "In the RAG paper, which pre-trained models are used as the retriever and "
                    "as the generator?",
        "expected_support": "Retrieval-Augmented Generation, Section 2: a DPR bi-encoder retriever "
                            "and a BART-large generator.",
    },
]


class VerificationError(RuntimeError):
    """A precondition failed; nothing further is attempted."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def token_from_environment(environ) -> str:
    token = environ.get("DEMO_ACCESS_TOKEN", "")
    if not (32 <= len(token) <= 512) or not token.isascii() or not token.isprintable() \
            or any(character.isspace() for character in token):
        raise VerificationError(
            "DEMO_ACCESS_TOKEN must be set in the environment (32-512 printable ASCII characters)."
        )
    return token


class Client:
    """Minimal urllib client. One attempt per request; no retries anywhere."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self._token = token

    def request(self, method: str, path: str, *, credential: str = "configured", body=None,
                timeout: float = 30.0, upload=None) -> dict:
        headers = {"Accept": "application/json"}
        if credential == "configured":
            headers["Authorization"] = "Bearer " + self._token
        elif credential == "wrong":
            headers["Authorization"] = "Bearer " + WRONG_TOKEN
        data = None
        if upload is not None:
            filename, content = upload
            boundary = "stage3-" + uuid.uuid4().hex
            data = (
                ("--" + boundary + "\r\n"
                 'Content-Disposition: form-data; name="file"; filename="' + filename + '"\r\n'
                 "Content-Type: application/pdf\r\n\r\n").encode()
                + content + ("\r\n--" + boundary + "--\r\n").encode()
            )
            headers["Content-Type"] = "multipart/form-data; boundary=" + boundary
        elif body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        started = datetime.now(timezone.utc)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status, raw, response_headers = response.status, response.read(), response.headers
        except urllib.error.HTTPError as error:
            status, raw, response_headers = error.code, error.read(), error.headers
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            return {"method": method, "path": path, "status": None,
                    "error": type(error).__name__, "at": started.isoformat()}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            parsed = {"non_json_body_bytes": len(raw)}
        kept_headers = {
            name: response_headers.get(name)
            for name in ("X-Request-ID", "WWW-Authenticate", "Retry-After", "Content-Type")
            if response_headers.get(name) is not None
        }
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        return {"method": method, "path": path, "status": status, "body": parsed,
                "headers": kept_headers, "at": started.isoformat(), "elapsed_seconds": round(elapsed, 3)}


def ledger_view(ready: dict) -> dict:
    """The identity and counts that must survive a restart, or None when unavailable."""
    body = ready.get("body") if isinstance(ready.get("body"), dict) else {}
    budget = body.get("model_budget") if isinstance(body.get("model_budget"), dict) else {}
    usage = budget.get("usage") if isinstance(budget.get("usage"), dict) else {}
    return {
        "status": ready.get("status"),
        "ready": body.get("ready"),
        "state": budget.get("state"),
        "token_bound": budget.get("token_bound"),
        "configured_model": body.get("configured_model"),
        "effective_generation": body.get("effective_generation"),
        "ledger_created_at": usage.get("ledger_created_at"),
        "calls_today": usage.get("calls_today"),
        "calls_total": usage.get("calls_total"),
        "tokens_charged_today": usage.get("tokens_charged_today"),
        "tokens_charged_total": usage.get("tokens_charged_total"),
        "tokens_measured_total": usage.get("tokens_measured_total"),
        "calls_unsettled": usage.get("calls_unsettled"),
    }


def load_manifest(path) -> list[dict]:
    if path is None:
        return [dict(entry) for entry in DEFAULT_MANIFEST]
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"file", "sha256"}
    if not isinstance(entries, list) or any(not required <= set(entry) for entry in entries):
        raise VerificationError("The manifest must be a list of objects with file and sha256.")
    return entries


def fixture_bytes(entry: dict, fixture_dir: Path, allow_download: bool) -> bytes:
    """Return verified fixture bytes; download only when explicitly allowed."""
    fixture_dir.mkdir(parents=True, exist_ok=True)
    location = fixture_dir / entry["file"]
    if not location.is_file():
        if not allow_download or not entry.get("source_url"):
            raise VerificationError("Fixture missing locally and downloads are not enabled: " + entry["file"])
        request = urllib.request.Request(entry["source_url"], headers={"User-Agent": "stage3-verify/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            location.write_bytes(response.read())
    content = location.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != entry["sha256"]:
        raise VerificationError("Fixture digest mismatch; refusing to upload: " + entry["file"])
    return content


def load_questions(path) -> list[dict]:
    if path is None:
        return [dict(question) for question in DEFAULT_QUESTIONS]
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(questions, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("question"), str) for item in questions
    ):
        raise VerificationError("The questions file must be a list of objects with a question string.")
    return questions


def safe_phase(client: Client, args, evidence: dict) -> bool:
    """Everything that never triggers a paid call. Returns True when all checks passed."""
    checks = evidence["checks"]
    before = client.request("GET", "/ready", credential="none", timeout=30)
    evidence["ready_before"] = before
    checks["ready_before_200"] = before.get("status") == 200
    if before.get("status") != 200:
        return False
    if args.compare_ledger:
        # A replaced ledger must be discovered on the first snapshot, never after spending.
        evidence["ledger_comparison"] = compare_ledger(args.compare_ledger, before)
        checks["ledger_persisted"] = evidence["ledger_comparison"]["ledger_persisted"]
        if not checks["ledger_persisted"]:
            return False

    missing = client.request("GET", "/papers", credential="none", timeout=30)
    wrong = client.request("GET", "/papers", credential="wrong", timeout=30)
    evidence["auth"] = {"missing_token": missing, "wrong_token": wrong}
    checks["missing_token_401"] = (
        missing.get("status") == 401 and "WWW-Authenticate" in (missing.get("headers") or {})
    )
    checks["wrong_token_401"] = wrong.get("status") == 401

    listing = client.request("GET", "/papers", timeout=30)
    evidence["papers_before"] = listing
    checks["authenticated_list_200"] = listing.get("status") == 200

    uploads = []
    all_uploads_ok = True
    all_uploads_match = True
    if not args.skip_uploads:
        for entry in load_manifest(args.manifest):
            content = fixture_bytes(entry, Path(args.fixture_dir), args.download_fixtures)
            result = client.request("POST", "/papers/upload", upload=(entry["file"], content),
                                    timeout=args.upload_timeout)
            body = result.get("body") if isinstance(result.get("body"), dict) else {}
            record = {
                "file": entry["file"], "sha256": entry["sha256"], "bytes": len(content), "response": result,
                "expected": {key: entry.get(key) for key in ("pages", "chunks") if key in entry},
            }
            if result.get("status") == 200:
                record["matches_expected"] = all(
                    body.get(key) == entry[key] for key in ("pages", "chunks") if key in entry
                )
                all_uploads_match = all_uploads_match and record["matches_expected"]
            else:
                all_uploads_ok = False
            uploads.append(record)
        evidence["papers_after_upload"] = client.request("GET", "/papers", timeout=30)
    evidence["uploads"] = uploads
    checks["uploads_200"] = all_uploads_ok
    # A paper that ingested with other page/chunk counts than declared is not the
    # reviewed fixture; the safe phase fails and no paid call is attempted.
    checks["uploads_match_expected"] = all_uploads_match

    abstention = client.request(
        "POST", "/query",
        body={"question": "What does this paper conclude?", "paper_id": "no-such-paper-" + uuid.uuid4().hex,
              "top_k": 3},
        timeout=args.query_timeout,
    )
    body = abstention.get("body") if isinstance(abstention.get("body"), dict) else {}
    evidence["abstention"] = abstention
    checks["abstention_not_invoked"] = (
        abstention.get("status") == 200 and body.get("citations") == []
        and body.get("model_used") == "not-invoked" and body.get("model_usage") is None
    )

    after = client.request("GET", "/ready", credential="none", timeout=30)
    evidence["ready_after_safe_phase"] = after
    checks["no_model_call_in_safe_phase"] = (
        after.get("status") == 200
        and ledger_view(after)["calls_total"] == ledger_view(before)["calls_total"]
        and ledger_view(after)["ledger_created_at"] == ledger_view(before)["ledger_created_at"]
    )
    return all(checks[name] for name in (
        "ready_before_200", "ledger_persisted", "missing_token_401", "wrong_token_401", "authenticated_list_200",
        "uploads_200", "uploads_match_expected", "abstention_not_invoked", "no_model_call_in_safe_phase",
    ) if name in checks)


def accounted_live_result(answer: dict, delta: dict, expected_model) -> list[str]:
    """Mechanical reasons a live answer is not an accounted model-backed answer (empty means accounted).

    Semantic quality and citation support stay a manual judgment (docs/STAGE3.md).
    """
    reasons = []
    body = answer.get("body") if isinstance(answer.get("body"), dict) else {}
    usage = body.get("model_usage") if isinstance(body.get("model_usage"), dict) else None
    if answer.get("status") != 200:
        reasons.append("status_not_200")
    if body.get("model_used") in (None, "not-invoked", "demo-mode") or (
        expected_model and body.get("model_used") != expected_model
    ):
        reasons.append("model_used_not_configured_model")
    if not body.get("citations"):
        reasons.append("no_citations")
    if usage is None or usage.get("accounting") != "measured" \
            or type(usage.get("input_tokens")) is not int or type(usage.get("output_tokens")) is not int:
        reasons.append("usage_not_measured")
    if delta.get("calls_total") != 1:
        reasons.append("ledger_calls_delta_not_one")
    if usage is not None and type(usage.get("tokens_charged")) is int \
            and delta.get("tokens_charged_total") != usage["tokens_charged"]:
        reasons.append("ledger_tokens_delta_mismatch")
    return reasons


def live_phase(client: Client, args, evidence: dict) -> None:
    """Explicit, bounded paid calls: one attempt each, no retry, stop at the first non-200.

    Sets checks.live_phase_passed: every attempted call answered 200 from the
    configured model with measured usage that the ledger delta confirms.
    """
    questions = load_questions(args.questions)[: args.max_live_calls]
    expected_model = ledger_view(evidence.get("ready_before") or {}).get("configured_model")
    results = []
    all_accounted = True
    for item in questions:
        before = client.request("GET", "/ready", credential="none", timeout=30)
        payload = {"question": item["question"], "top_k": int(item.get("top_k", 5))}
        if item.get("paper_id"):
            payload["paper_id"] = item["paper_id"]
        answer = client.request("POST", "/query", body=payload, timeout=args.query_timeout)
        after = client.request("GET", "/ready", credential="none", timeout=30)
        before_view, after_view = ledger_view(before), ledger_view(after)
        delta = {}
        for key in ("calls_total", "tokens_charged_total", "tokens_measured_total"):
            if isinstance(before_view.get(key), int) and isinstance(after_view.get(key), int):
                delta[key] = after_view[key] - before_view[key]
        reasons = accounted_live_result(answer, delta, expected_model)
        all_accounted = all_accounted and not reasons
        results.append({
            "id": item.get("id"), "request": payload, "expected_support": item.get("expected_support"),
            "ledger_before": before_view, "response": answer, "ledger_after": after_view, "ledger_delta": delta,
            "accounted": not reasons, "failure_reasons": reasons,
        })
        if answer.get("status") != 200:
            evidence["live_stopped_early"] = {"after_calls": len(results), "status": answer.get("status")}
            break
    evidence["live"] = results
    evidence["checks"]["live_calls_attempted"] = len(results)
    evidence["checks"]["live_phase_passed"] = all_accounted and "live_stopped_early" not in evidence


def compare_ledger(previous_path: str, current_ready: dict) -> dict:
    previous = json.loads(Path(previous_path).read_text(encoding="utf-8"))
    earlier = ledger_view(previous.get("ready_final") or previous.get("ready_before") or {})
    now = ledger_view(current_ready)
    persisted = (
        earlier.get("ledger_created_at") is not None
        and earlier.get("ledger_created_at") == now.get("ledger_created_at")
        and isinstance(now.get("calls_total"), int) and isinstance(earlier.get("calls_total"), int)
        and now["calls_total"] >= earlier["calls_total"]
        and isinstance(now.get("tokens_charged_total"), int)
        and now["tokens_charged_total"] >= (earlier.get("tokens_charged_total") or 0)
    )
    return {"previous_file": Path(previous_path).name, "previous": earlier, "current": now,
            "ledger_persisted": persisted}


def render_markdown(evidence: dict) -> str:
    checks = evidence["checks"]
    lines = [
        "# Stage 3 verification evidence", "",
        "Base URL: " + evidence["base_url"], "Recorded: " + evidence["recorded_at"], "",
        "## Checks", "",
    ]
    for name, value in checks.items():
        lines.append("- " + name + ": " + json.dumps(value))
    lines += ["", "## Ledger", ""]
    for label in ("ready_before", "ready_after_safe_phase", "ready_final"):
        if label in evidence:
            lines.append("- " + label + ": " + json.dumps(ledger_view(evidence[label])))
    if evidence.get("ledger_comparison"):
        lines.append("- comparison: " + json.dumps(evidence["ledger_comparison"]))
    if evidence.get("uploads"):
        lines += ["", "## Uploads", ""]
        for upload in evidence["uploads"]:
            body = upload["response"].get("body") if isinstance(upload["response"].get("body"), dict) else {}
            lines.append("- {file}: status {status}, paper_id {paper_id}, pages {pages}, chunks {chunks}, "
                         "matches_expected {match}".format(
                             file=upload["file"], status=upload["response"].get("status"),
                             paper_id=body.get("paper_id"), pages=body.get("pages"),
                             chunks=body.get("chunks"), match=upload.get("matches_expected")))
    if evidence.get("live"):
        lines += ["", "## Live generation", ""]
        for result in evidence["live"]:
            body = result["response"].get("body") if isinstance(result["response"].get("body"), dict) else {}
            lines += [
                "### " + str(result.get("id")),
                "", "Accounted: " + json.dumps(result.get("accounted")) + " "
                + json.dumps(result.get("failure_reasons")),
                "", "Question: " + result["request"]["question"],
                "", "Expected support: " + str(result.get("expected_support")),
                "", "Status: " + str(result["response"].get("status")) + "; model_used: "
                + str(body.get("model_used")) + "; model_usage: " + json.dumps(body.get("model_usage")),
                "", "Ledger delta: " + json.dumps(result["ledger_delta"]),
                "", "Answer:", "", "> " + str(body.get("answer", "")).replace("\n", "\n> "), "",
                "Citations:", "",
            ]
            for citation in body.get("citations") or []:
                lines.append("- {paper} page {page} (score {score}): {text}".format(
                    paper=citation.get("paper"), page=citation.get("page"),
                    score=citation.get("relevance_score"), text=json.dumps(citation.get("text"))))
            lines.append("")
    return "\n".join(lines) + "\n"


def write_evidence(evidence: dict, output_dir: Path, token: str) -> tuple[Path, Path]:
    serialized = json.dumps(evidence, indent=2, sort_keys=True)
    markdown = render_markdown(evidence)
    if token in serialized or token in markdown:
        raise VerificationError("The access token would appear in the evidence; nothing was written.")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = EVIDENCE_FILE_PREFIX + timestamp_slug()
    json_path, md_path = output_dir / (stem + ".json"), output_dir / (stem + ".md")
    json_path.write_text(serialized + "\n", encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="Deployed API origin, e.g. https://example.onrender.com")
    parser.add_argument("--output", default="stage3-evidence", help="Directory for the evidence files")
    parser.add_argument("--manifest", default=None, help="JSON list of fixtures (default: the two public papers)")
    parser.add_argument("--fixture-dir", default="stage3-fixtures", help="Where fixture PDFs are read from")
    parser.add_argument("--download-fixtures", action="store_true",
                        help="Download a missing fixture from its source_url before verifying its digest")
    parser.add_argument("--skip-uploads", action="store_true", help="Do not upload the manifest fixtures")
    parser.add_argument("--live", action="store_true",
                        help="Opt in to bounded real generation (one attempt per question, no retries)")
    parser.add_argument("--max-live-calls", type=int, default=3, choices=range(1, MAX_LIVE_CALLS + 1),
                        help="Upper bound on paid calls in this invocation (default 3, maximum %d)" % MAX_LIVE_CALLS)
    parser.add_argument("--questions", default=None, help="JSON list of {id, question, paper_id?, top_k?}")
    parser.add_argument("--compare-ledger", default=None, metavar="PREVIOUS_EVIDENCE_JSON",
                        help="Assert the ledger identity and counts from an earlier evidence file persisted")
    parser.add_argument("--upload-timeout", type=float, default=300.0)
    parser.add_argument("--query-timeout", type=float, default=120.0)
    return parser


def main(argv=None, environ=None) -> int:
    args = build_parser().parse_args(argv)
    environ = os.environ if environ is None else environ
    try:
        token = token_from_environment(environ)
    except VerificationError as error:
        print("refused:", error, file=sys.stderr)
        return 2
    client = Client(args.base_url, token)
    evidence = {
        "recorded_at": utc_now(), "base_url": client.base_url, "checks": {},
        "live_requested": bool(args.live), "max_live_calls": args.max_live_calls,
        "scope": "Explicit verification of one protected demo with public papers; paid generation only "
                 "with --live; nothing deleted; no retries.",
    }
    exit_code = 0
    try:
        safe_ok = safe_phase(client, args, evidence)
        evidence["checks"]["safe_phase_passed"] = safe_ok
        if not safe_ok:
            exit_code = 1
        if args.live and safe_ok:
            live_phase(client, args, evidence)
            if not evidence["checks"]["live_phase_passed"]:
                exit_code = 1
        evidence["ready_final"] = client.request("GET", "/ready", credential="none", timeout=30)
    except VerificationError as error:
        evidence["aborted"] = str(error)
        exit_code = 1
    try:
        json_path, md_path = write_evidence(evidence, Path(args.output), token)
    except VerificationError as error:
        print("refused:", error, file=sys.stderr)
        return 2
    print("evidence:", json_path, md_path)
    print("checks:", json.dumps(evidence["checks"]))
    if evidence.get("aborted"):
        print("aborted:", evidence["aborted"])
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
