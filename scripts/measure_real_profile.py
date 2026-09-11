"""Bounded, isolated Docker measurements: real local retrieval, no paid inference.

The runtime container has no external network. A placeholder key only constructs
the provider client; queries use a nonexistent paper filter and must abstain.
Fixture hashes come from the reviewed 2026-09-10 public-paper manifest.

Stage 2c: the container runs the protected API. A fixed fake access token and a
one-call budget with a throwaway ledger are passed as environment variables so
the measurement exercises authentication, limits and accounting. The token below
is a test fixture, not a credential, and no model call is ever attempted.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


MEASUREMENT_TOKEN = "measurement-only-fake-access-token-0123456789abcdef"
MEASUREMENT_ENVIRONMENT = (
    "DEMO_ACCESS_TOKEN=" + MEASUREMENT_TOKEN,
    "ALLOWED_ORIGINS=",
    # Inside a root-owned, non-writable (0755) tmpfs mount, so the measurement
    # proves the entrypoint hands the ledger directory to the runtime user.
    "MODEL_CALL_LEDGER_PATH=/var/data/ledger/model-calls.sqlite3",
    "MAX_MODEL_CALLS_PER_DAY=1", "MAX_MODEL_CALLS_TOTAL=1",
    "MAX_MODEL_TOKENS_PER_DAY=2000", "MAX_MODEL_TOKENS_TOTAL=2000",
)

FIXTURES = (
    {
        "file": "attention-is-all-you-need.pdf",
        "source_url": "https://arxiv.org/pdf/1706.03762",
        "sha256": "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697",
        "pages": 15,
        "chunks": 97,
    },
    {
        "file": "retrieval-augmented-generation.pdf",
        "source_url": "https://arxiv.org/pdf/2005.11401",
        "sha256": "23e3249e9a1e75418d82efecab0ea8c4d033b89c93742f63208d47ce01f21233",
        "pages": 19,
        "chunks": 171,
    },
)

METRICS_HELPER = r'''
import json
from pathlib import Path
result = {}
for field in ("memory.current", "memory.peak", "memory.max", "memory.events", "cpu.max"):
    path = Path("/sys/fs/cgroup") / field
    result[field] = path.read_text().strip() if path.exists() else None
status = Path("/proc/1/status").read_text().splitlines()
result["pid1_memory_kib"] = {
    line.split(":", 1)[0]: int(line.split()[1])
    for line in status if line.startswith(("VmRSS:", "VmHWM:"))
}
result["pid1_uid"] = next((int(line.split()[1]) for line in status if line.startswith("Uid:")), None)
result["pid1_cap_eff"] = next((line.split()[1] for line in status if line.startswith("CapEff:")), None)
result["pid1_argv"] = [part.decode("utf-8", errors="replace")
    for part in Path("/proc/1/cmdline").read_bytes().split(b"\0") if part]
print(json.dumps(result))
'''

HTTP_HELPER = r'''
import json, os, sys, urllib.error, urllib.request, uuid
from pathlib import Path
method, path, timeout, payload, credential = sys.argv[1:]
headers = {}
if credential == "configured":
    headers["Authorization"] = "Bearer " + os.environ["DEMO_ACCESS_TOKEN"]
elif credential == "wrong":
    headers["Authorization"] = "Bearer wrong-measurement-token-0123456789abcdef0123"
data = None
if method == "UPLOAD":
    boundary = "rag-fixture-" + uuid.uuid4().hex
    filename = Path(payload).name
    data = (
        ("--" + boundary + '\r\nContent-Disposition: form-data; name="file"; filename="'
         + filename + '"\r\nContent-Type: application/pdf\r\n\r\n').encode()
        + Path(payload).read_bytes() + ("\r\n--" + boundary + "--\r\n").encode()
    )
    headers["Content-Type"] = "multipart/form-data; boundary=" + boundary
    method = "POST"
elif payload:
    data = payload.encode()
    headers["Content-Type"] = "application/json"
request = urllib.request.Request("http://127.0.0.1:" + os.environ["PORT"] + path,
                                 data=data, headers=headers, method=method)
try:
    with urllib.request.urlopen(request, timeout=float(timeout)) as response:
        status, body = response.status, response.read().decode()
except urllib.error.HTTPError as error:
    status, body = error.code, error.read().decode()
try:
    body = json.loads(body)
except ValueError:
    pass
print(json.dumps({"status": status, "body": body}))
'''


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def command(args, timeout=30, check=True, log_path=None):
    if log_path:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        output = "See " + str(log_path)
    else:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        output = result.stdout.strip()
    if check and result.returncode:
        detail = output if log_path else (result.stderr or output)[-2000:]
        raise RuntimeError(f"{args[0]} {args[1]} failed ({result.returncode}): {detail}")
    return result.returncode, output


def container_state(name, timeout=30):
    code, output = command(["docker", "inspect", "--format", "{{json .State}}", name], timeout=timeout, check=False)
    return json.loads(output) if code == 0 else {"Running": False, "inspect_unavailable": True}


def metrics(name, timeout=20):
    code, output = command(["docker", "exec", name, "python", "-c", METRICS_HELPER], timeout=timeout, check=False)
    if code:
        return {"unavailable": True}
    return {"captured_at": utc_now(), **json.loads(output)}


def request(name, method, path, payload="", timeout=180, execution_timeout=None, credential="configured"):
    _, output = command(
        ["docker", "exec", name, "python", "-c", HTTP_HELPER, method, path, str(timeout), payload, credential],
        timeout=execution_timeout if execution_timeout is not None else timeout + 20,
    )
    return json.loads(output)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def download_fixtures(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for fixture in FIXTURES:
        target = directory / fixture["file"]
        if not target.exists():
            request_object = urllib.request.Request(
                fixture["source_url"], headers={"User-Agent": "rag-real-profile-fixture-verification/1.0"}
            )
            with urllib.request.urlopen(request_object, timeout=60) as response:
                data = response.read(20 * 1024 * 1024 + 1)
            require(len(data) <= 20 * 1024 * 1024, "Fixture download exceeded 20 MiB")
            target.write_bytes(data)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        require(digest == fixture["sha256"], f"Fixture checksum changed: {fixture['file']}; do not substitute revisions")
    write_json(directory / "manifest.json", list(FIXTURES))


def measure_case(image, memory, cpus, directory, fixture_dir, ready_timeout):
    name = "rag-measure-" + uuid.uuid4().hex[:12]
    report = {
        "started_at": utc_now(), "image": image,
        "limits": {"memory": memory, "memory_swap_total": memory, "cpus": cpus, "network": "none"},
        "startup_contract": {"PORT": 8765, "WEB_CONCURRENCY": 4, "expected_workers": 1},
        "status": "failed", "snapshots": {}, "requests": {},
        "scope": "Local embeddings and Chroma behind the protected API; provider client construction only; "
                 "no generation, accounting consumption or relevance evaluation.",
        "protection": {"access_token": "fixed fake measurement token", "model_call_allowance": 1,
                       "ledger": "throwaway file in a root-owned 0755 tmpfs mount at /var/data"},
        "measurement_note": "cgroup totals include temporary docker-exec HTTP/observer processes; PID 1 RSS is app-process memory.",
    }
    path = directory / (memory + ".json")
    try:
        started = time.monotonic()
        command([
            "docker", "run", "--detach", "--name", name,
            "--network", "none", "--memory", memory, "--memory-swap", memory, "--cpus", cpus,
            # A root-owned, non-world-writable mount like a platform disk: the
            # ledger is only creatable there if the entrypoint prepared the directory.
            "--mount", "type=tmpfs,destination=/var/data,tmpfs-mode=0755,tmpfs-size=16m",
            "--env", "LLM_PROVIDER=openai", "--env", "OPENAI_API_KEY=offline-placeholder-not-a-credential",
            "--env", "LLM_MODEL=gpt-4o-mini", "--env", "PORT=8765", "--env", "WEB_CONCURRENCY=4",
            "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
            "--env", "HF_HUB_DISABLE_TELEMETRY=1", "--env", "ANONYMIZED_TELEMETRY=FALSE",
            *[flag for variable in MEASUREMENT_ENVIRONMENT for flag in ("--env", variable)],
            image,
        ])
        deadline = started + ready_timeout
        last_probe = None
        while time.monotonic() < deadline:
            state = container_state(name, timeout=min(10, max(0.1, deadline - time.monotonic())))
            require(state.get("Running"), "Container exited before readiness")
            try:
                snapshot = metrics(name, timeout=min(20, max(0.1, deadline - time.monotonic())))
                if not snapshot.get("unavailable"):
                    report["snapshots"]["last_startup"] = snapshot
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                last_probe = request(name, "GET", "/ready", timeout=min(2, remaining),
                                     execution_timeout=min(22, remaining))
                if last_probe["status"] == 200:
                    break
                if last_probe["status"] == 503:
                    raise RuntimeError("Application initialized as unavailable: " + json.dumps(last_probe))
            except (subprocess.TimeoutExpired, RuntimeError) as error:
                if last_probe and last_probe.get("status") == 503:
                    raise
                report["last_startup_probe_error"] = str(error)[-1000:]
            time.sleep(min(3, max(0, deadline - time.monotonic())))
        require(last_probe is not None and last_probe.get("status") == 200,
                f"Readiness was not achieved within {ready_timeout} seconds")
        report["time_to_ready_seconds"] = round(time.monotonic() - started, 3)
        report["ready_response"] = last_probe
        ready = last_probe["body"]
        require(ready.get("ready") is True, "Readiness response is false")
        require(ready.get("effective_retrieval") == "chroma", "Real retrieval did not initialize")
        require(ready.get("effective_generation") == "openai", "Real provider client did not initialize")
        require(ready.get("provider_connection_verified") is False, "Unexpected remote verification claim")
        require(ready.get("access_configured") is True, "Access token was not reported as configured")
        budget = ready.get("model_budget") or {}
        require(budget.get("state") == "ok" and budget.get("configured") is True, "Model-call accounting is not ready")
        require((budget.get("usage") or {}).get("calls_total") == 0, "Ledger already holds calls before any query")
        require(budget.get("token_bound") == "tiktoken/o200k_base",
                "The offline tiktoken bound for gpt-4o-mini did not resolve from the build-time cache")
        require(MEASUREMENT_TOKEN not in json.dumps(ready), "Readiness must not expose the access token")
        report["snapshots"]["idle"] = metrics(name)
        require(all(report["snapshots"]["idle"].get(field) is not None
                    for field in ("memory.current", "memory.peak", "memory.events", "pid1_memory_kib")),
                "Required cgroup-v2/app memory evidence is unavailable")
        argv = report["snapshots"]["idle"]["pid1_argv"]
        require("--port" in argv and argv[argv.index("--port") + 1] == "8765",
                "PID 1 did not honor the nondefault PORT=8765")
        require("--workers" in argv and argv[argv.index("--workers") + 1] == "1",
                "PID 1 must explicitly keep one worker despite WEB_CONCURRENCY=4")
        idle = report["snapshots"]["idle"]
        require(idle.get("pid1_uid") == 10001 and idle.get("pid1_cap_eff") == "0000000000000000",
                "PID 1 must run as the unprivileged runtime user with no effective capabilities")
        for label, credential in (("unauthenticated_papers", "none"), ("wrong_token_papers", "wrong")):
            denied = request(name, "GET", "/papers", credential=credential)
            report["requests"][label] = denied
            require(denied["status"] == 401, "Protected route answered without a valid token: " + label)
            require(MEASUREMENT_TOKEN not in json.dumps(denied), "A rejection must not expose the access token")
        command(["docker", "exec", name, "mkdir", "-p", "/tmp/rag-fixtures"])
        uploads = []
        for fixture in FIXTURES:
            container_path = "/tmp/rag-fixtures/" + fixture["file"]
            command(["docker", "cp", str(fixture_dir / fixture["file"]), name + ":" + container_path])
            response = request(name, "UPLOAD", "/papers/upload", container_path, timeout=240)
            report["requests"][fixture["file"]] = response
            require(response["status"] == 200, "Fixture upload failed: " + fixture["file"])
            body = response["body"]
            require(body["paper_id"] == fixture["sha256"], "Unexpected full-document identity")
            require(body["pages"] == fixture["pages"] and body["chunks"] == fixture["chunks"], "Fixture extraction/chunk counts changed")
            uploads.append(body)
            report["snapshots"]["after_" + fixture["file"]] = metrics(name)
            write_json(path, report)
        duplicate_path = "/tmp/rag-fixtures/renamed-attention.pdf"
        command(["docker", "cp", str(fixture_dir / FIXTURES[0]["file"]), name + ":" + duplicate_path])
        duplicate = request(name, "UPLOAD", "/papers/upload", duplicate_path)
        report["requests"]["duplicate"] = duplicate
        require(duplicate["status"] == 200, "Duplicate upload failed")
        for key in ("paper_id", "filename", "pages", "chunks"):
            require(duplicate["body"][key] == uploads[0][key], "Duplicate changed canonical " + key)
        listed = request(name, "GET", "/papers")
        report["requests"]["papers"] = listed
        require(listed["status"] == 200 and len(listed["body"]) == 2, "Duplicate modified the paper registry")
        empty_query = request(name, "POST", "/query", json.dumps({
            "question": "What is the attention mechanism?", "paper_id": "nonexistent-offline-test-paper", "top_k": 3,
        }))
        report["requests"]["empty_filter_query"] = empty_query
        require(empty_query["status"] == 200, "Offline real retrieval failed")
        require(empty_query["body"]["citations"] == [], "Nonexistent filter returned citations")
        require(empty_query["body"]["model_used"] == "not-invoked", "Query attempted generation instead of abstaining")
        require(empty_query["body"].get("model_usage") is None, "Abstention must not report model usage")
        final_ready = request(name, "GET", "/ready")
        report["requests"]["ready_after_query"] = final_ready
        require(final_ready["status"] == 200, "Readiness degraded after the offline query")
        require(final_ready["body"]["model_budget"]["usage"]["calls_total"] == 0,
                "The abstaining query must not consume the model-call allowance")
        report["snapshots"]["after_query"] = metrics(name)
        require(report["snapshots"]["after_query"].get("memory.peak") is not None,
                "Post-ingestion peak memory evidence is unavailable")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)[-3000:]
    finally:
        # Inspect before removing the container, preserving OOM/exit evidence.
        try:
            report["container_state_before_cleanup"] = container_state(name)
            if report["container_state_before_cleanup"].get("Running"):
                report["snapshots"]["final"] = metrics(name)
            command(["docker", "logs", name], check=False, log_path=directory / (memory + "-container.log"))
            oom_kills = []
            for snapshot in report["snapshots"].values():
                events = dict(line.split() for line in (snapshot.get("memory.events") or "").splitlines())
                oom_kills.append(int(events.get("oom_kill", "0")))
            report["observed_oom_kill_count"] = max(oom_kills, default=0)
            if report["observed_oom_kill_count"] or report["container_state_before_cleanup"].get("OOMKilled"):
                report["status"] = "failed"
                report.setdefault("error", "An OOM kill occurred under these limits")
        except Exception as error:
            report["collection_error"] = str(error)[-1000:]
        report["finished_at"] = utc_now()
        write_json(path, report)
        command(["docker", "rm", "--force", name], check=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="rag-real-profile:measurement")
    parser.add_argument("--output", type=Path, default=Path("measurement-artifacts"))
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--ready-timeout", type=int, default=300)
    args = parser.parse_args()
    require(1 <= args.ready_timeout <= 300, "Readiness timeout must be between 1 and 300 seconds")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {"started_at": utc_now(), "status": "failed", "cases": [], "real_generation_verified": False,
               "semantic_relevance_verified": False, "external_runtime_network": "disabled"}
    exit_code = 1
    try:
        _, summary["commit"] = command(["git", "rev-parse", "HEAD"])
        _, summary["docker_platform"] = command(["docker", "info", "--format", "{{.OSType}}/{{.Architecture}}"])
        require(summary["docker_platform"] in ("linux/x86_64", "linux/amd64"),
                "This pinned CPU profile requires a Linux x86-64 Docker engine")
        if not args.skip_build:
            command(["docker", "build", "--file", "Dockerfile.real", "--target", "runtime", "--tag", args.image, "."],
                    timeout=1800, log_path=output / "build.log")
        _, image_info = command(["docker", "image", "inspect", "--format", "{{.Id}} {{.Size}}", args.image])
        image_id, image_bytes = image_info.split()
        summary.update({"image_id": image_id, "image_bytes": int(image_bytes)})
        fixture_dir = (args.fixture_dir or output / "fixtures").resolve()
        download_fixtures(fixture_dir)
        summary["fixtures"] = list(FIXTURES)
        for memory, cpus in (("512m", "0.1"), ("2g", "1")):
            case = measure_case(args.image, memory, cpus, output, fixture_dir, args.ready_timeout)
            summary["cases"].append(case)
            write_json(output / "summary.json", summary)
            if memory == "2g" and case["status"] == "passed":
                summary["status"] = "passed"
                summary["passing_limits"] = case["limits"]
                exit_code = 0
        summary["comparison_limit"] = "Fallback changes both memory and CPU; this is not an equal-CPU capacity comparison or a Render measurement."
    except Exception as error:
        summary["error"] = type(error).__name__ + ": " + str(error)[-3000:]
    finally:
        summary["finished_at"] = utc_now()
        write_json(output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "summary": str(output / "summary.json")}))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
