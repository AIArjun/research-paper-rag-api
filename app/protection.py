"""
Shared-demo protection
======================
Pure ASGI guards and admission slots for one public-paper demo served by one
worker:

- AccessTokenMiddleware rejects requests without the shared bearer token
  before any body is read, using a constant-time comparison. Public pages and
  readiness stay open and never expose the token.
- RequestBodyLimitMiddleware bounds total request bytes at the receive
  boundary, so a missing or misleading Content-Length cannot bypass the cap.
- AdmissionSlot/Admission bound concurrent work. A slot transferred to a
  worker thread stays held until that thread finishes, even if the awaiting
  request is cancelled; a thread cannot be interrupted, so the slot must not
  be freed early either.
- AdmissionMiddleware admits an upload before a single body byte is received,
  because FastAPI parses and spools a multipart body before any route code
  runs. A refused upload gets 429 without receive() ever being called.
"""

import hmac
import json
import threading
import uuid
from functools import partial
from typing import Callable, Optional

from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from app.config import access_token_is_valid

PUBLIC_PATHS = frozenset({
    "/", "/docs", "/redoc", "/openapi.json", "/health", "/ready", "/docs/oauth2-redirect",
})
CHALLENGE = 'Bearer realm="research-paper-rag-demo"'
INVALID_TOKEN_CHALLENGE = CHALLENGE + ', error="invalid_token"'


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def presented_token_matches(authorization: Optional[bytes], configured: str) -> bool:
    """Constant-time bearer comparison; a malformed header simply does not match."""
    if not authorization or not access_token_is_valid(configured):
        return False
    scheme, _, credentials = authorization.strip().partition(b" ")
    if scheme.lower() != b"bearer":
        return False
    credentials = credentials.strip()
    if not credentials:
        return False
    return hmac.compare_digest(credentials, configured.encode("utf-8"))


def _header(scope: dict, name: bytes) -> Optional[bytes]:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return value
    return None


async def _send_json(send, status: int, payload: dict, headers: Optional[list] = None) -> None:
    body = json.dumps(payload).encode("utf-8")
    response_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        (b"cache-control", b"no-store"),
    ] + list(headers or [])
    await send({"type": "http.response.start", "status": status, "headers": response_headers})
    await send({"type": "http.response.body", "body": body, "more_body": False})


def _with_request_id(send, request_id: str):
    async def sender(message):
        if message["type"] == "http.response.start":
            headers = list(message.get("headers") or [])
            headers.append((b"x-request-id", request_id.encode("ascii")))
            message = {**message, "headers": headers}
        await send(message)

    return sender


class AccessTokenMiddleware:
    """Require `Authorization: Bearer <DEMO_ACCESS_TOKEN>` on every non-public path."""

    def __init__(self, app, configured_token: Callable[[], object], public_paths=PUBLIC_PATHS):
        self.app = app
        self._configured_token = configured_token
        self._public_paths = frozenset(public_paths)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = new_request_id()
        scope.setdefault("state", {})["request_id"] = request_id
        send = _with_request_id(send, request_id)
        if scope.get("path") in self._public_paths:
            await self.app(scope, receive, send)
            return
        configured = self._configured_token()
        if not access_token_is_valid(configured):
            # The token itself is never rendered; the category is the whole message.
            await _send_json(send, 503, {"detail": {
                "message": "This demo has no valid access token configured, so protected routes are disabled.",
                "category": "access_not_configured",
                "request_id": request_id,
            }})
            return
        authorization = _header(scope, b"authorization")
        if not presented_token_matches(authorization, configured):
            challenge = CHALLENGE if authorization is None else INVALID_TOKEN_CHALLENGE
            await _send_json(send, 401, {"detail": {
                "message": "A valid bearer access token is required.",
                "category": "unauthorized",
                "request_id": request_id,
            }}, headers=[(b"www-authenticate", challenge.encode("ascii"))])
            return
        await self.app(scope, receive, send)


class RequestEntityTooLarge(HTTPException):
    """Raised from the receive boundary; FastAPI re-raises HTTPException unchanged."""

    def __init__(self, limit: int, request_id: Optional[str]):
        super().__init__(status_code=413, detail={
            "message": "The request body exceeds the configured limit.",
            "category": "request_too_large",
            "limit_bytes": limit,
            "request_id": request_id,
        })


class RequestBodyLimitMiddleware:
    """Reject bodies above the per-route limit without trusting Content-Length.

    When limits cannot be computed (malformed configuration) every route gets
    the smallest cap; readiness then reports invalid_configuration itself.
    """

    def __init__(self, app, limit_for_scope: Callable[[dict], int], fallback_limit: int = 16 * 1024):
        self.app = app
        self._limit_for_scope = limit_for_scope
        self._fallback_limit = fallback_limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = (scope.get("state") or {}).get("request_id")
        try:
            limit = self._limit_for_scope(scope)
        except ValueError:
            limit = self._fallback_limit
        declared = _header(scope, b"content-length")
        if declared is not None:
            if not declared.isdigit():
                await _send_json(send, 400, {"detail": {
                    "message": "Invalid Content-Length header.",
                    "category": "invalid_request",
                    "request_id": request_id,
                }})
                return
            if int(declared) > limit:
                await _send_json(send, 413, {"detail": RequestEntityTooLarge(limit, request_id).detail})
                return

        received = 0
        response_started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise RequestEntityTooLarge(limit, request_id)
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except RequestEntityTooLarge as error:
            # Normally FastAPI's exception middleware has already answered; this
            # covers a body read outside that scope. Never answer twice.
            if response_started:
                raise
            await _send_json(send, 413, {"detail": error.detail})


class AdmissionMiddleware:
    """Admit matching requests into a slot before the body is received.

    The admission travels to the route in scope["state"]["admission"]; the
    route transfers it to its worker thread. Whichever side finishes last
    releases the slot: this middleware's release is a no-op once a worker is
    running, and it is what frees the slot after a parse failure, an oversize
    body, a validation error, a disconnect or a cancellation before dispatch.
    """

    def __init__(self, app, slot: Callable[[], "AdmissionSlot"], matches: Callable[[dict], bool],
                 retry_after: int = 5):
        self.app = app
        self._slot = slot
        self._matches = matches
        self._retry_after = retry_after

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self._matches(scope):
            await self.app(scope, receive, send)
            return
        request_id = (scope.get("state") or {}).get("request_id")
        admission = self._slot().admit()
        if admission is None:
            await _send_json(send, 429, {"detail": {
                "message": "The demo is busy with another request. Retry shortly.",
                "category": "busy",
                "request_id": request_id,
            }}, headers=[(b"retry-after", str(self._retry_after).encode("ascii"))])
            return
        scope.setdefault("state", {})["admission"] = admission
        try:
            await self.app(scope, receive, send)
        finally:
            admission.release()


class AdmissionSlot:
    """A fixed number of concurrently admitted units of work."""

    def __init__(self, capacity: int, name: str = "slot"):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Slot capacity must be a positive integer.")
        self.capacity = capacity
        self.name = name
        self._lock = threading.Lock()
        self._in_use = 0

    @property
    def in_use(self) -> int:
        with self._lock:
            return self._in_use

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._in_use >= self.capacity

    def admit(self) -> Optional["Admission"]:
        """Non-blocking; None means the caller should answer 429 immediately."""
        with self._lock:
            if self._in_use >= self.capacity:
                return None
            self._in_use += 1
        return Admission(self)

    def _release(self) -> None:
        with self._lock:
            if self._in_use > 0:
                self._in_use -= 1


class Admission:
    """One admitted unit of work whose slot is released exactly once.

    Ownership moves to the worker thread when run() dispatches it. If the
    thread never starts (cancelled before dispatch), the awaiting side
    releases; once the thread is running, only the thread releases, however
    the awaiting request ends.
    """

    PENDING, TRANSFERRED, RUNNING, RELEASED = "pending", "transferred", "running", "released"

    def __init__(self, slot: AdmissionSlot):
        self._slot = slot
        self._state = self.PENDING
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def release(self) -> None:
        """Release only while the awaiting side still owns the slot."""
        with self._lock:
            owned = self._state in (self.PENDING, self.TRANSFERRED)
            if owned:
                self._state = self.RELEASED
        if owned:
            self._slot._release()

    async def run(self, fn: Callable, *args, **kwargs):
        """Run fn in a worker thread; the slot outlives any cancellation of this await."""
        with self._lock:
            if self._state != self.PENDING:
                raise RuntimeError("This admission was already used.")
            self._state = self.TRANSFERRED
        try:
            return await run_in_threadpool(self._guarded, partial(fn, *args, **kwargs))
        finally:
            self.release()

    def _guarded(self, work: Callable):
        with self._lock:
            if self._state != self.TRANSFERRED:
                return None  # abandoned before the thread started; nothing to run
            self._state = self.RUNNING
        try:
            return work()
        finally:
            with self._lock:
                self._state = self.RELEASED
            self._slot._release()
