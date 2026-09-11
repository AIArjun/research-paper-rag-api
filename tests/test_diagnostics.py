"""Diagnostics retain useful fixed labels without exposing exception content."""

import logging
from types import SimpleNamespace

import pytest

from app.diagnostics import log_safe_error, safe_error_details


def test_builtin_exception_logs_only_safe_labels_without_traceback(caplog):
    secret = "sk-example-secret https://private.example/path?token=secret request-body"
    try:
        raise ValueError(secret)
    except ValueError as error:
        log_safe_error(logging.getLogger("test.diagnostics"), "invalid_configuration", error)
    assert "category=invalid_configuration" in caplog.text
    assert "exception_type=ValueError" in caplog.text
    assert "exception_module=builtins" in caplog.text
    assert secret not in caplog.text
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None and record.stack_info is None for record in caplog.records)
    assert all(not isinstance(arg, BaseException) for record in caplog.records for arg in record.args)


def test_diagnostics_never_render_exception_text_or_probe_request_data(caplog):
    class DangerousError(Exception):
        def __str__(self):
            pytest.fail("Never stringify the exception")

        @property
        def request(self):
            pytest.fail("Never inspect the request")

        @property
        def body(self):
            pytest.fail("Never inspect the response body")

    error = DangerousError()
    log_safe_error(logging.getLogger("test.diagnostics"), "generation_failed", error)
    assert "exception_type=unknown" in caplog.text
    assert "exception_module=unknown" in caplog.text


def test_allowlisted_sdk_exception_and_status_are_preserved_without_payloads():
    authentication_error = type("AuthenticationError", (Exception,), {"__module__": "openai._exceptions"})
    error = authentication_error("private provider response")
    error.status_code = 401
    error.body = {"key": "private-key", "request": "private-query"}
    error.request = SimpleNamespace(url="https://private.example?token=private-key")
    assert safe_error_details("generation_failed", error) == {
        "category": "generation_failed",
        "exception_type": "AuthenticationError",
        "exception_module": "openai",
        "dependency_module": "none",
        "status_code": 401,
    }


def test_response_http_status_is_read_without_url_or_body():
    class Response:
        status_code = 429

        @property
        def url(self):
            pytest.fail("Never inspect URLs")

        @property
        def text(self):
            pytest.fail("Never inspect response text")

        @property
        def headers(self):
            pytest.fail("Never inspect response headers")

    status_error = type("HTTPStatusError", (Exception,), {"__module__": "httpx"})
    error = status_error("private body and URL")
    error.response = Response()
    details = safe_error_details("generation_failed", error)
    assert details["status_code"] == 429
    assert details["exception_type"] == "HTTPStatusError"
    assert details["exception_module"] == "httpx"


@pytest.mark.parametrize("status", ["401", 401.0, True, 999, 200, {"secret": "value"}])
def test_unapproved_or_noninteger_status_values_are_not_logged(status):
    error = RuntimeError("private payload")
    error.status_code = status
    assert safe_error_details("generation_failed", error)["status_code"] is None


def test_hostile_diagnostic_attributes_cannot_replace_the_original_error():
    class HostileAttributes(Exception):
        @property
        def status_code(self):
            raise RuntimeError("private attribute error")

        @property
        def response(self):
            raise RuntimeError("another private attribute error")

    details = safe_error_details("generation_failed", HostileAttributes())
    assert details["status_code"] is None
    assert details["category"] == "generation_failed"


@pytest.mark.parametrize(
    "missing_name,expected", [("langchain_chroma", "langchain_chroma"), ("torch._C", "torch"), ("private.secret", "unknown")]
)
def test_missing_dependency_name_is_reduced_to_an_allowlisted_root(missing_name, expected):
    error = ModuleNotFoundError("private local path and credentials", name=missing_name)
    details = safe_error_details("missing_dependency", error)
    assert details["exception_type"] == "ModuleNotFoundError"
    assert details["dependency_module"] == expected
    assert "private local path" not in str(details)


def test_untrusted_class_module_and_category_names_are_not_echoed():
    private_error = type("private-secret-class", (Exception,), {"__module__": "private.secret.module"})
    details = safe_error_details("private-secret-category", private_error("private message"))
    assert details["category"] == "unknown_error"
    assert details["exception_type"] == "unknown"
    assert details["exception_module"] == "unknown"
    assert "private" not in str(details)


def test_missing_key_diagnostic_requires_no_exception_or_secret():
    assert safe_error_details("missing_api_key") == {
        "category": "missing_api_key",
        "exception_type": "none",
        "exception_module": "none",
        "dependency_module": "none",
        "status_code": None,
    }
