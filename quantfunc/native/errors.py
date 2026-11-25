"""Typed QuantFunc exception hierarchy, C-API status mapping, and console-safe output."""

from __future__ import annotations

import sys

from .capi import (
    QF_ERR_AUTH,
    QF_ERR_BACKEND,
    QF_ERR_BUSY,
    QF_ERR_INTERNAL,
    QF_ERR_INVALID_ARG,
    QF_ERR_IO,
    QF_ERR_OOM,
    QF_ERR_UNSUPPORTED,
    QF_OK,
)


def console_safe(text: str) -> str:
    """Escape characters that active stdout/stderr code pages cannot encode."""
    out = str(text)
    for stream in (sys.stdout, sys.stderr):
        enc = getattr(stream, "encoding", None)
        if enc:
            try:
                out = out.encode(enc, "backslashreplace").decode(enc)
            except LookupError:
                out = out.encode("ascii", "backslashreplace").decode("ascii")
    return out


class QuantFuncError(RuntimeError):
    """Base exception for QuantFunc native engine errors."""

    default_status_code: int | None = None

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: str = "",
    ) -> None:
        self.status_code: int | None = self.default_status_code if status_code is None else int(status_code)
        self.detail: str = console_safe(detail) if detail else ""
        super().__init__(console_safe(message))


class InvalidArgError(QuantFuncError):
    """Status 1 (QF_ERR_INVALID_ARG): invalid argument, config, or bounds."""

    default_status_code = QF_ERR_INVALID_ARG


class BackendError(QuantFuncError):
    """Status 2 (QF_ERR_BACKEND): device runtime or kernel backend failure."""

    default_status_code = QF_ERR_BACKEND


class OutOfMemoryError(QuantFuncError):
    """Status 3 (QF_ERR_OOM): insufficient device or host memory."""

    default_status_code = QF_ERR_OOM


class EngineIOError(QuantFuncError):
    """Status 4 (QF_ERR_IO): checkpoint, plugin, or file I/O failure."""

    default_status_code = QF_ERR_IO


class InternalEngineError(QuantFuncError):
    """Status 5 (QF_ERR_INTERNAL): unexpected native engine internal failure."""

    default_status_code = QF_ERR_INTERNAL


class EngineBusyError(QuantFuncError):
    """Status 6 (QF_ERR_BUSY): engine session or resource is currently busy."""

    default_status_code = QF_ERR_BUSY


class AuthError(QuantFuncError):
    """Status 7 (QF_ERR_AUTH): API key missing, invalid, or rejected."""

    default_status_code = QF_ERR_AUTH


class UnsupportedError(QuantFuncError):
    """Status 8 (QF_ERR_UNSUPPORTED): unsupported architecture, GPU, or configuration."""

    default_status_code = QF_ERR_UNSUPPORTED


class EngineNotInstallable(RuntimeError):
    """Raised when the current machine/platform cannot install a published engine build."""


STATUS_EXCEPTION_MAP: dict[int, type[QuantFuncError]] = {
    QF_ERR_INVALID_ARG: InvalidArgError,
    QF_ERR_BACKEND: BackendError,
    QF_ERR_OOM: OutOfMemoryError,
    QF_ERR_IO: EngineIOError,
    QF_ERR_INTERNAL: InternalEngineError,
    QF_ERR_BUSY: EngineBusyError,
    QF_ERR_AUTH: AuthError,
    QF_ERR_UNSUPPORTED: UnsupportedError,
}


def status_to_exception(
    status: int,
    context: str = "",
    detail: str = "",
) -> QuantFuncError | None:
    """Convert a qf_status_t integer to the corresponding QuantFuncError subclass (or None on QF_OK)."""
    code = int(status)
    if code == QF_OK:
        return None
    exc_cls = STATUS_EXCEPTION_MAP.get(code, QuantFuncError)
    clean_detail = console_safe(detail) if detail else "(none)"
    prefix = f"{context} failed" if context else "QuantFunc native call failed"
    message = f"{prefix} (status={code}): {clean_detail}"
    return exc_cls(message, status_code=code, detail=clean_detail)


def raise_for_status(status: int, context: str = "", detail: str = "") -> None:
    """Raise the mapped QuantFuncError if status != QF_OK."""
    exc = status_to_exception(status, context=context, detail=detail)
    if exc is not None:
        raise exc


__all__ = [
    "QuantFuncError",
    "InvalidArgError",
    "BackendError",
    "OutOfMemoryError",
    "EngineIOError",
    "InternalEngineError",
    "EngineBusyError",
    "AuthError",
    "UnsupportedError",
    "EngineNotInstallable",
    "STATUS_EXCEPTION_MAP",
    "status_to_exception",
    "raise_for_status",
    "console_safe",
]
