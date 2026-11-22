"""Process-wide network and process-spawn kill switch.

How it works
------------
CPython raises an *audit event* (PEP 578) inside its C implementation every
time Python code creates a socket, connects, resolves a host name, opens a URL,
or spawns a process. `sys.addaudithook` lets us register a hook that sees every
one of those events, and raising an exception from the hook aborts the
operation. Audit hooks:

  * cannot be removed once added (not even by other libraries),
  * fire for the low-level `_socket` C module too, so code that bypasses the
    `socket` Python wrapper is still caught,
  * are installed here BEFORE numpy / OpenCV / ONNX Runtime are imported.

What it does NOT cover (be honest with yourself when reviewing):
  * Native C/C++ code that opens OS sockets directly without going through
    Python (e.g. a compiled library calling connect() itself). Neither OpenCV
    nor ONNX Runtime does this for local files, and aoi_tool never passes them
    a URL, but the only *absolute* guarantee is OS-level isolation - see
    README "Running fully air-gapped".
  * Windows ETW trace events from the official ONNX Runtime Windows wheel are
    switched off separately in onnx_utils.py (ort.disable_telemetry_events()).

Nothing in this module talks to the network; it only imports `socket` to read
the AF_UNIX constant.
"""

import os
import socket
import sys

_INSTALLED = False
_ALLOW_SUBPROCESS = False
_SPAWN_LOG = []


class NetworkBlockedError(PermissionError):
    """Raised whenever any code in this process attempts network access."""


# Audit events that are always blocked (network).
_NETWORK_EVENTS = {
    "socket.bind",
    "socket.connect",
    "socket.getaddrinfo",
    "socket.gethostbyaddr",
    "socket.gethostbyname",
    "socket.getnameinfo",
    "socket.getservbyname",
    "socket.getservbyport",
    "socket.sendmsg",
    "socket.sendto",
    "socket.sethostname",
    "urllib.Request",
    "http.client.connect",
    "http.client.send",
    "ftplib.connect",
    "smtplib.connect",
    "smtplib.send",
    "imaplib.open",
    "poplib.connect",
    "nntplib.connect",
    "telnetlib.Telnet.open",
    "webbrowser.open",
}

# Audit events blocked unless --allow-subprocess is given (defence in depth:
# stops a library from shelling out to curl/wget/pip or opening a browser).
_SPAWN_EVENTS = {
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.posix_spawn",
    "os.spawn",
    "os.startfile",
    "os.fork",
    "os.forkpty",
}

_AF_UNIX = getattr(socket, "AF_UNIX", None)


def _hook(event, args):
    if event == "socket.__new__":
        # args = (socket_obj, family, type, proto). Allow only local
        # Unix-domain sockets (used by nothing in aoi_tool, but harmless).
        family = args[1] if len(args) > 1 else None
        if _AF_UNIX is not None and family == _AF_UNIX:
            return
        raise NetworkBlockedError(
            f"aoi_tool network guard: blocked creation of a network socket "
            f"(family={family}). This program must never access the network.")
    if event in _NETWORK_EVENTS:
        raise NetworkBlockedError(
            f"aoi_tool network guard: blocked '{event}' {args!r:.200}. "
            f"This program must never access the network.")
    if event in _SPAWN_EVENTS:
        if _ALLOW_SUBPROCESS:
            _SPAWN_LOG.append((event, repr(args)[:300]))
            sys.stderr.write(f"[netguard] WARNING process spawn allowed: {event} {args!r:.300}\n")
            return
        raise NetworkBlockedError(
            f"aoi_tool network guard: blocked process spawn '{event}' {args!r:.200}. "
            f"Re-run with --allow-subprocess only if you have reviewed why this happens.")


def install(allow_subprocess=False):
    """Install the guard. Safe to call more than once (later calls are no-ops,
    except that allow_subprocess can only be switched from True to False)."""
    global _INSTALLED, _ALLOW_SUBPROCESS
    if _INSTALLED:
        if not allow_subprocess:
            _ALLOW_SUBPROCESS = False
        return
    _ALLOW_SUBPROCESS = bool(allow_subprocess)

    # Belt and braces for libraries that honour these variables. None of the
    # libraries aoi_tool uses need them, but they cost nothing.
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "YOLO_OFFLINE": "1",
    }.items():
        os.environ[key] = value

    sys.addaudithook(_hook)
    _INSTALLED = True


def is_installed():
    return _INSTALLED


def self_test():
    """Try several ways of reaching the network; every one must be blocked.
    Returns a list of (method, blocked: bool, message)."""
    results = []

    def attempt(name, fn):
        try:
            fn()
            results.append((name, False, "NOT BLOCKED"))
        except NetworkBlockedError as exc:
            results.append((name, True, str(exc)[:120]))
        except Exception as exc:  # any other failure is not proof of blocking
            results.append((name, False, f"unexpected {type(exc).__name__}: {exc}"))

    attempt("socket.socket(AF_INET)", lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM))
    attempt("socket.getaddrinfo", lambda: socket.getaddrinfo("example.com", 443))
    attempt("socket.gethostbyname", lambda: socket.gethostbyname("example.com"))
    attempt("socket.create_connection", lambda: socket.create_connection(("93.184.215.14", 80), timeout=1))

    def _raw_c_socket():
        import _socket  # the C module underneath `socket`
        _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    attempt("_socket.socket (C module)", _raw_c_socket)

    def _urllib():
        import urllib.request
        urllib.request.urlopen("http://example.com", timeout=1)
    attempt("urllib.request.urlopen", _urllib)

    def _http_client():
        import http.client
        http.client.HTTPConnection("example.com", 80, timeout=1).connect()
    attempt("http.client.connect", _http_client)

    if not _ALLOW_SUBPROCESS:
        def _spawn():
            import subprocess
            subprocess.run(["curl", "http://example.com"], check=False)
        attempt("subprocess (e.g. curl)", _spawn)
        attempt("os.system", lambda: os.system("echo hi"))
    return results
