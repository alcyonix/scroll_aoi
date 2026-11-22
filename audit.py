"""Static audit: scan source files for anything that could reach the network,
spawn processes, or execute dynamic code. Uses Python's `ast` (no execution).

Run:  python -m aoi_tool audit            (this project)
      python -m aoi_tool audit --deps     (also list dependency files that
                                           import network modules, for review)
"""

import ast
import os
import re

NETWORK_MODULES = {
    "socket", "_socket", "ssl", "urllib", "urllib2", "urllib3", "http", "httplib",
    "requests", "httpx", "aiohttp", "ftplib", "smtplib", "poplib", "imaplib",
    "telnetlib", "nntplib", "xmlrpc", "websocket", "websockets", "grpc",
    "asyncio", "selectors", "socketserver", "webbrowser", "paramiko",
    "huggingface_hub", "torch.hub", "boto3", "google.cloud",
}
PROCESS_MODULES = {"subprocess", "multiprocessing", "pty", "ctypes"}
DYNAMIC_CALLS = {"eval", "exec", "compile", "__import__"}
RISKY_ATTRS = {("os", "system"), ("os", "popen"), ("os", "startfile"), ("pickle", "load"),
               ("pickle", "loads"), ("yaml", "load"), ("yaml", "unsafe_load"),
               ("importlib", "import_module")}
URL_RE = re.compile(r"\b(?:https?|ftp|wss?|rtsp)://", re.IGNORECASE)

# Documented exceptions inside THIS project, each with its reason.
ALLOWED = {
    ("netguard.py", "socket"): "imported only to read the AF_UNIX constant for the guard",
    ("netguard.py", "_socket"): "self_test(): proves the raw C socket module is blocked",
    ("netguard.py", "urllib"): "self_test(): proves urllib is blocked",
    ("netguard.py", "http"): "self_test(): proves http.client is blocked",
    ("netguard.py", "subprocess"): "self_test(): proves process spawning is blocked",
    ("netguard.py", "os.system"): "self_test(): proves os.system is blocked",
    ("netguard.py", "http://example.com"): "self_test(): target URL that must be blocked",
    ("audit.py", "http://example.com"): "text of the allowlist entry above (data, not a call)",
}


def _scan_file(path, project_mode):
    findings = []
    try:
        src = open(path, "r", encoding="utf-8", errors="replace").read()
        tree = ast.parse(src, filename=path)
    except (SyntaxError, ValueError) as exc:
        return [("parse-error", 0, str(exc), None)]
    base = os.path.basename(path)

    def add(kind, node, what):
        reason = ALLOWED.get((base, what)) if project_mode else None
        findings.append((kind, getattr(node, "lineno", 0), what, reason))

    # Docstrings / bare string statements are documentation and cannot execute.
    doc_ids = {id(n.value) for n in ast.walk(tree)
               if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
               and isinstance(n.value.value, str)}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if a.name in NETWORK_MODULES or top in NETWORK_MODULES:
                    add("network-import", node, top)
                elif top in PROCESS_MODULES:
                    add("process-import", node, top)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if node.module in NETWORK_MODULES or top in NETWORK_MODULES:
                add("network-import", node, top)
            elif top in PROCESS_MODULES:
                add("process-import", node, top)
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in DYNAMIC_CALLS:
                add("dynamic-code", node, f.id)
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                if (f.value.id, f.attr) in RISKY_ATTRS:
                    add("risky-call", node, f"{f.value.id}.{f.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_ids:
            if URL_RE.search(node.value):
                add("url-literal", node, node.value[:80])
    return findings


def audit_project(root):
    report, problems = [], 0
    for dirpath, _, files in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            for kind, line, what, reason in _scan_file(p, project_mode=True):
                rel = os.path.relpath(p, root)
                if reason:
                    report.append(f"  allowed  {rel}:{line}  {kind}: {what}  ({reason})")
                else:
                    problems += 1
                    report.append(f"  FLAG     {rel}:{line}  {kind}: {what}")
    return problems, report


def audit_dependencies(packages=("numpy", "cv2", "onnxruntime", "yaml")):
    """List .py files inside installed dependencies that import network
    modules. This is for HUMAN REVIEW: an import is not a connection, and the
    runtime guard blocks any actual attempt."""
    import importlib.util
    lines = []
    for pkg in packages:
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.submodule_search_locations:
            lines.append(f"[{pkg}] not found")
            continue
        root = list(spec.submodule_search_locations)[0]
        hits = []
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if fn.endswith(".py"):
                    p = os.path.join(dirpath, fn)
                    for kind, line, what, _ in _scan_file(p, project_mode=False):
                        if kind == "network-import":
                            hits.append(f"    {os.path.relpath(p, root)}:{line} imports {what}")
        lines.append(f"[{pkg}] {root}: {len(hits)} network imports")
        lines.extend(sorted(set(hits)))
    return lines
