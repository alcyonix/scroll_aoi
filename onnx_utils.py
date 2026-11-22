"""ONNX Runtime session creation with a strict local-only provider allowlist."""

import hashlib
import os

import onnxruntime as ort

# Official Windows wheels of ONNX Runtime emit ETW trace events that Windows'
# own diagnostics service may upload (see ONNX Runtime docs/Privacy.md).
# Turn that off for this process before any session is created.
try:
    ort.disable_telemetry_events()
except Exception:  # very old versions do not expose it
    pass

# Execution providers that run entirely on this machine. Anything not listed
# here is refused - notably "AzureExecutionProvider", which exists to send
# inference requests to a remote endpoint.
LOCAL_PROVIDERS = {
    "CPUExecutionProvider",
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
}


def make_session(model_file, cfg):
    if not os.path.isfile(model_file):
        raise FileNotFoundError(
            f"Model file not found: {model_file}\n"
            f"See MODELS.md for where to download it (manually, once).")
    requested = list(cfg.get("runtime", {}).get("onnx_providers", ["CPUExecutionProvider"]))
    for p in requested:
        if p not in LOCAL_PROVIDERS:
            raise ValueError(f"Execution provider '{p}' is not on the local-only allowlist.")
    available = set(ort.get_available_providers())
    providers = [p for p in requested if p in available] or ["CPUExecutionProvider"]

    so = ort.SessionOptions()
    threads = int(cfg.get("runtime", {}).get("intra_op_threads", 0) or 0)
    if threads > 0:
        so.intra_op_num_threads = threads
    so.log_severity_level = 3
    return ort.InferenceSession(model_file, sess_options=so, providers=providers)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
