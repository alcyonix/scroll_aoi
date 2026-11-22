"""Command-line interface. Always launched through `python -m aoi_tool`, which
installs the network guard first (see __main__.py)."""

import argparse
import os
import sys

from . import netguard


def _cmd_run(args):
    from .config import load_config
    from .pipeline import Pipeline
    cfg = load_config(args.config)
    if args.providers:
        cfg["runtime"]["onnx_providers"] = args.providers.split(",")
    p = Pipeline(cfg, args.video, args.out, gaze_path=args.gaze, seeds_path=args.seeds,
                 skip_missing=args.skip_missing, qa_video=False if args.no_qa else None)
    n = p.run(args.start, args.end)
    print(f"done: {n} frames -> {os.path.abspath(args.out)}")
    if p.warnings:
        print("warnings:\n  " + "\n  ".join(p.warnings))
    return 0


def _cmd_selftest(args):
    import numpy as np
    from .config import load_config, load_query_items, model_path
    from .onnx_utils import LOCAL_PROVIDERS, sha256_file
    ok = True
    missing = 0
    print("== network guard ==")
    print(f"installed: {netguard.is_installed()}")
    for name, blocked, msg in netguard.self_test():
        ok &= blocked
        print(f"  [{'BLOCKED' if blocked else 'FAIL   '}] {name}")
    import onnxruntime as ort
    print("== onnxruntime ==")
    print(f"  version {ort.__version__}; available providers: {ort.get_available_providers()}")
    print(f"  local-only allowlist enforced: {sorted(LOCAL_PROVIDERS)}")

    cfg = load_config(args.config)
    dummy = (np.random.default_rng(0).random((cfg['screen']['rect_height'],
                                              cfg['screen']['rect_width'], 3)) * 255).astype(np.uint8)
    print("== models ==")
    checks = []
    from .detectors.faces import FaceDetector
    from .detectors.openvocab import OwlV2Detector
    from .detectors.pose import PoseDetector
    from .detectors.text import TextDetector
    checks = [("faces", FaceDetector), ("text", TextDetector), ("pose", PoseDetector),
              ("openvocab", lambda c: OwlV2Detector(c, load_query_items(c)))]
    for name, ctor in checks:
        try:
            m = ctor(cfg)
            res = m(dummy)
            line = f"  [OK     ] {name}: {os.path.relpath(m.path, model_path(cfg, '.'))} " \
                   f"sha256={sha256_file(m.path)[:16]}... dummy-run returned {len(res)} detections"
            print(line)
            if name == "openvocab":
                good, ids = m.tok.self_check()
                ok &= good
                print(f"  [{'OK     ' if good else 'FAIL   '}] CLIP tokenizer check 'a photo of a cat' -> {ids}")
                print(f"            {len(m.items)} explicit text queries loaded")
                if m.truncated_prompts:
                    print(f"            WARNING truncated prompts: {m.truncated_prompts}")
        except FileNotFoundError as exc:
            missing += 1
            print(f"  [MISSING] {name}: {exc}")
        except Exception as exc:
            ok = False
            print(f"  [ERROR  ] {name}: {type(exc).__name__}: {exc}")
    verdict = "FAIL (see above)" if not ok else (
        f"INCOMPLETE ({missing} model(s) missing - see MODELS.md)" if missing else "PASS")
    print("== result ==", verdict)
    return 0 if ok else 1


def _cmd_audit(args):
    from .audit import audit_dependencies, audit_project
    root = os.path.dirname(os.path.abspath(__file__))
    problems, report = audit_project(root)
    print(f"Static audit of {root}")
    print("\n".join(report) if report else "  (nothing to report)")
    print(f"unexplained findings: {problems}")
    if args.deps:
        print("\nDependency scan (for review; imports are not connections):")
        print("\n".join(audit_dependencies()))
    return 0 if problems == 0 else 1


def _cmd_annotate(args):
    from .annotate import run_annotator
    run_annotator(args.video, args.seeds, args.config)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m aoi_tool",
                                 description="Offline AOI detection for phone screens in Tobii scene video.")
    ap.add_argument("--allow-subprocess", action="store_true",
                    help="log instead of block process spawns (network stays blocked)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="process a scene video")
    r.add_argument("--video", required=True, help="Tobii scenevideo.mp4 (local file)")
    r.add_argument("--out", required=True, help="output folder")
    r.add_argument("--config", help="YAML overriding config/default.yaml")
    r.add_argument("--gaze", help="gazedata.gz (Glasses 3) or CSV timestamp_s,x_norm,y_norm")
    r.add_argument("--seeds", help="manual screen corners CSV from annotate-screen")
    r.add_argument("--start", type=int, help="first frame")
    r.add_argument("--end", type=int, help="stop before this frame")
    r.add_argument("--providers", help="e.g. CUDAExecutionProvider,CPUExecutionProvider")
    r.add_argument("--no-qa", action="store_true", help="skip the QA overlay video")
    r.add_argument("--skip-missing", action="store_true",
                   help="disable detectors whose model file is missing instead of failing")
    r.set_defaults(fn=_cmd_run)

    s = sub.add_parser("selftest", help="verify network guard, models and tokenizer")
    s.add_argument("--config")
    s.set_defaults(fn=_cmd_selftest)

    a = sub.add_parser("audit", help="static scan of the source for network/process/dynamic code")
    a.add_argument("--deps", action="store_true", help="also scan installed dependencies")
    a.set_defaults(fn=_cmd_audit)

    n = sub.add_parser("annotate-screen", help="click phone-screen corners on hard frames")
    n.add_argument("--video", required=True)
    n.add_argument("--seeds", required=True, help="CSV to create/update")
    n.add_argument("--config")
    n.set_defaults(fn=_cmd_annotate)

    args = ap.parse_args(argv)
    if not netguard.is_installed():
        print("Refusing to run: network guard not installed. Use `python -m aoi_tool ...`.",
              file=sys.stderr)
        return 2
    return args.fn(args)
