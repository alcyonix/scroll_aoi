"""Configuration loading (YAML via yaml.safe_load only - never yaml.load)."""

import copy
import os

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config", "default.yaml")


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(user_config_path=None):
    cfg = load_yaml(DEFAULT_CONFIG)
    if user_config_path:
        cfg = _deep_merge(cfg, load_yaml(user_config_path))
    return cfg


def resolve(path):
    """Resolve a path relative to the project root (absolute paths pass through)."""
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def model_path(cfg, relative):
    return os.path.join(resolve(cfg.get("models_dir", "models")), relative)


def _item_list(entries, template, default_thr, group, valence=None, category=None):
    items = []
    for e in entries or []:
        name = str(e["name"]).strip()
        prompt = e.get("prompt") or template.format(name)
        items.append({
            "name": name,
            "prompt": prompt,
            "threshold": float(e.get("threshold", default_thr)),
            "group": group,
            "valence": valence,
            "category": category,
        })
    return items


def load_query_items(cfg):
    """Build the explicit list of open-vocabulary queries from the YAML files.
    Returns a list of dicts: name, prompt, threshold, group, valence, category."""
    ov = cfg["detectors"]["openvocab"]
    template = ov.get("query_template", "a photo of {}")
    thr = float(ov.get("default_threshold", 0.2))
    items = []

    emo = load_yaml(resolve(cfg["aoi"]["emotional_items_file"]))
    for valence in ("positive", "neutral", "negative"):
        for cat in emo.get(valence, []) or []:
            items += _item_list(cat.get("items"), template, thr, "emotional_item",
                                valence=valence, category=cat.get("category"))

    if cfg["aoi"]["salient_object"].get("enabled", True):
        sal = load_yaml(resolve(cfg["aoi"]["salient_prompts_file"]))
        items += _item_list(sal.get("items"), template, thr, "salient_candidate",
                            category="product / object")

    inter = cfg["aoi"]["interaction"]
    if inter.get("enabled", True) and inter.get("use_openvocab_prompts", False):
        items += _item_list(inter.get("prompts"), template,
                            float(inter.get("openvocab_threshold", thr)),
                            "interaction_prompt", category="physical interaction")
    return items
