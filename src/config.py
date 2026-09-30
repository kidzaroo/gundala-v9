"""Default configuration, YAML loading, CLI overrides and validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

SUPPORTED_MODELS = ("random_forest", "lightgbm", "xgboost", "mlp")
SUPPORTED_BALANCING = ("none", "random_oversampling", "smote", "smotenc")
SUPPORTED_SPLITS = ("stratified_random", "chronological")
SUPPORTED_CHRONO_MODES = ("global", "per_class")
SUPPORTED_HISTORY_POLICIES = ("split_isolated", "train_carryover")
SUPPORTED_FORMATS = ("auto", "jsonl", "json_array")
SUPPORTED_THRESHOLD_MODES = ("fixed", "validation_f1")
SUPPORTED_CLASS_WEIGHTS = ("none", "balanced")
STRING_SAMPLING_STRATEGIES = ("auto", "minority", "not majority", "not minority", "all")
DEFAULT_WINDOWS = (5, 10, 15, 30, 60)

DEFAULT_CONFIG: dict[str, Any] = {
    "random_seed": 42,
    "data_origin": "unspecified",
    "output_dir": "outputs/experiment_01",
    "input": {
        "benign_path": None,
        "tunnel_path": None,
        "format": "auto",
        "encoding": "utf-8",
    },
    "field_mapping": {
        "timestamp": ["ts"],
        "src_ip": ["id.orig_h"],
        "dst_ip": ["id.resp_h"],
        "src_port": ["id.orig_p"],
        "dst_port": ["id.resp_p"],
        "proto": ["proto"],
        "query": ["query"],
        "qtype": ["qtype"],
        "qtype_name": ["qtype_name"],
        "rcode": ["rcode"],
        "rcode_name": ["rcode_name"],
        "answers": ["answers"],
        "ttls": ["TTLs", "ttls"],
        "rejected": ["rejected"],
        "aa": ["AA"],
        "tc": ["TC"],
        "rd": ["RD"],
        "ra": ["RA"],
    },
    "validation": {"required_fields": ["timestamp"], "max_invalid_fraction": 0.5},
    "dedupe": {"enabled": True},
    "domain": {
        "include_private_suffixes": False,
        "local_suffixes": [
            "local", "lan", "home", "internal", "localdomain", "corp", "intranet", "home.arpa",
        ],
    },
    "split": {
        "strategy": "stratified_random",
        "test_size": 0.30,
        "chronological_mode": "global",
        "manifest_path": None,
    },
    "features": {"include_categorical": True},
    "windows": {"sizes": list(DEFAULT_WINDOWS), "history_policy": "split_isolated"},
    "run_baseline": True,
    "balancing": {
        "method": "random_oversampling",
        "sampling_strategy": "auto",
        "smote_k_neighbors": 5,
        "class_weight": "none",
        "allow_class_weight_with_resampling": False,
    },
    "models": {
        "enabled": list(SUPPORTED_MODELS),
        "params": {
            "random_forest": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
            "lightgbm": {
                "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
                "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1,
                "colsample_bytree": 0.8,
            },
            "xgboost": {
                "n_estimators": 300, "learning_rate": 0.1, "max_depth": 6,
                "subsample": 0.8, "colsample_bytree": 0.8,
            },
            "mlp": {
                "hidden_layer_sizes": [128, 64], "activation": "relu", "alpha": 0.0001,
                "batch_size": 256, "learning_rate_init": 0.001, "max_iter": 200,
                "early_stopping": False, "n_iter_no_change": 10,
            },
        },
    },
    "threshold": {"mode": "fixed", "value": 0.5, "validation_size": 0.2},
    "tuning": {
        "enabled": False,
        "cv_folds": 3,
        "n_iter": 6,
        "scoring": "average_precision",
        "param_grids": {
            "random_forest": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 12, 24],
                "clf__min_samples_leaf": [1, 3],
            },
            "lightgbm": {"clf__num_leaves": [15, 31, 63], "clf__learning_rate": [0.03, 0.05, 0.1]},
            "xgboost": {"clf__max_depth": [4, 6, 8], "clf__learning_rate": [0.05, 0.1, 0.2]},
            "mlp": {
                "clf__hidden_layer_sizes": [[64], [128, 64], [128, 64, 32]],
                "clf__alpha": [0.0001, 0.001],
            },
        },
    },
    "runtime": {"n_jobs": 1},
    "outputs": {"save_models": True, "save_plots": True, "top_k_importance": 20},
}


def deep_update(base: dict, updates: Mapping) -> dict:
    """Recursively merge ``updates`` into ``base`` (in place) and return it."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path: Optional[str | Path] = None) -> dict:
    """Load defaults, then merge the YAML file at ``path`` (if given)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return cfg
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        user_cfg = yaml.safe_load(fh) or {}
    if not isinstance(user_cfg, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at top level.")
    unknown = sorted(set(user_cfg) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {unknown}. Valid keys: {sorted(DEFAULT_CONFIG)}")
    return deep_update(cfg, user_cfg)


def apply_overrides(cfg: dict, overrides: Mapping[str, Any]) -> dict:
    """Apply dotted-path overrides, e.g. ``{"split.strategy": "chronological"}``.

    Entries whose value is ``None`` are ignored (meaning "not given on the CLI").
    """
    for dotted, value in overrides.items():
        if value is None:
            continue
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def normalize_sampling_strategy(value: Any) -> Any:
    """Return a value acceptable as imbalanced-learn ``sampling_strategy``."""
    if isinstance(value, bool):
        raise ValueError("sampling_strategy must not be a boolean.")
    if isinstance(value, (int, float)):
        value = float(value)
        if not 0.0 < value <= 1.0:
            raise ValueError("Numeric sampling_strategy must satisfy 0 < ratio <= 1 (minority/majority).")
        return value
    if isinstance(value, dict):
        return {int(k): int(v) for k, v in value.items()}
    if isinstance(value, str):
        text = value.strip().lower()
        if text in STRING_SAMPLING_STRATEGIES:
            return text
        try:
            return normalize_sampling_strategy(float(text))
        except ValueError:
            pass
    raise ValueError(
        f"Invalid sampling_strategy {value!r}. Use one of {STRING_SAMPLING_STRATEGIES}, "
        "a float ratio in (0, 1], or a dict {class: n_samples}."
    )


def _require_in(name: str, value: Any, allowed: tuple) -> None:
    if value not in allowed:
        raise ValueError(f"Invalid {name}={value!r}. Allowed: {list(allowed)}")


def validate_config(cfg: dict) -> dict:
    """Validate (and lightly normalise) the configuration. Returns ``cfg``."""
    _require_in("input.format", cfg["input"]["format"], SUPPORTED_FORMATS)
    _require_in("split.strategy", cfg["split"]["strategy"], SUPPORTED_SPLITS)
    _require_in("split.chronological_mode", cfg["split"]["chronological_mode"], SUPPORTED_CHRONO_MODES)
    _require_in("windows.history_policy", cfg["windows"]["history_policy"], SUPPORTED_HISTORY_POLICIES)
    _require_in("balancing.method", cfg["balancing"]["method"], SUPPORTED_BALANCING)
    _require_in("balancing.class_weight", cfg["balancing"]["class_weight"], SUPPORTED_CLASS_WEIGHTS)
    _require_in("threshold.mode", cfg["threshold"]["mode"], SUPPORTED_THRESHOLD_MODES)

    test_size = float(cfg["split"]["test_size"])
    if not 0.0 < test_size < 1.0:
        raise ValueError("split.test_size must be in (0, 1).")
    cfg["split"]["test_size"] = test_size

    thr = float(cfg["threshold"]["value"])
    if not 0.0 < thr < 1.0:
        raise ValueError("threshold.value must be in (0, 1).")
    val_size = float(cfg["threshold"]["validation_size"])
    if not 0.0 < val_size < 1.0:
        raise ValueError("threshold.validation_size must be in (0, 1).")

    models = list(cfg["models"]["enabled"])
    if not models:
        raise ValueError("models.enabled must contain at least one classifier.")
    for m in models:
        _require_in("model name", m, SUPPORTED_MODELS)
    cfg["models"]["enabled"] = models

    sizes = cfg["windows"]["sizes"] or []
    cleaned = []
    for w in sizes:
        w = float(w)
        if w <= 0:
            raise ValueError("Window sizes must be positive numbers of seconds.")
        cleaned.append(int(w) if w.is_integer() else w)
    cfg["windows"]["sizes"] = sorted(set(cleaned))
    if not cfg["windows"]["sizes"] and not cfg["run_baseline"]:
        raise ValueError("Nothing to run: no windows given and run_baseline is false.")

    if cfg["windows"]["history_policy"] == "train_carryover" and cfg["split"]["strategy"] != "chronological":
        raise ValueError(
            "windows.history_policy='train_carryover' is only defined for split.strategy='chronological' "
            "(train history must precede test events)."
        )

    cfg["balancing"]["sampling_strategy"] = normalize_sampling_strategy(cfg["balancing"]["sampling_strategy"])
    if int(cfg["balancing"]["smote_k_neighbors"]) < 1:
        raise ValueError("balancing.smote_k_neighbors must be >= 1.")

    n_jobs = int(cfg["runtime"]["n_jobs"])
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("runtime.n_jobs must be -1 or a positive integer.")
    cfg["runtime"]["n_jobs"] = n_jobs

    req = cfg["validation"]["required_fields"]
    unknown = [f for f in req if f not in cfg["field_mapping"]]
    if unknown:
        raise ValueError(f"validation.required_fields contains unknown fields: {unknown}")
    cfg["random_seed"] = int(cfg["random_seed"])
    return cfg
