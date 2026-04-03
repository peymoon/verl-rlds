"""
Online data selection for GRPO/PPO training.

Provides a pluggable interface for adaptive data selection during training.
The training loop calls a stable DataSelector interface; concrete implementations
handle the strategy (random, DOTS, cluster-based, etc.).

Usage:
    from verl.trainer.ppo.data_selector import build_selector
    selector = build_selector(config.data_selection)
    selector.initialize(train_dataset)
"""

from .base import DataSelectionConfig, DataSelector


def build_selector(config) -> DataSelector:
    """Factory: instantiate the right DataSelector from config.

    Args:
        config: OmegaConf or dict with at least a 'method' key.
                Recognized methods: 'none', 'random', 'dots', 'cluster'.

    Returns:
        A DataSelector instance ready for initialize().
    """
    if config is None:
        return _build_none_selector()

    if hasattr(config, "method"):
        method = config.method
    elif isinstance(config, dict):
        method = config.get("method", "none")
    else:
        return _build_none_selector()

    ds_config = _to_ds_config(config)

    if method == "none":
        return _build_none_selector()
    elif method == "random":
        from .random_selector import RandomSelector
        return RandomSelector(ds_config)
    elif method == "dots":
        from .dots_selector import DotsSelector
        return DotsSelector(ds_config)
    elif method == "cluster":
        from .cluster_selector import ClusterSelector
        return ClusterSelector(ds_config)
    else:
        raise ValueError(
            f"Unknown data selection method: '{method}'. "
            f"Choose from: none, random, dots, cluster"
        )


def _build_none_selector() -> DataSelector:
    """Build a no-op selector that returns all indices."""
    from .random_selector import RandomSelector

    cfg = DataSelectionConfig(method="none", reselect_interval=0)
    return RandomSelector(cfg)


def _to_ds_config(config) -> DataSelectionConfig:
    """Convert OmegaConf or dict to DataSelectionConfig."""
    if isinstance(config, DataSelectionConfig):
        return config

    if hasattr(config, "__iter__"):
        raw = {}
        for key in (
            "method",
            "reselect_schedule",
            "reselect_interval",
            "selection_budget",
            "selection_budget_pct",
            "cluster",
            "dots",
        ):
            val = getattr(config, key, None) if hasattr(config, key) else None
            if val is None and isinstance(config, dict):
                val = config.get(key)
            if val is not None:
                if hasattr(val, "items"):
                    raw[key] = dict(val)
                else:
                    raw[key] = val
        return DataSelectionConfig(**raw)

    return DataSelectionConfig()


__all__ = [
    "DataSelector",
    "DataSelectionConfig",
    "build_selector",
]
