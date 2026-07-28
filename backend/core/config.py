"""
Typed configuration loading for the Adaptive-Symbolic-Execution project.

Every module should obtain configuration by calling :func:`load_config`
rather than reading ``configs/config.yaml`` directly. This keeps the schema
in one place and lets us validate the file once at startup instead of
letting malformed config surface as a confusing error three modules deep.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


class ProjectConfig(BaseModel):
    name: str
    version: str
    seed: int = 42


class PathsConfig(BaseModel):
    dataset_dir: str
    results_dir: str
    experiments_dir: str
    models_dir: str
    logs_dir: str


class KleeConfig(BaseModel):
    binary: str = "klee"
    bitcode_compile_flags: list[str] = Field(default_factory=list)
    max_time_seconds: int = 3600
    max_memory_mb: int = 4096
    search_heuristics_baselines: list[str] = Field(default_factory=list)


class FeaturesConfig(BaseModel):
    names: list[str]


class LLMConfig(BaseModel):
    provider: str = "local"
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 400
    cache_dir: str
    response_schema: dict[str, str]


class NeuralNetConfig(BaseModel):
    hidden_sizes: list[int]
    dropout: float
    lr: float
    epochs: int


class CrossValidationConfig(BaseModel):
    folds: int = 5


class MLRankingConfig(BaseModel):
    models: list[str]
    neural_net: NeuralNetConfig
    cross_validation: CrossValidationConfig


class RewardConfig(BaseModel):
    coverage_gain_weight: float
    timeout_penalty: float
    solver_overhead_penalty_scale: float
    state_explosion_penalty: float


class RLTrainingConfig(BaseModel):
    total_timesteps: int
    eval_interval: int


class ReinforcementLearningConfig(BaseModel):
    algorithm: str = "dqn"
    gamma: float = 0.99
    learning_rate: float = 3e-4
    replay_buffer_size: int = 100_000
    batch_size: int = 64
    target_update_interval: int = 1000
    reward: RewardConfig
    training: RLTrainingConfig


class EvaluationConfig(BaseModel):
    metrics: list[str]
    significance_test: str = "mannwhitneyu"
    alpha: float = 0.05
    num_repetitions: int = 10


class DashboardConfig(BaseModel):
    backend_port: int = 8000
    frontend_port: int = 3000


class AppConfig(BaseModel):
    """Root configuration object mirroring configs/config.yaml."""

    project: ProjectConfig
    paths: PathsConfig
    klee: KleeConfig
    features: FeaturesConfig
    llm: LLMConfig
    ml_ranking: MLRankingConfig
    reinforcement_learning: ReinforcementLearningConfig
    evaluation: EvaluationConfig
    dashboard: DashboardConfig


@functools.lru_cache(maxsize=None)
def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and validate ``config.yaml``. Cached so repeated calls are cheap.

    Args:
        config_path: Path to the YAML config file. Defaults to
            ``configs/config.yaml`` at the repository root.

    Returns:
        A validated :class:`AppConfig` instance.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return AppConfig(**raw)
