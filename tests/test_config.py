"""Phase 1 smoke test: config.yaml loads and validates with no KLEE needed."""

from backend.core.config import load_config


def test_config_loads():
    cfg = load_config()
    assert cfg.project.name == "adaptive-symbolic-execution"
    assert cfg.klee.binary == "klee"
    assert len(cfg.features.names) == 14


def test_baseline_heuristics_present():
    cfg = load_config()
    expected = {"dfs", "bfs", "random-state", "random-path", "nurs:covnew", "nurs:md2u", "cov-opt"}
    assert expected.issubset(set(cfg.klee.search_heuristics_baselines))


def test_reward_config_signs():
    cfg = load_config()
    r = cfg.reinforcement_learning.reward
    assert r.coverage_gain_weight > 0
    assert r.timeout_penalty < 0
    assert r.state_explosion_penalty < 0
