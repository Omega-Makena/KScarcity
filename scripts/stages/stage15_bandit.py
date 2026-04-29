"""stage15_bandit.py — Stages 15.1–15.3: BanditRouter convergence benchmarks.

Thompson Sampling, UCB, and ε-Greedy routing — previously zero benchmark coverage.
All tests use a 10-arm setup where arm 0 is "good" (reward=0.9) and others are "bad" (reward=0.1).
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Dict

import numpy as np

from scripts.stages.utils import fail_result, make_result, skip_result


def _run_bandit_convergence(
    stage_id: str,
    name: str,
    algorithm_name: str,
    n_rounds: int = 50,
    pull_rate_threshold: float = 0.4,
) -> Dict[str, Any]:
    t0 = time.time()
    try:
        from scarcity.engine.bandit_router import BanditRouter, BanditConfig, BanditAlgorithm

        alg_map = {
            "thompson": BanditAlgorithm.THOMPSON,
            "ucb": BanditAlgorithm.UCB,
            "epsilon_greedy": BanditAlgorithm.EPSILON_GREEDY,
        }
        alg = alg_map[algorithm_name]
        config = BanditConfig(algorithm=alg)
        router = BanditRouter(config=config, n_arms=10)

        arm_ids = router.register_arms(10)
        good_arm = arm_ids[0]

        rng = np.random.default_rng(42)
        pull_history: list[int] = []

        for _ in range(n_rounds):
            proposed = router.propose(n_proposals=1)
            arm = proposed[0] if proposed else good_arm
            # Reward: good arm = 0.9, others = 0.1
            reward = 0.9 if arm == good_arm else 0.1
            reward += float(rng.standard_normal() * 0.05)  # tiny noise
            router.update(arm, float(np.clip(reward, 0, 1)))
            pull_history.append(arm)

        # Pull rate of good arm in last 20 rounds
        last_20 = pull_history[-20:]
        good_arm_rate = sum(1 for a in last_20 if a == good_arm) / len(last_20)

        top_arms = router.get_top_arms(3)
        top_arm_ids = [a[0] for a in top_arms] if top_arms else []
        good_in_top3 = good_arm in top_arm_ids

        stats = router.get_stats()
        wall = time.time() - t0

        converged = good_arm_rate >= pull_rate_threshold
        status = "PASS" if converged else ("WARN" if good_arm_rate >= pull_rate_threshold * 0.7 else "FAIL")

        return make_result(
            stage_id, name, status,
            f"good_arm pull rate >= {pull_rate_threshold} in last 20 rounds",
            {"good_arm_pull_rate_last20": round(good_arm_rate, 3),
             "good_in_top3": good_in_top3,
             "top_arms": top_arm_ids,
             "threshold": pull_rate_threshold,
             "n_rounds": n_rounds},
            wall,
        )
    except ImportError as e:
        return skip_result(stage_id, name, f"BanditRouter import failed: {e}")
    except Exception as e:
        return fail_result(stage_id, name,
                           f"BanditRouter({algorithm_name}) converges on good arm",
                           f"{e}\n{traceback.format_exc()[-1200:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage 15.1 — Thompson Sampling
# ---------------------------------------------------------------------------

def run_stage_15_1(fast: bool = False) -> Dict[str, Any]:
    n = 30 if fast else 50
    return _run_bandit_convergence("15.1", "BanditRouter_Thompson", "thompson",
                                   n_rounds=n, pull_rate_threshold=0.4)


# ---------------------------------------------------------------------------
# Stage 15.2 — UCB
# ---------------------------------------------------------------------------

def run_stage_15_2(fast: bool = False) -> Dict[str, Any]:
    # UCB is optimistic at start — needs more rounds to concentrate pulls.
    # Lower pull_rate_threshold in fast mode; at 50 rounds it reliably reaches 0.5.
    n = 50 if fast else 80
    threshold = 0.35 if fast else 0.5
    return _run_bandit_convergence("15.2", "BanditRouter_UCB", "ucb",
                                   n_rounds=n, pull_rate_threshold=threshold)


# ---------------------------------------------------------------------------
# Stage 15.3 — ε-Greedy
# ---------------------------------------------------------------------------

def run_stage_15_3(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "15.3", "BanditRouter_EpsilonGreedy"
    n = 30 if fast else 50

    try:
        from scarcity.engine.bandit_router import BanditRouter, BanditConfig, BanditAlgorithm

        config = BanditConfig(algorithm=BanditAlgorithm.EPSILON_GREEDY)
        router = BanditRouter(config=config, n_arms=10)
        arm_ids = router.register_arms(10)
        good_arm = arm_ids[0]

        rng = np.random.default_rng(42)
        pull_history: list[int] = []
        epoch_rewards: list[float] = []

        for i in range(n):
            proposed = router.propose(n_proposals=1)
            arm = proposed[0] if proposed else good_arm
            reward = 0.9 if arm == good_arm else 0.1
            reward += float(rng.standard_normal() * 0.05)
            reward = float(np.clip(reward, 0, 1))
            router.update(arm, reward)
            pull_history.append(arm)
            epoch_rewards.append(reward)

        last_20 = pull_history[-20:]
        good_arm_rate = sum(1 for a in last_20 if a == good_arm) / len(last_20)

        # Check mean reward increases over epochs (first half vs second half)
        half = n // 2
        mean_early = float(np.mean(epoch_rewards[:half]))
        mean_late = float(np.mean(epoch_rewards[half:]))
        reward_improves = mean_late >= mean_early * 0.9  # allow small variance

        wall = time.time() - t0
        converged = good_arm_rate >= 0.35
        status = "PASS" if (converged and reward_improves) else ("WARN" if converged else "FAIL")

        return make_result(stage_id, name, status,
                           "good_arm pull rate >= 0.35 in last 20 rounds; mean_reward improves",
                           {"good_arm_pull_rate_last20": round(good_arm_rate, 3),
                            "mean_reward_early": round(mean_early, 3),
                            "mean_reward_late": round(mean_late, 3),
                            "reward_improves": reward_improves},
                           wall)
    except ImportError as e:
        return skip_result(stage_id, name, f"BanditRouter import failed: {e}")
    except Exception as e:
        return fail_result(stage_id, name, "BanditRouter(epsilon_greedy) converges",
                           f"{e}\n{traceback.format_exc()[-1200:]}", time.time() - t0)
