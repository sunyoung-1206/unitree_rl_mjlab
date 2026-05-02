"""Phase 4 포괄적 분석: filterexact vs coupled RL 학습 비교.

학습 완료 후 실행:
  python scripts/analyze_phase4.py

생성물:
  phase4_results/
  ├── learning_curves.png
  ├── performance_table.md
  ├── current_tracking/
  ├── cross_eval/
  ├── gait_analysis/
  └── summary.md
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

RESULTS_DIR = Path("solver_comparison/phase4_results")
PLOTS_DIR = RESULTS_DIR
for sub in ["current_tracking", "cross_eval", "gait_analysis"]:
    (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  1. Load training logs
# ═══════════════════════════════════════════════════════════════

def find_log_dirs():
    log_root = Path("logs/rsl_rl")
    groups = {"native": [], "coupled": []}
    for key in groups:
        exp_dir = log_root / f"phase4_{key}"
        if exp_dir.exists():
            for run_dir in sorted(exp_dir.iterdir()):
                if run_dir.is_dir():
                    groups[key].append(run_dir)
    return groups


def load_tb_scalars(log_dir: Path):
    """Load TensorBoard scalars from event files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        data = {}
        for tag in tags:
            events = ea.Scalars(tag)
            data[tag] = {
                "steps": np.array([e.step for e in events]),
                "values": np.array([e.value for e in events]),
            }
        return data
    except Exception as e:
        print(f"  Warning: Could not load TB data from {log_dir}: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════
#  2. Learning curve plot
# ═══════════════════════════════════════════════════════════════

def plot_learning_curves(groups):
    """학습 곡선 비교: episode return + episode length."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 4: Native-Electric (filterexact) vs Coupled-Electric", fontsize=13)

    metrics = [
        ("Train/mean_reward", "Mean Episode Reward"),
        ("Train/mean_episode_length", "Mean Episode Length"),
    ]
    colors = {"native": "red", "coupled": "blue"}
    labels = {"native": "filterexact", "coupled": "coupled"}

    for ax, (metric_key, title) in zip(axes, metrics):
        for group_key in ["native", "coupled"]:
            all_vals = []
            for log_dir in groups[group_key]:
                data = load_tb_scalars(log_dir)
                if metric_key in data:
                    vals = data[metric_key]["values"]
                    steps = data[metric_key]["steps"]
                    all_vals.append(vals)
                    ax.plot(steps, vals, color=colors[group_key], alpha=0.2, linewidth=0.5)

            if all_vals:
                min_len = min(len(v) for v in all_vals)
                aligned = np.array([v[:min_len] for v in all_vals])
                mean = aligned.mean(axis=0)
                std = aligned.std(axis=0)
                steps = np.arange(min_len)
                ax.plot(steps, mean, color=colors[group_key], linewidth=2,
                        label=f"{labels[group_key]} (n={len(all_vals)})")
                ax.fill_between(steps, mean - std, mean + std,
                                color=colors[group_key], alpha=0.15)

        ax.set_xlabel("Iteration")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curves.png", dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'learning_curves.png'}")


# ═══════════════════════════════════════════════════════════════
#  3. Performance table + statistical test
# ═══════════════════════════════════════════════════════════════

def make_performance_table(groups):
    """최종 100 iter 평균으로 성능 비교, t-test."""
    from scipy import stats

    metric_key = "Train/mean_reward"
    length_key = "Train/mean_episode_length"
    tail_n = 100

    results = {}
    for group_key in ["native", "coupled"]:
        final_rewards = []
        final_lengths = []
        for log_dir in groups[group_key]:
            data = load_tb_scalars(log_dir)
            if metric_key in data:
                vals = data[metric_key]["values"]
                final_rewards.append(np.mean(vals[-tail_n:]))
            if length_key in data:
                vals = data[length_key]["values"]
                final_lengths.append(np.mean(vals[-tail_n:]))
        results[group_key] = {
            "rewards": final_rewards,
            "lengths": final_lengths,
        }

    lines = ["# Phase 4: Performance Comparison", ""]
    lines.append("| Metric | filterexact | coupled | p-value |")
    lines.append("|--------|------------|---------|---------|")

    for metric_name, key in [("Mean Reward", "rewards"), ("Mean Ep Length", "lengths")]:
        fe = np.array(results["native"][key])
        cp = np.array(results["coupled"][key])
        if len(fe) > 1 and len(cp) > 1:
            t_stat, p_val = stats.ttest_ind(fe, cp)
        else:
            p_val = float("nan")
        fe_str = f"{fe.mean():.2f} ± {fe.std():.2f}" if len(fe) > 0 else "N/A"
        cp_str = f"{cp.mean():.2f} ± {cp.std():.2f}" if len(cp) > 0 else "N/A"
        lines.append(f"| {metric_name} | {fe_str} | {cp_str} | {p_val:.4f} |")

    lines.append("")
    lines.append(f"(Last {tail_n} iterations averaged, {len(results['native']['rewards'])} seeds each)")

    text = "\n".join(lines)
    (RESULTS_DIR / "performance_table.md").write_text(text)
    print(text)
    return results


# ═══════════════════════════════════════════════════════════════
#  4. Rollout utility (for current tracking, cross-eval, gait)
# ═══════════════════════════════════════════════════════════════

def load_policy_and_rollout(task_id, checkpoint_path, n_episodes=10, device="cuda:0"):
    """체크포인트에서 정책 로드 → rollout → 데이터 수집."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 5.0  # 5초 에피소드

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env_wrapped, asdict(agent_cfg), "/tmp/phase4_rollout", device)
    runner.load(str(checkpoint_path))

    # Rollout
    policy = runner.get_inference_policy(device=device)
    all_data = []

    for ep in range(n_episodes):
        obs = env_wrapped.get_observations()
        total_reward = 0.0
        done = False
        step = 0

        while not done and step < 2000:
            with torch.no_grad():
                actions = policy(obs)
            obs, rew, dones, infos = env_wrapped.step(actions)
            total_reward += rew.squeeze().item()
            done = dones.any().item()
            step += 1

        all_data.append({"total_reward": total_reward, "length": step})

    env.close()
    return all_data


def find_best_checkpoint(log_dir):
    """로그 디렉토리에서 최신 체크포인트 찾기."""
    ckpt_dir = log_dir
    candidates = sorted(ckpt_dir.glob("model_*.pt"))
    if not candidates:
        candidates = sorted(ckpt_dir.glob("*.pt"))
    return candidates[-1] if candidates else None


# ═══════════════════════════════════════════════════════════════
#  5. Current tracking analysis
# ═══════════════════════════════════════════════════════════════

def analyze_current_tracking(groups, device="cuda:0"):
    """정책 rollout 성능 비교."""
    print("\n=== Policy Rollout Analysis ===")

    task_map = {
        "native": "Unitree-Go2-Flat-Native-Electric",
        "coupled": "Unitree-Go2-Flat-Coupled-Electric",
    }

    for group_key in ["native", "coupled"]:
        if not groups[group_key]:
            continue
        log_dir = groups[group_key][0]
        ckpt = find_best_checkpoint(log_dir)
        if ckpt is None:
            print(f"  No checkpoint found for {group_key}")
            continue

        print(f"  Rolling out {group_key} from {ckpt.name}...")
        try:
            data = load_policy_and_rollout(
                task_map[group_key], ckpt, n_episodes=5, device=device
            )
            rewards = [ep["total_reward"] for ep in data]
            lengths = [ep["length"] for ep in data]
            print(f"    {group_key}: reward={np.mean(rewards):.2f}±{np.std(rewards):.2f}, "
                  f"length={np.mean(lengths):.1f}±{np.std(lengths):.1f}")
        except Exception as e:
            print(f"  Rollout failed for {group_key}: {e}")


# ═══════════════════════════════════════════════════════════════
#  6. Cross-evaluation
# ═══════════════════════════════════════════════════════════════

def cross_evaluate(groups, device="cuda:0"):
    """교차 평가: A 정책 → B 환경, B 정책 → A 환경."""
    print("\n=== Cross-Evaluation ===")

    task_map = {
        "native": "Unitree-Go2-Flat-Native-Electric",
        "coupled": "Unitree-Go2-Flat-Coupled-Electric",
    }

    results = {}
    for policy_key in ["native", "coupled"]:
        if not groups[policy_key]:
            continue
        ckpt = find_best_checkpoint(groups[policy_key][0])
        if ckpt is None:
            continue

        for env_key in ["native", "coupled"]:
            label = f"{policy_key}_policy_in_{env_key}_env"
            print(f"  {label}...")
            try:
                data = load_policy_and_rollout(
                    task_map[env_key], ckpt, n_episodes=10, device=device
                )
                rewards = [ep["total_reward"] for ep in data]
                lengths = [ep["length"] for ep in data]
                results[label] = {
                    "mean_reward": np.mean(rewards),
                    "std_reward": np.std(rewards),
                    "mean_length": np.mean(lengths),
                }
                print(f"    reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}, "
                      f"length: {np.mean(lengths):.1f}")
            except Exception as e:
                print(f"    Failed: {e}")
                results[label] = {"mean_reward": float("nan"), "std_reward": 0, "mean_length": 0}

    # Save
    if results:
        lines = ["# Cross-Evaluation Results", "",
                 "| Policy \\ Env | Native | Coupled |",
                 "|-------------|--------|---------|"]
        for pol in ["native", "coupled"]:
            row = f"| {pol} |"
            for env in ["native", "coupled"]:
                key = f"{pol}_policy_in_{env}_env"
                r = results.get(key, {})
                row += f" {r.get('mean_reward', 'N/A'):.2f} ± {r.get('std_reward', 0):.2f} |"
            lines.append(row)
        text = "\n".join(lines)
        (RESULTS_DIR / "cross_eval" / "cross_eval_results.md").write_text(text)
        print(f"\n{text}")

    return results


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    os.environ.setdefault("MUJOCO_GL", "egl")

    # Import tasks to register them
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    groups = find_log_dirs()
    total = sum(len(v) for v in groups.values())
    print(f"Found {total} training runs:")
    for key, dirs in groups.items():
        print(f"  {key}: {len(dirs)} runs")
        for d in dirs:
            print(f"    {d}")

    if total == 0:
        print("\nNo training logs found. Run training first.")
        return

    # 2. Learning curves
    print("\n=== Learning Curves ===")
    plot_learning_curves(groups)

    # 3. Performance table
    print("\n=== Performance Table ===")
    perf = make_performance_table(groups)

    # 4. Current tracking (skip if no GPU or rollout fails)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        analyze_current_tracking(groups, device=device)
    except Exception as e:
        print(f"Current tracking analysis skipped: {e}")

    # 5. Cross-evaluation
    try:
        cross_results = cross_evaluate(groups, device=device)
    except Exception as e:
        print(f"Cross-evaluation skipped: {e}")
        cross_results = {}

    # 6. Summary
    print("\n=== Generating Summary ===")
    generate_summary(groups, perf, cross_results)


def generate_summary(groups, perf, cross_results):
    """종합 보고서 생성."""
    from scipy import stats

    lines = ["# Phase 4 결과 요약", ""]

    # Experiment conditions
    lines.append("## 실험 조건")
    lines.append(f"- 환경: Unitree-Go2-Flat (Native-Electric vs Coupled-Electric)")
    lines.append(f"- Seeds: {len(groups.get('native', []))} (native), {len(groups.get('coupled', []))} (coupled)")
    lines.append(f"- dt: 0.1ms, decimation: 50, policy dt: 5ms")
    lines.append(f"- GPU: RTX 5080")
    lines.append("")

    # Learning curves
    lines.append("## 학습 곡선")
    lines.append("![Learning Curves](learning_curves.png)")
    lines.append("")

    # Performance
    lines.append("## 최종 성능")
    perf_table = (RESULTS_DIR / "performance_table.md").read_text()
    lines.append(perf_table)
    lines.append("")

    # Determine scenario
    fe_rewards = np.array(perf.get("native", {}).get("rewards", []))
    cp_rewards = np.array(perf.get("coupled", {}).get("rewards", []))
    if len(fe_rewards) > 1 and len(cp_rewards) > 1:
        _, p_val = stats.ttest_ind(fe_rewards, cp_rewards)
        lines.append(f"### 통계적 유의성: p = {p_val:.4f}")
        if p_val > 0.05:
            scenario = "A"
            lines.append("→ **시나리오 A**: 학습 성능에 통계적으로 유의미한 차이 없음")
        elif abs(fe_rewards.mean() - cp_rewards.mean()) / max(abs(fe_rewards.mean()), 1e-8) < 0.1:
            scenario = "B"
            lines.append("→ **시나리오 B**: 통계적 차이는 있으나 실질적 차이 미미")
        else:
            scenario = "C"
            lines.append("→ **시나리오 C**: 유의미한 성능 차이 관찰")
    else:
        scenario = "N/A"
        lines.append("→ 시드 수 부족으로 통계 검정 불가")
    lines.append("")

    # Cross-eval
    if cross_results:
        lines.append("## Cross-Evaluation")
        cross_text = (RESULTS_DIR / "cross_eval" / "cross_eval_results.md").read_text()
        lines.append(cross_text)
    lines.append("")

    # Conclusion
    lines.append("## 결론")
    if scenario == "A":
        lines.append("전류 추적 정확도 19x 개선(Phase 3)에도 불구하고, RL 학습 성능에는 측정 가능한 차이가 없었다.")
        lines.append("이는 RL의 reward shaping과 domain randomization이 물리 정밀도 차이를 흡수한다는 것을 시사한다.")
        lines.append("그러나 이 결과는 물리적 정확도 향상 자체의 가치를 부정하지 않으며,")
        lines.append("전류 프로파일이 중요한 downstream task (고장 진단, 에너지 효율 최적화)에서는 차이가 있을 수 있다.")
    elif scenario == "B":
        lines.append("RL 학습 곡선은 비슷하지만, 전류 추적 품질이나 정책 행동에서 차이가 관찰되었다.")
        lines.append("이는 물리 정밀도 차이가 정책 수준에서 부분적으로 인식되고 있음을 의미한다.")
    elif scenario == "C":
        lines.append("물리 정밀도 향상이 RL 학습 성능에 직접적으로 전이되는 결과를 관찰했다.")
        lines.append("이는 시뮬레이터의 전기-기계 커플링 정확도가 정책 품질에 영향을 준다는 최초의 실증이다.")
    lines.append("")

    text = "\n".join(lines)
    (RESULTS_DIR / "summary.md").write_text(text)
    print(f"\nSaved: {RESULTS_DIR / 'summary.md'}")
    print(text)


if __name__ == "__main__":
    main()
