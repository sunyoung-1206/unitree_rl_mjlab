"""Phase 4: filterexact vs filterexact_coupled RL 학습 비교.

Usage:
  # 3 seeds 순차 실행 (단일 GPU)
  python scripts/train_phase4_comparison.py

  # 특정 seed만
  python scripts/train_phase4_comparison.py --seeds 42

  # 짧은 학습 (디버그)
  python scripts/train_phase4_comparison.py --max-iters 100

  # CPU only
  python scripts/train_phase4_comparison.py --device cpu
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


TASKS = {
    "native":  "Unitree-Go2-Flat-Native-Electric",
    "coupled": "Unitree-Go2-Flat-Coupled-Electric",
}

DEFAULT_SEEDS = [42, 123, 7]
DEFAULT_MAX_ITERS = 10001


def run_training(task_id: str, seed: int, max_iters: int, gpu_id: int | None, run_name: str):
    """단일 학습 실행."""
    cmd = [
        sys.executable, "scripts/train.py", task_id,
        f"--agent.seed={seed}",
        f"--agent.max-iterations={max_iters}",
        f"--agent.experiment-name=phase4_{run_name}",
        f"--agent.run-name=seed{seed}",
        "--env.scene.num-envs=4096",
    ]
    if gpu_id is not None:
        cmd.append(f"--gpu-ids=[{gpu_id}]")
    else:
        cmd.append("--gpu-ids=[]")

    print(f"\n{'='*60}")
    print(f"Training: {run_name} | task={task_id} | seed={seed} | max_iters={max_iters}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    result = subprocess.run(cmd, env=env, cwd=str(Path(__file__).parent.parent))
    if result.returncode != 0:
        print(f"WARNING: Training failed with return code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Phase 4: RL training comparison")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Random seeds for training")
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS,
                        help="Max training iterations per run")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use (set -1 for CPU)")
    parser.add_argument("--tasks", nargs="+", choices=["native", "coupled"], default=["native", "coupled"],
                        help="Which tasks to train")
    args = parser.parse_args()

    gpu_id = args.gpu if args.gpu >= 0 else None

    print("Phase 4: filterexact vs coupled RL training comparison")
    print(f"Seeds: {args.seeds}")
    print(f"Max iterations: {args.max_iters}")
    print(f"GPU: {'CPU' if gpu_id is None else gpu_id}")
    print(f"Tasks: {args.tasks}")

    results = []
    for task_key in args.tasks:
        task_id = TASKS[task_key]
        for seed in args.seeds:
            rc = run_training(task_id, seed, args.max_iters, gpu_id, task_key)
            results.append((task_key, seed, rc))

    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for task_key, seed, rc in results:
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"  {task_key:10s} seed={seed:3d}  {status}")

    log_dirs = list(Path("logs/rsl_rl").glob("phase4_*"))
    print(f"\nLog directories: {len(log_dirs)}")
    for d in sorted(log_dirs):
        print(f"  {d}")

    print("\nTo analyze results, run:")
    print("  python scripts/analyze_phase4.py")


if __name__ == "__main__":
    main()
