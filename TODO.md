#!/usr/bin/env python3
"""
Step 0 - Deployment interface contract inspector (rsl_rl / mjlab -> unitree_go2_deploy)

What it does
------------
1. Locates the latest rsl_rl checkpoint (model_*.pt) in a run directory.
2. Reads the *actor* network directly from the checkpoint state_dict and infers
   - input dim  == num_obs
   - output dim == num_actions
   - hidden layer sizes / activation count
3. Loads every serialized config it can find (params/*.yaml, *.json, *.pkl) and
   greps for the fields that matter for deployment (scales, action_scale, PD
   gains, default joint angles, joint order, decimation/dt, command ranges,
   gait/phase, obs term ordering).
4. Prints a side-by-side comparison against the `unitree_go2_deploy` runtime
   contract and an alignment checklist with PASS / CHECK / FAIL markers.

It does NOT modify anything. It only reads and reports.

Usage
-----
    python inspect_policy_contract.py [RUN_DIR] [--checkpoint model_XXXX.pt] [--out report.json]

Default RUN_DIR is the one you gave; override on the CLI if needed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

import torch


# --------------------------------------------------------------------------- #
# Tolerant YAML loader
# --------------------------------------------------------------------------- #
# mjlab env.yaml uses `!!python/object:...`, `!!python/tuple`,
# `!!python/name:...`, etc. PyYAML's `unsafe_load` should handle most, but
# some module paths are unresolvable in this read-only context (we don't want
# to import the training env just to inspect a checkpoint). Register tolerant
# constructors so the load never fails: tuples → lists, names → fully-qualified
# string, unknown objects → {"__class__": <name>, **state}.
if yaml is not None:
    class _TolerantLoader(yaml.UnsafeLoader):
        pass

    def _tuple_ctor(loader, node):
        return loader.construct_sequence(node, deep=True)

    def _name_ctor(loader, suffix, node):
        return f"<name:{suffix}>"

    def _object_ctor(loader, suffix, node):
        try:
            state = loader.construct_mapping(node, deep=True)
        except Exception:
            try:
                state = loader.construct_sequence(node, deep=True)
            except Exception:
                state = loader.construct_scalar(node)
        out = {"__class__": suffix}
        if isinstance(state, dict):
            out.update(state)
        else:
            out["__state__"] = state
        return out

    _TolerantLoader.add_constructor("tag:yaml.org,2002:python/tuple", _tuple_ctor)
    _TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", _name_ctor)
    _TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/object:", _object_ctor)
    _TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/object/new:", _object_ctor)
    _TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/object/apply:", _object_ctor)


# --------------------------------------------------------------------------- #
# Deploy runtime contract
# (from unitree_go2_deploy/configs/go2_deploy_baseline_teleop.yaml + sim_to_real.py)
# --------------------------------------------------------------------------- #
DEPLOY_CONTRACT = {
    "num_obs": 47,
    "num_actions": 12,
    # ordered observation layout the deploy runner assumes (name, dim)
    "obs_layout": [
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("velocity_command", 3),
        ("joint_pos_error", 12),
        ("joint_vel", 12),
        ("prev_action", 12),
        ("gait_phase_sin_cos", 2),
    ],
    "scales": {
        "ang_vel_scale": 0.25,
        "dof_pos_scale": 1.0,
        "dof_vel_scale": 0.05,
        "action_scale": 0.25,
        "cmd_scale": [2.0, 2.0, 0.25],
    },
    "pd": {"kp": 20.0, "kd": 0.5},
    "control": {"simulation_dt": 0.005, "control_decimation": 4, "policy_hz": 50.0},
    "default_angles": [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    # policy joint order used by deploy_mujoco; SDK order differs and is remapped
    # via POLICY_TO_SDK_INDEX = [3,4,5,0,1,2,9,10,11,6,7,8] in sim_to_real.py
    "joint_order_policy": [
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
    ],
    "cycle_sec": 0.6,  # gait phase period; run name "gait05" suggests you trained with 0.5
}

DEFAULT_RUN_DIR = "~/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/2026-05-20_16-03-49_deploy_gait05_h100_seed42"

# config keys worth surfacing for contract alignment
GREP_KEYWORDS = [
    "scale", "action_scale", "kp", "kd", "stiffness", "damping",
    "default_joint", "default_angle", "init_state", "decimation", "dt",
    "command", "ranges", "gait", "phase", "cycle", "clip", "num_obs",
    "observation", "noise", "lin_vel", "ang_vel", "projected_gravity",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def hr(title: str = "") -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}" if title else line)


def find_checkpoints(run_dir: Path):
    ckpts = list(run_dir.glob("model_*.pt"))

    def iter_num(p: Path) -> int:
        m = re.search(r"model_(\d+)", p.stem)
        return int(m.group(1)) if m else -1

    return sorted(ckpts, key=iter_num)


def inspect_actor(ckpt_path: Path) -> dict:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        sd = blob["model_state_dict"]
        extra = {k: blob[k] for k in ("iter", "infos") if k in blob}
    elif isinstance(blob, dict) and "actor_state_dict" in blob:
        # rsl_rl modern layout: nested actor / critic / optimizer
        sd = blob["actor_state_dict"]
        extra = {k: blob[k] for k in ("iter", "infos") if k in blob}
    elif isinstance(blob, dict):
        # could already be a raw state_dict
        sd = blob
        extra = {}
    else:
        raise RuntimeError(f"Unexpected checkpoint object: {type(blob)}")

    # collect 2D Linear weights that belong to the actor's policy MLP.
    # Skip critic / value / obs_normalizer / distribution std parameters.
    layer_re = re.compile(r"(?:^|\.)(?:actor|mlp|net|policy|trunk)\.(\d+)\.weight$")
    skip_substrings = ("critic", "value", "normaliz", "distribution", "std")
    indexed, fallback = [], []
    for k, v in sd.items():
        if not hasattr(v, "ndim") or v.ndim != 2:
            continue
        kl = k.lower()
        if any(s in kl for s in skip_substrings):
            continue
        m = layer_re.search(k)
        if m:
            indexed.append((int(m.group(1)), k, tuple(v.shape)))
        else:
            fallback.append((k, tuple(v.shape)))

    if indexed:
        indexed.sort(key=lambda t: t[0])
        layers = [(k, shp) for _, k, shp in indexed]
    else:
        layers = fallback  # best effort, dict insertion order

    info = {"checkpoint": str(ckpt_path), **extra}
    if not layers:
        info["error"] = "Could not locate actor Linear weights in state_dict."
        info["state_dict_keys_sample"] = list(sd.keys())[:40]
        return info

    in_dim = layers[0][1][1]
    out_dim = layers[-1][1][0]
    hidden = [shp[0] for _, shp in layers[:-1]]
    info.update(
        num_obs=int(in_dim),
        num_actions=int(out_dim),
        hidden_dims=hidden,
        n_linear_layers=len(layers),
        layer_shapes=[(k, shp) for k, shp in layers],
        has_std=any("std" in k.lower() for k in sd.keys()),
        has_obs_normalizer=any("normaliz" in k.lower() for k in sd.keys()),
    )
    return info


def to_plain(obj, depth: int = 0):
    if depth > 8:
        return repr(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_plain(x, depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_plain(v, depth + 1) for k, v in obj.items()}
    if hasattr(obj, "to_dict"):
        try:
            return to_plain(obj.to_dict(), depth + 1)
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {k: to_plain(v, depth + 1) for k, v in vars(obj).items()
                if not k.startswith("_")}
    return repr(obj)


def load_all_configs(run_dir: Path) -> dict:
    out = {}
    if yaml is not None:
        for p in list(run_dir.rglob("*.yaml")) + list(run_dir.rglob("*.yml")):
            key = str(p.relative_to(run_dir))
            try:
                raw = yaml.load(p.read_text(), Loader=_TolerantLoader)
            except Exception as e:
                out[key] = f"<yaml error: {e}>"
                continue
            out[key] = to_plain(raw)
    for p in run_dir.rglob("*.json"):
        try:
            out[str(p.relative_to(run_dir))] = json.loads(p.read_text())
        except Exception as e:
            out[str(p.relative_to(run_dir))] = f"<json error: {e}>"
    for p in run_dir.rglob("*.pkl"):
        try:
            with open(p, "rb") as f:
                out[str(p.relative_to(run_dir))] = to_plain(pickle.load(f))
        except Exception as e:
            out[str(p.relative_to(run_dir))] = f"<pickle error: {e} (needs training env on path)>"
    return out


def flatten(obj, prefix="") -> dict:
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            items.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], (dict, list)):
        for i, v in enumerate(obj):
            items.update(flatten(v, f"{prefix}{i}."))
    else:
        items[prefix[:-1]] = obj
    return items


def grep_config(flat: dict) -> dict:
    hits = {}
    for k, v in flat.items():
        kl = k.lower()
        if any(kw in kl for kw in GREP_KEYWORDS):
            sv = repr(v)
            hits[k] = sv if len(sv) <= 120 else sv[:117] + "..."
    return hits


def find_obs_terms(plain) -> list | None:
    """Recursively locate an observations group with an actor/policy subgroup and
    return its ordered term names. mjlab uses `observations.actor.terms.<name>`;
    older isaaclab configs use `observations.policy.<name>` directly."""
    found = []
    META = {
        "concatenate_terms", "concatenate_dim", "enable_corruption",
        "history_length", "flatten_history_dim", "nan_policy",
        "nan_check_per_term",
    }

    def collect(group: dict) -> list:
        # mjlab style: { terms: { name: cfg, ... }, ...meta }
        if "terms" in group and isinstance(group["terms"], dict):
            return [k for k in group["terms"].keys() if k not in META]
        # legacy style: name keys directly under the group
        return [k for k in group.keys() if k not in META]

    def walk(o, in_obs: bool):
        if isinstance(o, dict):
            # Only treat actor/policy as obs term groups when nested inside
            # an `observations` dict — otherwise the rsl_rl agent.yaml's
            # `policy:` network-architecture block matches and pollutes the
            # result.
            if in_obs:
                for key in ("actor", "policy"):
                    sub = o.get(key)
                    if isinstance(sub, dict):
                        terms = collect(sub)
                        if terms:
                            found.append(terms)
            for k, v in o.items():
                walk(v, in_obs=(in_obs or k == "observations"))
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, in_obs)

    walk(plain, in_obs=False)
    return found[0] if found else None


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def mark(ok) -> str:
    return {True: "PASS", False: "FAIL", None: "CHECK"}[ok]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=DEFAULT_RUN_DIR)
    ap.add_argument("--checkpoint", default=None,
                    help="specific model_*.pt filename inside run_dir")
    ap.add_argument("--out", default=None, help="optional JSON report path")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    hr("RUN DIRECTORY")
    print(run_dir)
    if not run_dir.is_dir():
        sys.exit(f"[FATAL] run dir not found: {run_dir}")

    print("\nfiles:")
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(run_dir)}  ({p.stat().st_size} B)")

    # ---- checkpoint / actor ------------------------------------------------ #
    if args.checkpoint:
        ckpt = run_dir / args.checkpoint
    else:
        ckpts = find_checkpoints(run_dir)
        if not ckpts:
            sys.exit("[FATAL] no model_*.pt checkpoint found in run dir.")
        ckpt = ckpts[-1]

    hr("ACTOR NETWORK (from checkpoint)")
    actor = inspect_actor(ckpt)
    print(f"checkpoint : {actor.get('checkpoint')}")
    if "iter" in actor:
        print(f"iteration  : {actor['iter']}")
    if "error" in actor:
        print(f"[WARN] {actor['error']}")
        for k in actor.get("state_dict_keys_sample", []):
            print(f"   key: {k}")
    else:
        print(f"num_obs (input)   : {actor['num_obs']}")
        print(f"num_actions (out) : {actor['num_actions']}")
        print(f"hidden_dims       : {actor['hidden_dims']}")
        print(f"linear layers     : {actor['n_linear_layers']}")
        print(f"has action std    : {actor['has_std']}")
        print(f"has obs normalizer: {actor['has_obs_normalizer']}  "
              f"(if True, normalization stats must be baked into the exported policy)")
        print("layer shapes:")
        for k, shp in actor["layer_shapes"]:
            print(f"   {k:32s} {shp}")

    # ---- configs ----------------------------------------------------------- #
    hr("CONFIG FILES FOUND")
    cfgs = load_all_configs(run_dir)
    for name in cfgs:
        print(f"  {name}")
    if not cfgs:
        print("  (none - rsl_rl usually writes params/env.yaml and params/agent.yaml;")
        print("   if missing, re-export or point to the params/ dir manually)")

    hr("CONFIG VALUES RELEVANT TO DEPLOY CONTRACT")
    obs_terms = None
    for name, cfg in cfgs.items():
        if isinstance(cfg, str):  # error string
            continue
        flat = flatten(cfg)
        hits = grep_config(flat)
        if hits:
            print(f"\n--- {name} ---")
            for k in sorted(hits):
                print(f"  {k} = {hits[k]}")
        terms = find_obs_terms(cfg)
        if terms and obs_terms is None:
            obs_terms = terms

    hr("OBSERVATION TERM ORDER (training)  vs  DEPLOY LAYOUT")
    print("deploy expects (ordered):")
    for n, d in DEPLOY_CONTRACT["obs_layout"]:
        print(f"   {n:24s} dim {d}")
    print(f"   ---> total {DEPLOY_CONTRACT['num_obs']}")
    if obs_terms:
        print("\ntraining policy obs terms (in config order):")
        for t in obs_terms:
            print(f"   {t}")
        print("\n[!] Verify these map 1:1 and IN THE SAME ORDER as the deploy layout.")
        print("    Order mismatches do not change num_obs but break the policy on deploy.")
    else:
        print("\n[!] Could not auto-extract obs term order from config.")
        print("    Open params/env.yaml manually and read observations.policy term order.")

    # ---- comparison + checklist ------------------------------------------- #
    hr("ALIGNMENT CHECKLIST")
    c = DEPLOY_CONTRACT
    obs_ok = actor.get("num_obs") == c["num_obs"] if "num_obs" in actor else None
    act_ok = actor.get("num_actions") == c["num_actions"] if "num_actions" in actor else None

    print(f"[{mark(obs_ok)}] num_obs:     training={actor.get('num_obs')}  deploy={c['num_obs']}")
    print(f"[{mark(act_ok)}] num_actions: training={actor.get('num_actions')}  deploy={c['num_actions']}")
    print(f"[CHECK] obs scales must equal deploy YAML: {c['scales']}")
    print(f"[CHECK] PD gains must equal deploy YAML:   kp={c['pd']['kp']} kd={c['pd']['kd']}")
    print(f"[CHECK] default joint angles must equal:   {c['default_angles']}")
    print(f"[CHECK] policy joint order must equal:     {c['joint_order_policy']}")
    print(f"[CHECK] control rate: sim_dt={c['control']['simulation_dt']} "
          f"decimation={c['control']['control_decimation']} -> {c['control']['policy_hz']} Hz")
    print(f"[CHECK] gait cycle_sec: deploy default={c['cycle_sec']}  "
          f"(verify against training: phase term `period` in observations.actor.terms.phase.params)")
    print(f"[CHECK] action clipping / normalization baked into exported .pt")

    print("\nNext: fix any FAIL, manually confirm every CHECK by editing the deploy")
    print("YAML to match the printed training values, THEN move to Step 1 (export .pt).")

    # ---- optional JSON ----------------------------------------------------- #
    if args.out:
        report = {
            "run_dir": str(run_dir),
            "actor": {k: v for k, v in actor.items() if k != "layer_shapes"},
            "actor_layers": actor.get("layer_shapes"),
            "obs_terms": obs_terms,
            "deploy_contract": c,
        }
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nJSON report written to {args.out}")


if __name__ == "__main__":
    main()