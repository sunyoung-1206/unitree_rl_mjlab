"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    # Pinned below mjlab's own (loose) requirements because the GPU coupling
    # patch in vendor/mujoco_warp_3.6.0_patch/ targets mujoco_warp 3.6.0's
    # internal API. Newer mujoco-warp (>=3.9.1) removed `ls_parallel`, and
    # newer warp-lang (>=1.13) removed `wp.context.runtime` that mjlab 1.2.0
    # reads for CUDA graph support.
    "mujoco-warp==3.6.0",
    "mujoco==3.6.0",
    "warp-lang==1.12.1",
]

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=["src"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
