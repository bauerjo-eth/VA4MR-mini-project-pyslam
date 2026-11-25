# rpg_trajectory_evaluation (vendored)

- unmodified copy of `src/rpg_trajectory_evaluation/` + `LICENSE` (MIT) from
  https://github.com/uzh-rpg/rpg_trajectory_evaluation, commit `8c8ceec55c5c5094a6494208cfc5f54afe0bbc4d` (2022-09-14)
- Z. Zhang, D. Scaramuzza: "A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry", IROS 2018
- used by `baselines/slam_rte.py` (alignment, ATE, relative errors); runs under Python 3 as is
- not used: `plot_utils.py` (needs LaTeX), `multiple_traj_errors.py`
- extra dependencies (in the pySLAM env): `colorama`, `pyyaml`
