"""
Phase 1 benchmark: one command trains what is missing and evaluates the whole grid.

    python -m geomatch.bench.run geomatch/experiments/phase1.yaml

Steps (each skipped when its result already exists):
  1. learned depth descriptors: train one per backbone x learned depth descriptor (TRAIN split)
  2. learned filters: train one per backbone x depth descriptor x learned filter (TRAIN split)
  3. evaluate every backbone x depth descriptor x filter at every gap on the TEST split
  4. report: geomatch/results/<name>_<timestamp>/report.md (+ grid.csv, per_pair.csv, figure)

Options:
  --backbones / --depth / --filters NAME...   run only part of the grid
  --gaps G...                                  override the test gaps
  --retrain NAME...                            retrain models of these depth descriptors or
                                               filters (after changing their code)
  --no-train                                   skip grid cells whose model is missing
  --quick N                                    smoke test: only the first N test anchor frames
"""

import argparse
import logging
import shutil
from datetime import datetime

import torch
import yaml

import geomatch.components  # noqa: F401  (registers all components)
from geomatch.bench.common import (RESULTS, desc_model_path, filter_model_path, is_learned,
                                   load_experiment)
from geomatch.bench.evaluate import evaluate
from geomatch.bench.report import write_report
from geomatch.bench.train_descriptor import train_descriptor
from geomatch.bench.train_filter import train_filters

log = logging.getLogger("geomatch")


def subset(exp, key, names):
    if names:
        missing = set(names) - set(exp[key])
        if missing:
            raise KeyError(f"{key}: {sorted(missing)} not in the experiment file")
        exp[key] = {k: v for k, v in exp[key].items() if k in names}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    ap.add_argument("--backbones", nargs="+")
    ap.add_argument("--depth", nargs="+")
    ap.add_argument("--filters", nargs="+")
    ap.add_argument("--gaps", nargs="+", type=int)
    ap.add_argument("--retrain", nargs="+", default=[])
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--quick", type=int, default=None)
    args = ap.parse_args()

    exp = load_experiment(args.experiment)
    for key, names in (("backbones", args.backbones), ("depth_descriptors", args.depth), ("filters", args.filters)):
        subset(exp, key, names)
    if "none" not in exp["depth_descriptors"]:
        exp["depth_descriptors"] = {"none": None, **exp["depth_descriptors"]}  # baseline is always needed
    gaps = args.gaps or exp["data"]["test_pairs"]["gaps"]
    tag = f"_quick{args.quick}" if args.quick else ""
    out_dir = RESULTS / f"{exp['name']}{tag}_{datetime.now():%Y-%m-%d_%H%M%S}"
    out_dir.mkdir(parents=True)
    shutil.copy(args.experiment, out_dir / "experiment.yaml")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "log.txt")])
    log.info(f"results: {out_dir}")
    info = {}

    # 1. learned depth descriptors
    for b in list(exp["backbones"]):
        for d, spec in list(exp["depth_descriptors"].items()):
            if not is_learned(spec):
                continue
            path = desc_model_path(exp, b, d)
            if d in args.retrain and path.exists():
                path.unlink()
            if not path.exists():
                if args.no_train:
                    log.warning(f"no model for {b}+{d} ({path.name}); --no-train: dropped from the grid")
                    exp.setdefault("_skip", set()).add((b, d))
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                train_descriptor(exp, b, d, path)
            ck = torch.load(path, map_location="cpu", weights_only=False)
            info[f"descriptor {b}+{d}"] = (f"`{path.name}`, epoch {ck['epoch']}, val NN accuracy "
                                           f"{ck['val']['all']['acc']:.4f} (visual alone {ck['val']['all']['acc_visual']:.4f})")

    skip = exp.pop("_skip", set())
    if skip:  # a learned descriptor without model: drop it for every backbone (keeps the grid square)
        for _, d in skip:
            exp["depth_descriptors"].pop(d, None)

    # 2. learned filters
    jobs = []
    for b in exp["backbones"]:
        for d, dspec in exp["depth_descriptors"].items():
            for f, fspec in exp["filters"].items():
                if not is_learned(fspec):
                    continue
                path = filter_model_path(exp, b, d, f)
                if (f in args.retrain or d in args.retrain) and path.exists():
                    path.unlink()
                if not path.exists():
                    jobs.append((b, d, f, desc_model_path(exp, b, d) if is_learned(dspec) else None))
    if jobs and args.no_train:
        raise SystemExit(f"--no-train but {len(jobs)} filter model(s) are missing: {[j[:3] for j in jobs]}")
    if jobs:
        train_filters(exp, jobs)
    for b in exp["backbones"]:
        for d in exp["depth_descriptors"]:
            for f, fspec in exp["filters"].items():
                if is_learned(fspec):
                    ck = torch.load(filter_model_path(exp, b, d, f), map_location="cpu", weights_only=False)
                    v = ck["val"]
                    info[f"filter {b}+{d}+{f}"] = (f"val AUC {v['auc']:.3f}, keeps {v['keep_ratio']:.0%}, "
                                                   f"precision {v['precision_all']:.3f} -> {v['precision_kept']:.3f}")

    # 3. + 4. evaluate and report
    (out_dir / "experiment_effective.yaml").write_text(yaml.safe_dump(exp, sort_keys=False))
    rows, summ = evaluate(exp, gaps, args.quick)
    write_report(exp, out_dir, rows, summ, gaps, info)
    log.info(f"report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
