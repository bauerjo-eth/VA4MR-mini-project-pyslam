"""
Match filters (WP3): decide which putative matches to keep before RANSAC.

1. HAND-DESIGNED filters are functions in geomatch/filtering.py:

    @register("my_filter")
    def my_filter(ctx, **params) -> np.ndarray:   # bool mask over ctx.i1 / ctx.i2
        ...

   In the experiment file: {type: my_filter, <params>}.

2. LEARNED filters: a per-match feature function + a model. The benchmark collects features
   and labels (GT epipolar error < grading.correct_px) on the TRAIN split, trains one model per
   descriptor combination (descriptor distances differ between combinations) and evaluates it
   on the TEST split.

    @learned_filter("my_filter")
    class MyFilter:
        def features(self, ctx):          # -> N x F float32, one row per match
            ...
        def make_model(self, in_dim):     # -> nn.Module mapping B x F -> B logits
            ...
        train = dict(lr=1e-3, epochs=30, batch=4096)

   In the experiment file: {learned: my_filter, thresh: 0.5}.
"""

LEARNED_FILTERS = {}


def learned_filter(name):
    def deco(cls):
        LEARNED_FILTERS[name] = cls
        return cls
    return deco


@learned_filter("mlp")
class MlpFilter:
    """David (WP3): MLP (64, 32) over 22 hand-made per-match features: depths, relative depth,
    descriptor distance, depth window statistics, depth gradient and discontinuity, position.
    Class-balanced BCE loss, as in train_filter.py."""
    train = dict(lr=1e-3, epochs=30, batch=4096)

    def features(self, ctx):
        from geomatch.learned_filter import build_mlp_features
        return build_mlp_features(ctx)

    def make_model(self, in_dim):
        from geomatch.learned_filter import MLP
        return MLP(in_dim=in_dim, hidden=(64, 32))
