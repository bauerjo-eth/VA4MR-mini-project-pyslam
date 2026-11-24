"""
Phase 1 components: the parts you design. The benchmark (geomatch/bench) trains and evaluates
whatever is registered here; an experiment file (geomatch/experiments/*.yaml) picks by name.

  backbones.py          keypoints + visual descriptor          (SIFT, SuperPoint, ...)
  depth_descriptors.py  geometric (hand-designed) depth descriptors, learned-descriptor
                        networks and their input channels
  filters.py            learned match filters (WP3); hand-designed filters live in
                        geomatch/filtering.py (@register)

Importing this package registers everything.
"""

from geomatch.components import backbones, depth_descriptors, filters  # noqa: F401
