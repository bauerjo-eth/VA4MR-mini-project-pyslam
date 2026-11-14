"""
Compatibility only: the functions that used to live here moved to

  descriptors.py   detect, depth_at, depth_descriptor, concat
  matchers.py      match

They are re-imported here so that `from geomatch.core import ...` (used by demo.py) keeps
working. New code should import from descriptors.py and matchers.py directly.
"""

from geomatch.descriptors import concat, depth_at, depth_descriptor, detect
from geomatch.matchers import match

__all__ = ["concat", "depth_at", "depth_descriptor", "detect", "match"]
