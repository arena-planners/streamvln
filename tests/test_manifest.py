from __future__ import annotations

from arena_planners import goal, weights
from arena_planners.bridge.discrete import Primitives, primitives_from_manifest
from arena_planners.sdk import load_manifest

import planner


def test_manifest_declares_vlnce_primitives_and_an_instruction_goal():
    manifest = load_manifest(planner._DIR / "planner.yaml")
    assert manifest["action_type"] == "discrete"
    assert primitives_from_manifest(manifest) == Primitives(forward_m=0.25, turn_deg=15.0)
    assert goal.declared(manifest) == frozenset({"instruction"})
    assert manifest["signals"] == ["arrived"]
    assert manifest["observations"]["datasources"]["rgb"]["params"] == {"sensor": "image", "output_size": [640, 480]}


def test_weights_pin_every_file_the_loader_requires():
    entries = weights.read(planner._DIR)
    assert {entry.dest for entry in entries} >= {f"model/{name}" for name in planner._REQUIRED}
    assert all(entry.revision for entry in entries)
