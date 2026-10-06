from __future__ import annotations

import time

import numpy as np
import pytest
import torch

import planner

_HAVE = all((planner._MODEL / name).is_file() for name in planner._REQUIRED)
_MIN_FREE_GIB = 30.0
_INSTRUCTION = (
    "Walk down the hallway past the yellow painting and turn right at the end. Go through the open door into the "
    "bedroom and stop next to the bed."
)

pytestmark = [
    pytest.mark.skipif(not _HAVE, reason=f"no checkpoint in {planner._MODEL}, fetch the streamvln weights"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="StreamVLN needs a CUDA GPU"),
]


@pytest.fixture(scope="module")
def vln() -> planner.StreamVln:
    free, _ = torch.cuda.mem_get_info()
    if free / 2**30 < _MIN_FREE_GIB:
        pytest.skip(f"only {free / 2**30:.1f} GiB GPU memory free, need {_MIN_FREE_GIB}")
    device = torch.device("cuda")
    started = time.monotonic()
    model = planner.load_model(planner._MODEL, device)
    print(f"\nload {time.monotonic() - started:.1f} s, {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated")
    return planner.StreamVln(model, planner.load_tokenizer(planner._MODEL), device)


def _frame(frames: list[np.ndarray], step: int) -> np.ndarray:
    return frames[min(step // 10, len(frames) - 1)]


def test_episode_emits_valid_primitives_until_stop_or_budget(vln: planner.StreamVln, frames: list[np.ndarray]):
    vln.reset({"instruction": _INSTRUCTION, "signal": "arrived"})
    torch.cuda.reset_peak_memory_stats()
    steps, calls = [], []
    for _ in range(40):
        step_id, calls_model = vln.episode.step_id, vln.episode.idle
        step = vln.step({"rgb": _frame(frames, step_id), "robot_pose": np.zeros(3)})
        if calls_model:
            calls.append((step_id, vln.last_output, vln.last_latency_s))
        steps.append(step)
        if step.signal:
            break
    for step_id, output, latency in calls:
        print(f"step {step_id:3d} {latency:6.3f} s {output!r}")
    print(f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, {len(steps)} steps")
    assert all(s.command in {"forward", "left", "right"} or s == planner.Step(signal="arrived") for s in steps)
    assert all(s.amount == 0.0 and not s.action and not s.chunk for s in steps)
    assert calls[0][0] == 0
    if steps[-1].signal:
        assert vln.step({"rgb": frames[0]}) == planner.Step(signal="arrived")


def test_memory_window_call_after_32_steps(vln: planner.StreamVln, frames: list[np.ndarray]):
    vln.reset({"instruction": _INSTRUCTION})
    for step_id in range(32):
        vln.episode.observe(planner.preprocess(vln.processor, _frame(frames, step_id)))
        if vln.episode.idle:
            vln.episode.plan(["forward"] * 4)
        vln.episode.advance()
    vln.episode.observe(planner.preprocess(vln.processor, frames[-1]))
    turn = vln.episode.turn()
    assert turn.with_memory
    assert len(turn.indices) == planner.NUM_HISTORY + 1
    output = vln.generate(turn)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"\nmemory turn {vln.last_latency_s:.3f} s {output!r}, peak {peak:.2f} GiB")
    assert all(a in {"forward", "left", "right", planner.STOP} for a in planner.parse_actions(output))
    vln.episode.plan(planner.parse_actions(output))
    vln.episode.advance()


def test_empty_instruction_stands_still_without_the_model(vln: planner.StreamVln, frames: list[np.ndarray]):
    vln.reset({"instruction": "  ", "signal": "arrived"})
    assert vln.step({"rgb": frames[0]}) == planner.Step()
    assert vln.episode.frames == []
