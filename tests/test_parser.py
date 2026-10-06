from __future__ import annotations

import pytest

import planner


@pytest.mark.parametrize(
    ("text", "actions"),
    [
        ("<|im_start|>assistant\n\u2192\u2192\u2192\u2192<|im_end|>", ["right"] * 4),
        ("\u2191\u2191\u2190\u2192", ["forward", "forward", "left", "right"]),
        ("\u2191\u2191STOP", ["forward", "forward", "stop"]),
        ("STOP", ["stop"]),
        ("<|im_start|>assistant\n<|im_end|>", ["stop"]),
        ("", ["stop"]),
        ("move forward then stop", ["stop"]),
        ("\u2193\u2191", ["forward"]),
    ],
)
def test_parse_actions_matches_upstream_findall(text: str, actions: list[str]):
    assert planner.parse_actions(text) == actions


@pytest.mark.parametrize(("action", "command"), [("forward", "forward"), ("left", "left"), ("right", "right")])
def test_moves_map_to_commands_with_the_primitive_amount(action: str, command: str):
    assert planner.action_step(action, "arrived") == planner.Step(command=command)


@pytest.mark.parametrize(("signal", "expected"), [("arrived", planner.Step(signal="arrived")), ("", planner.Step())])
def test_stop_sends_the_expected_signal_or_stands_still(signal: str, expected: planner.Step):
    assert planner.action_step(planner.STOP, signal) == expected
