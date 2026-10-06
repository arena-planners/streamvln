from __future__ import annotations

import pytest

import planner

_INSTRUCTION = "Walk past the couch and stop at the door."
_TASK = (
    "You are an autonomous navigation assistant. Your task is to Walk past the couch and stop at the door. Devise an "
    "action sequence to follow the instruction using the four actions: "
    "TURN LEFT (\u2190) or TURN RIGHT (\u2192) by 15 degrees, MOVE FORWARD (\u2191) by 25 centimeters, or STOP."
)


def test_prompt_arrows_are_the_upstream_code_points():
    assert "TURN LEFT (\u2190)" in planner.PROMPT
    assert "TURN RIGHT (\u2192)" in planner.PROMPT
    assert "MOVE FORWARD (\u2191)" in planner.PROMPT
    assert [hex(ord(c)) for c in planner.PROMPT if ord(c) > 127] == ["0x2190", "0x2192", "0x2191"]


def test_window_opening_turn_at_step_zero():
    text = planner.human_turn(_INSTRUCTION, opens_window=True, with_memory=False)
    assert text == _TASK + " you can see <image>."


def test_window_opening_turn_with_memory():
    text = planner.human_turn(_INSTRUCTION, opens_window=True, with_memory=True)
    assert text == _TASK + " These are your historical observations <memory>. you can see <image>."


@pytest.mark.parametrize("with_memory", [False, True])
def test_continuing_turn_is_only_the_new_frame(with_memory: bool):
    assert planner.human_turn(_INSTRUCTION, opens_window=False, with_memory=with_memory) == "you can see <image>."


def test_instruction_replaces_the_placeholder_and_its_period():
    text = planner.human_turn("go to the kitchen", opens_window=True, with_memory=False)
    assert "Your task is to go to the kitchen Devise an action sequence" in text
    assert "<instruction>" not in text


def test_chat_template_has_no_system_message():
    assert "system" not in planner.CHAT_TEMPLATE
    assert planner.CHAT_TEMPLATE.startswith("{% for message in messages %}{{'<|im_start|>' + message['role']")
