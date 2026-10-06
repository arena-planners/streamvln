from __future__ import annotations

import pytest

import planner

_FILES = ("tokenizer.json", "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json")
_HAVE = all((planner._MODEL / name).is_file() for name in _FILES)

pytestmark = pytest.mark.skipif(not _HAVE, reason=f"no tokenizer in {planner._MODEL}, fetch the streamvln weights")

_INSTRUCTION = "Walk past the couch and stop at the door."
_SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
_REPLY = "<|im_start|>assistant\n<|im_end|>\n"


@pytest.fixture(scope="module")
def tokenizer():
    return planner.load_tokenizer(planner._MODEL)


def _decode(tokenizer, ids: list[int]) -> str:
    back = {
        planner.IMAGE_TOKEN_INDEX: tokenizer.convert_tokens_to_ids(planner.IMAGE_TOKEN),
        planner.MEMORY_TOKEN_INDEX: tokenizer.convert_tokens_to_ids(planner.MEMORY_TOKEN),
    }
    return tokenizer.decode([back.get(i, i) for i in ids])


def test_special_tokens_take_the_upstream_ids(tokenizer):
    assert tokenizer.convert_tokens_to_ids(planner.IMAGE_TOKEN) == 151646
    assert tokenizer.convert_tokens_to_ids(planner.MEMORY_TOKEN) == 151647
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") == 151645


def test_window_opening_turn_encodes_system_user_and_empty_reply(tokenizer):
    human = planner.human_turn(_INSTRUCTION, opens_window=True, with_memory=False)
    ids = planner.encode_turn(tokenizer, human, opens_window=True)
    assert _decode(tokenizer, ids) == _SYSTEM + "<|im_start|>user\n" + human + "<|im_end|>\n" + _REPLY
    assert ids.count(planner.IMAGE_TOKEN_INDEX) == 1
    assert planner.MEMORY_TOKEN_INDEX not in ids
    whole = tokenizer(_SYSTEM + "<|im_start|>user\n" + human + "<|im_end|>\n" + _REPLY).input_ids
    image = tokenizer.convert_tokens_to_ids(planner.IMAGE_TOKEN)
    assert ids == [planner.IMAGE_TOKEN_INDEX if i == image else i for i in whole]


def test_memory_turn_places_memory_before_the_frame(tokenizer):
    human = planner.human_turn(_INSTRUCTION, opens_window=True, with_memory=True)
    ids = planner.encode_turn(tokenizer, human, opens_window=True)
    assert ids.count(planner.MEMORY_TOKEN_INDEX) == 1
    assert ids.count(planner.IMAGE_TOKEN_INDEX) == 1
    assert ids.index(planner.MEMORY_TOKEN_INDEX) < ids.index(planner.IMAGE_TOKEN_INDEX)


def test_continuing_turn_has_no_system_message(tokenizer):
    human = planner.human_turn(_INSTRUCTION, opens_window=False, with_memory=False)
    ids = planner.encode_turn(tokenizer, human, opens_window=False)
    assert _decode(tokenizer, ids) == "<|im_start|>user\nyou can see <image>.<|im_end|>\n" + _REPLY


def test_arrows_round_trip_through_the_tokenizer(tokenizer):
    text = "\u2191\u2190\u2192STOP"
    assert tokenizer.decode(tokenizer(text).input_ids) == text
    assert planner.parse_actions(tokenizer.decode(tokenizer(text).input_ids)) == ["forward", "left", "right", "stop"]
