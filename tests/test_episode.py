from __future__ import annotations

import pytest

import planner

_FOUR = ["forward", "left", "forward", "right"]


def _run(episode: planner.Episode, chunk: list[str], steps: int) -> list[planner.Turn]:
    turns = []
    while episode.step_id < steps:
        episode.observe(f"frame{episode.step_id}")
        if episode.idle:
            turn = episode.turn()
            turns.append(turn)
            episode.plan(list(chunk))
        episode.advance()
    return turns


def test_first_turn_opens_a_window_without_memory():
    episode = planner.Episode()
    episode.observe("frame0")
    assert episode.turn() == planner.Turn(indices=[0], time_ids=[0], opens_window=True, with_memory=False)


def test_four_action_chunks_call_every_fourth_step_and_continue_the_window():
    episode = planner.Episode()
    turns = _run(episode, _FOUR, 32)
    assert [t.indices[-1] for t in turns] == list(range(0, 32, 4))
    assert [t.opens_window for t in turns] == [True] + [False] * 7
    assert not any(t.with_memory for t in turns)
    assert turns[3].time_ids == list(range(13))
    assert episode.frames == [f"frame{i}" for i in range(32)]


def test_window_closes_every_32_steps_and_reopens_with_eight_memory_frames():
    episode = planner.Episode()
    turns = _run(episode, _FOUR, 100)
    openers = [t for t in turns if t.opens_window]
    assert [t.indices[-1] for t in openers] == [0, 32, 64, 96]
    assert openers[1] == planner.Turn(
        indices=[0, 4, 8, 12, 16, 20, 24, 28, 32], time_ids=[32], opens_window=True, with_memory=True
    )
    assert openers[2].indices == [0, 8, 16, 24, 32, 40, 48, 56, 64]
    assert openers[3].indices == [0, 12, 24, 36, 48, 60, 72, 84, 96]
    assert all(len(t.indices) == planner.NUM_HISTORY + 1 for t in openers[1:])
    assert all(episode.frames[i] == f"frame{i}" for t in openers for i in t.indices)


def test_window_closed_mid_chunk_reopens_at_the_next_call_with_memory():
    episode = planner.Episode()
    turns = _run(episode, ["forward"] * 3, 40)
    opener = next(t for t in turns[1:] if t.opens_window)
    assert opener.indices[-1] == 33
    assert opener.time_ids == [32, 33]
    assert opener.indices[:-1] == list(range(0, 32, 4))
    assert opener.with_memory


def test_stop_latches_and_drops_the_rest_of_the_chunk():
    episode = planner.Episode()
    episode.observe("frame0")
    episode.plan(["forward", planner.STOP, "left"])
    assert episode.advance() == "forward"
    episode.observe("frame1")
    assert episode.advance() == planner.STOP
    assert episode.stopped
    assert episode.idle
    assert episode.step_id == 1


def test_empty_plan_is_stop():
    episode = planner.Episode()
    episode.observe("frame0")
    episode.plan([])
    assert episode.advance() == planner.STOP
    assert episode.stopped


def test_second_frame_for_the_same_step_is_an_error():
    episode = planner.Episode()
    episode.observe("frame0")
    with pytest.raises(RuntimeError, match="observed at step 0"):
        episode.observe("again")


def test_reset_clears_every_counter():
    episode = planner.Episode()
    _run(episode, _FOUR, 34)
    episode.plan([planner.STOP])
    episode.advance()
    episode.reset()
    assert (episode.step_id, episode.frames, episode.time_ids, episode.stopped) == (0, [], [], False)
    assert episode.idle
    assert episode.opens_window
