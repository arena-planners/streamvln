"""StreamVLN (video LLM on Qwen2-7B, StreamVLN_Video_qwen_1_5 v1_3) wrapper for the arena_planners bridge."""

from __future__ import annotations

import collections
import dataclasses
import logging
import pathlib
import re
import time
import typing

import numpy as np
import torch
import transformers
from arena_planners.sdk import Step, load_manifest, main_loop
from PIL import Image

from model_streamvln import SigLipImageProcessor, StreamVLNForCausalLM

_DIR = pathlib.Path(__file__).parent
_MODEL = _DIR / "model"
_FETCH_HINT = "run `arena feature planners add streamvln` (or `python -m arena_planners fetch streamvln`)"
_REQUIRED = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)

PROMPT = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow "
    "the instruction using the four actions: TURN LEFT (\u2190) or TURN RIGHT (\u2192) by 15 degrees, "
    "MOVE FORWARD (\u2191) by 25 centimeters, or STOP."
)
MEMORY_PROMPT = " These are your historical observations <memory>."
CONJUNCTION = "you can see "
IMAGE_TOKEN = "<image>"
MEMORY_TOKEN = "<memory>"
IMAGE_TOKEN_INDEX = -200
MEMORY_TOKEN_INDEX = -300
SYSTEM_MESSAGE = "You are a helpful assistant."
CHAT_TEMPLATE = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + "
    "'\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

STOP = "stop"
ACTIONS: dict[str, str] = {"STOP": STOP, "\u2191": "forward", "\u2190": "left", "\u2192": "right"}
_ACTION_PATTERN = re.compile("|".join(re.escape(token) for token in ACTIONS))

NUM_FRAMES = 32
NUM_HISTORY = 8
MAX_NEW_TOKENS = 10000
MODEL_MAX_LENGTH = 4096

_log = logging.getLogger("streamvln")


def human_turn(instruction: str, opens_window: bool, with_memory: bool) -> str:
    """User message of one model call: the full task prompt when a window opens, else only the new frame."""
    image = f"{CONJUNCTION}{IMAGE_TOKEN}."
    if not opens_window:
        return image
    text = PROMPT + (MEMORY_PROMPT if with_memory else "")
    return text.replace("<instruction>.", instruction) + " " + image


def parse_actions(text: str) -> list[str]:
    """Actions in the order the model wrote them, `[STOP]` when it wrote none."""
    return [ACTIONS[token] for token in _ACTION_PATTERN.findall(text)] or [STOP]


def action_step(action: str, signal: str) -> Step:
    """One primitive as a bridge command, STOP as the expected signal (none: stand still)."""
    return Step(signal=signal) if action == STOP else Step(command=action)


@dataclasses.dataclass(frozen=True)
class Turn:
    """One model call: frame indices to encode (memory first, current last), their time ids and the prompt kind."""

    indices: list[int]
    time_ids: list[int]
    opens_window: bool
    with_memory: bool


class Episode:
    """Step, window and memory bookkeeping of upstream's `eval_action` loop, one frame per executed primitive."""

    def __init__(self, num_frames: int = NUM_FRAMES, num_history: int = NUM_HISTORY) -> None:
        self.num_frames = num_frames
        self.num_history = num_history
        self.reset()

    def reset(self) -> None:
        self.step_id = 0
        self.frames: list[typing.Any] = []
        self.time_ids: list[int] = []
        self.queue: collections.deque[str] = collections.deque()
        self.opens_window = True
        self.stopped = False

    @property
    def idle(self) -> bool:
        return not self.queue

    def observe(self, frame: typing.Any) -> None:
        """Record the frame of the current step, at most one per step."""
        if len(self.frames) != self.step_id:
            raise RuntimeError(f"streamvln: frame {len(self.frames)} observed at step {self.step_id}")
        self.frames.append(frame)
        self.time_ids.append(self.step_id)

    def turn(self) -> Turn:
        with_memory = self.opens_window and self.step_id != 0
        indices = [self.step_id]
        if with_memory:
            start = self.time_ids[0]
            indices = list(range(0, start, start // self.num_history)) + indices
        return Turn(indices, list(self.time_ids), self.opens_window, with_memory)

    def plan(self, actions: list[str]) -> None:
        self.queue.extend(actions or [STOP])
        self.opens_window = False

    def advance(self) -> str:
        """Pop the next action, latching STOP and closing the window after every `num_frames` executed steps."""
        action = self.queue.popleft()
        if action == STOP:
            self.queue.clear()
            self.stopped = True
            return action
        self.step_id += 1
        if self.step_id % self.num_frames == 0:
            self.time_ids = []
            self.opens_window = True
        return action


def preprocess(processor: SigLipImageProcessor, rgb: np.ndarray) -> torch.Tensor:
    """SigLIP pixel values `(3, 384, 384)` in bf16, the frame squashed to a square as upstream."""
    image = Image.fromarray(np.ascontiguousarray(np.asarray(rgb)[..., :3], dtype=np.uint8)).convert("RGB")
    return processor.preprocess(images=image, return_tensors="pt")["pixel_values"][0].to(torch.bfloat16)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def check_model_dir(model_dir: pathlib.Path) -> None:
    missing = [name for name in _REQUIRED if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"streamvln files missing in {model_dir}: {missing}, {_FETCH_HINT}")


def load_tokenizer(model_dir: pathlib.Path) -> transformers.PreTrainedTokenizerBase:
    """Upstream's tokenizer with `<image>` and `<memory>` added and the per-message Qwen chat template."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_dir, model_max_length=MODEL_MAX_LENGTH, padding_side="right"
    )
    tokenizer.add_tokens([IMAGE_TOKEN], special_tokens=True)
    tokenizer.add_tokens([MEMORY_TOKEN], special_tokens=True)
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def encode_turn(tokenizer: transformers.PreTrainedTokenizerBase, human: str, opens_window: bool) -> list[int]:
    """Upstream's `preprocess_qwen` ids for one call, `<image>` and `<memory>` replaced by their sentinel indices."""
    ids: list[int] = []
    if opens_window:
        ids += tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_MESSAGE}])
    ids += tokenizer.apply_chat_template([{"role": "user", "content": human}])
    ids += tokenizer.apply_chat_template([{"role": "assistant", "content": ""}])
    sentinels = {
        tokenizer.convert_tokens_to_ids(IMAGE_TOKEN): IMAGE_TOKEN_INDEX,
        tokenizer.convert_tokens_to_ids(MEMORY_TOKEN): MEMORY_TOKEN_INDEX,
    }
    return [sentinels.get(token, token) for token in ids]


def load_model(model_dir: pathlib.Path, device: torch.device) -> StreamVLNForCausalLM:
    check_model_dir(model_dir)
    config = transformers.AutoConfig.from_pretrained(model_dir)
    model = StreamVLNForCausalLM.from_pretrained(
        model_dir,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=False,
    )
    model.model.num_history = NUM_HISTORY
    model.reset(1)
    model.requires_grad_(False)
    return model.to(device).eval()


class StreamVln:
    def __init__(
        self,
        model: StreamVLNForCausalLM,
        tokenizer: transformers.PreTrainedTokenizerBase,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.processor = model.get_vision_tower().image_processor
        self.episode = Episode()
        self.last_output = ""
        self.last_latency_s = 0.0
        self._warned: set[str] = set()
        self.reset(None)

    def reset(self, initial_state: dict | None) -> None:
        self.instruction = ""
        self.signal = ""
        self.output_ids: torch.Tensor | None = None
        self.past_key_values: typing.Any = None
        self.episode.reset()
        self.model.reset_for_env(0)
        if initial_state is None:
            return
        self.signal = str(initial_state.get("signal", ""))
        instruction = str(initial_state.get("instruction", ""))
        if instruction.strip():
            self.instruction = instruction
        else:
            self._warn_once("empty", "streamvln: empty instruction for this episode, standing still")

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            _log.warning(msg)

    @torch.inference_mode()
    def generate(self, turn: Turn) -> str:
        """Run one call of the streaming dialogue and return the decoded answer."""
        if turn.opens_window:
            self.model.reset_for_env(0)
            self.output_ids = None
            self.past_key_values = None
        human = human_turn(self.instruction, turn.opens_window, turn.with_memory)
        ids = encode_turn(self.tokenizer, human, turn.opens_window)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        if self.output_ids is not None:
            input_ids = torch.cat([self.output_ids, input_ids], dim=1)
        images = torch.stack([self.episode.frames[i] for i in turn.indices]).unsqueeze(0).to(self.device)
        started = time.monotonic()
        outputs = self.model.generate(
            inputs=input_ids,
            images=images,
            env_id=0,
            time_ids=[turn.time_ids],
            do_sample=False,
            num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            return_dict_in_generate=True,
            past_key_values=self.past_key_values,
        )
        self.last_latency_s = time.monotonic() - started
        self.output_ids = outputs.sequences
        self.past_key_values = outputs.past_key_values
        self.last_output = self.tokenizer.batch_decode(self.output_ids, skip_special_tokens=False)[0].strip()
        return self.last_output

    def step(self, features: dict) -> Step:
        if self.episode.stopped:
            return Step(signal=self.signal)
        if not self.instruction:
            return Step()
        rgb = features.get("rgb")
        if rgb is None or np.asarray(rgb).size == 0:
            self._warn_once("rgb", "streamvln: rgb missing, standing still")
            return Step()
        self.episode.observe(preprocess(self.processor, rgb))
        if self.episode.idle:
            self.episode.plan(parse_actions(self.generate(self.episode.turn())))
            _log.info("streamvln: %r in %.3f s", self.last_output, self.last_latency_s)
        return action_step(self.episode.advance(), self.signal)


_planner: StreamVln | None = None


def on_init(config: dict) -> None:
    global _planner
    device = select_device(str(config.get("device", "auto")))
    _planner = StreamVln(load_model(_MODEL, device), load_tokenizer(_MODEL), device)
    _log.warning("streamvln: loaded on %s", device)


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    assert _planner is not None
    _planner.reset(initial_state)


def step(features: dict) -> Step:
    assert _planner is not None
    return _planner.step(features)


if __name__ == "__main__":
    logging.basicConfig()
    manifest = load_manifest(_DIR / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset, on_init=on_init)
