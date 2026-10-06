# streamvln

[StreamVLN](https://streamvln.github.io/) (Wei et al., 2025, "StreamVLN: Streaming Vision-and-Language Navigation via
SlowFast Context Modeling", arXiv 2507.05240) behind the arena_planners bridge. A LLaVA-Video model on Qwen2-7B follows
a natural-language instruction from the egocentric RGB stream alone, without a goal pose, and answers each call with a
chunk of discrete actions (normally four): `TURN LEFT`, `TURN RIGHT`, `MOVE FORWARD` or `STOP`. The checkpoint is
`StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3`, reported at SR 56.4 and SPL 50.2 on R2R-CE val-unseen.

```sh
arena feature planners add streamvln
arena launch sim:=gazebo robot:=jackal[camera=vln_streamvln] robot.planner:=streamvln robot.mobile.goal:=instruction
```

`robot.planner:=streamvln` selects `robot.mobile:=vla`. The planner accepts the instruction only (`goal_inputs:
[instruction]`), so `robot.mobile.goal:=instruction` is also the default. It needs a CUDA GPU with about 18 GB free
(16 GB of bf16 weights plus the KV cache and frame features).

## Inputs

| Datasource | Topic (jackal `camera=vln_streamvln`) | Model input |
|---|---|---|
| `rgb` (`image` sensor) | `vln_camera/image` | 640x480 RGB, squashed to 384x384 for SigLIP, 196 tokens per frame |
| instruction | from the goal | inserted verbatim into the task prompt |

StreamVLN trained in Habitat with a 640x480 RGB camera at 79 degrees HFOV, 1.25 m above the floor, which the
`vln_streamvln` camera reproduces. Depth is not used.

## Primitives

The planner returns discrete commands (`action_type: discrete`) and the bridge executes each as a motion primitive:

| Model token | Command | Primitive |
|---|---|---|
| up arrow U+2191 (`MOVE FORWARD`) | `forward` | 0.25 m straight ahead |
| left arrow U+2190 (`TURN LEFT`) | `left` | 15 degrees counterclockwise |
| right arrow U+2192 (`TURN RIGHT`) | `right` | 15 degrees clockwise |
| `STOP` | none | ends the episode for the planner |

A call runs only when the previous chunk is used up: the planner executes the chunk one primitive per observation and
records each step's frame, as upstream's evaluation loop does. An answer without any action token counts as `STOP`.

On `STOP` the planner latches: it stops calling the model and, for a goto phase with `signal: arrived`, keeps sending
`arrived` until the next reset. Without an expected signal it stands still. Before an instruction arrives (a reset
without a goal) or with an empty instruction, the planner stands still without running the model.

## Streaming memory

Within a window of 32 executed steps the dialogue is multi-turn: each call appends the new frame as a user turn and the
KV cache carries the earlier turns. Every 32 steps the cache and dialogue are dropped, and the next call reopens the
window with the task prompt plus 8 frames sampled uniformly from all earlier steps of the episode as memory. Decoding
is greedy with the checkpoint's `repetition_penalty` of 1.05.

Instructions come from the scenario, per goto phase:

```yaml
robots:
  - start: [1.0, 1.0, 0.0]
    phases:
      - goto: [8.0, 8.0, 0.0]
        instruction: walk past the sofa and stop at the door to the kitchen
        signal: arrived
```

The goal pose decides success but never reaches the planner.

## Config

Set in `planner.yaml > config`, override per launch with `robot.mobile.config.<key>:=<value>`.

| Key | Default | Effect |
|---|---|---|
| `device` | `auto` | Torch device, `auto` picks CUDA when available. |

## Performance

On an RTX PRO 6000 Blackwell (torch 2.14 cu130, SDPA attention): about 0.11 s per call inside a window, 0.25 s for a
window-opening call with 8 memory frames, 0.8 s for the first call after loading, 17.3 GB peak GPU memory over a
40-step episode.

## Weights and license

`arena feature planners add streamvln` fetches the v1_3 checkpoint (4 safetensors shards, 16.06 GB, with its config and
generation config) and, because that repository has no tokenizer, the tokenizer files of the same authors'
`StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world` checkpoint. The SigLIP vision tower comes from the
checkpoint itself. The code is CC BY-NC-SA 4.0 like upstream, with Apache-2.0 parts from LLaVA-NeXT, see LICENSE and
ATTRIBUTION.md. The checkpoints carry no license of their own and are trained on Matterport3D-based data, so treat them
as non-commercial and research-only. Cite Wei et al. 2025 when using this planner:

```bibtex
@article{wei2025streamvln,
  title={StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling},
  author={Wei, Meng and Wan, Chenyang and Yu, Xiqian and Wang, Tai and Yang, Yuqiang and Mao, Xiaohan and Zhu, Chenming and Cai, Wenzhe and Wang, Hanqing and Chen, Yilun and others},
  journal={arXiv preprint arXiv:2507.05240},
  year={2025}
}
```
