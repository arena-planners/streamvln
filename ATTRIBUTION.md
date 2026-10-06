# Attribution

## Code

`model_streamvln.py` is assembled from StreamVLN, commit e48f6ff (2025-11-02),
https://github.com/InternRobotics/StreamVLN (formerly OpenRobotLab/StreamVLN), CC BY-NC-SA 4.0, and the LLaVA-NeXT
subset StreamVLN vendors under `llava/` (https://github.com/LLaVA-VL/LLaVA-NeXT, Apache-2.0). Each section names its
source file:

- `streamvln/model/stream_video_vln.py` (`StreamVLNModel`, `StreamVLNForCausalLM`): verbatim.
- `streamvln/utils/utils.py`: the token constants.
- `llava/utils.py`: `rank0_print`.
- `llava/model/multimodal_encoder/siglip_encoder.py`: verbatim except `SigLipVisionTower.load_model`, which builds
  `SigLipVisionModel` from the default `SigLipVisionConfig` (so400m-patch14-384) instead of
  `SigLipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")`. The v1_3 checkpoint holds all 421 vision
  tower tensors (the 26 encoder layers left after upstream drops the last one), so loading it leaves no key missing and
  no SigLIP download is needed.
- `llava/model/multimodal_encoder/builder.py` (`build_vision_tower`): the SigLIP branch only.
- `llava/model/multimodal_projector/builder.py`: without the `pooler` projector.
- `llava/model/multimodal_resampler/builder.py`: the identity resampler only, its `IdentityMap` renamed
  `ResamplerIdentityMap` to keep it apart from the projector's.
- `llava/model/llava_arch.py`: `LlavaMetaModel.__init__` and `get_vision_tower`, `LlavaMetaForCausalLM.get_model` and
  `get_vision_tower`, the only members on StreamVLN's inference path.
- `llava/model/language_model/llava_qwen.py`: `LlavaQwenConfig` and `LlavaQwenModel`.

`planner.py` ports the inference loop of `streamvln/streamvln_eval.py` (`VLNEvaluator.eval_action`, `preprocess_qwen`,
`parse_actions`) and the model setup of its `eval()`: the task prompt, the memory sentence, the system-free per-message
Qwen chat template with an empty assistant turn, the window of 32 steps, the 8 memory frames sampled with
`slice(0, start, start // 8)`, the KV cache and `output_ids` carried between calls, greedy decoding with
`max_new_tokens=10000`, and the action regex. Differences:

- The conjunction before each frame is always the first one, `you can see `, as in `streamvln/streamvln_agent.py`.
  The evaluation draws it with `random.choice`.
- A window that closes while a chunk is still executing reopens at the next call with the memory frames. Upstream
  attaches them only on calls that fall on a multiple of 32 steps, while its prompt carries `<memory>` on every
  reopening call, which would leave the memory token without features (inferred from the code, not run upstream).
- Attention is PyTorch SDPA instead of `flash_attention_2`, which has no wheels for Blackwell GPUs with torch 2.14.
- Depth, pose and intrinsics are not passed to `generate`, `encode_rgbd` reads only the images and `time_ids`.
- Frames are stored as bf16 pixel values, the dtype upstream casts them to before the call.

## Weights

- `mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3`, revision f1f76c6: `config.json`,
  `generation_config.json`, `model.safetensors.index.json` and the four safetensors shards.
- `mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world`, revision 7b0269a: `tokenizer.json`,
  `tokenizer_config.json`, `vocab.json`, `merges.txt`, `added_tokens.json`, `special_tokens_map.json`. The v1_3
  repository ships no tokenizer (StreamVLN issue #64). Both checkpoints use the Qwen2 vocabulary with `<image>` at
  151646, `<memory>` is added at load as 151647, as upstream does.

Neither repository states a license. The model is fine-tuned from LLaVA-Video-7B-Qwen2 on R2R, RxR, EnvDrop and
ScaleVLN data built on Matterport3D and HM3D scenes, whose terms restrict use to non-commercial research.

## Test data

`tests/data/teaser_*.png` are the four observation frames of StreamVLN's `assets/teaser.gif`, cropped to 4:3.

Cite:

```bibtex
@article{wei2025streamvln,
  title={StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling},
  author={Wei, Meng and Wan, Chenyang and Yu, Xiqian and Wang, Tai and Yang, Yuqiang and Mao, Xiaohan and Zhu, Chenming and Cai, Wenzhe and Wang, Hanqing and Chen, Yilun and others},
  journal={arXiv preprint arXiv:2507.05240},
  year={2025}
}
```

The Apache License 2.0 text covering the LLaVA-NeXT parts is in LICENSE.
