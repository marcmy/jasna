# Experimental Kiyo-LTX-v1 compatibility

This development-only integration uses independent plaintext Kiyo-LTX-v1 weights published by [Kukkiri](https://github.com/treehog2/kukkiri), not the official encrypted Jasna weights. The official protection path is unchanged. The converter and runtime compatibility code are ported from [sh202603/jasna:modi](https://github.com/sh202603/jasna/tree/modi), under AGPL-3.0. See the [reference walkthrough](https://github.com/sh202603/jasna/blob/modi/docs/en/ltx_kukkiri.md) for source file details and checksums.

Download the transformer, VAE and prompt safetensors from [the original publisher](https://huggingface.co/treehog/kukkiri/tree/main/ltx-v1), under the LTX-2.x Community License. Do not redistribute the models with Jasna. Allow around 30 GB of disk space for input and converted files. From a source development installation, run in PowerShell:

```powershell
.venv\Scripts\python scripts\ltx_lora_training\convert_kukkiri_ltx.py D:\kukkiri-ltx-v1 model_weights\ltx-kukkiri
.venv\Scripts\python -m jasna --input in.mp4 --output out.mp4 --restoration-model-name ltx --restoration-model-path model_weights\ltx-kukkiri
```

Use a non-sensitive test clip and the full VAE defaults, INT8 only. This first port has not been tested on SM 7.5 or 8 GB VRAM; the convolutional decoder does not yet tile or retry after out-of-memory. Do not use the frozen distribution, NVFP4 or trial mode for this test. A dedicated model selector and automatic download are deferred until independent inference is verified. The generated output metadata identifies the independent model source.

Reference tests: `tests/test_ltx_position_fps_modi.py`, `tests/test_ltx_conv_vae_modi.py`, `tests/test_convert_kukkiri_ltx_modi.py`.
