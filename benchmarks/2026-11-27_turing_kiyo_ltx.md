# Kiyo-LTX v1 on RTX 2070 SUPER (Windows, 2026-11-27)

Experimental source-build measurements on SM 7.5, 8 GB VRAM, Ryzen 7 5800X, PyTorch 2.12.1+cu130, CUDA toolkit 13.5, native INT8 transformer. RF-DETR v6; assets/test_clip1_1080p.mp4; 300 frames, 14 tracks, 14 windows at 512 pixels, 28 window-steps. All timings in seconds unless indicated. These are individual runs, not controlled repeated trials; available VRAM and compilation/cache state varied.

| Run | Scan | Encode | Denoise | Compose | Total | Resident | Whole-card peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial PyTorch fallback | 3.3 | 61.8 | 146.7 | 56.2 | 268.0 | 5/48 | 7877 |
| Native, early run | 3.3 | 66.2 | 165.8 | 58.0 | 293.4 | 5/48 | 7835 |
| Detector released | 3.2 | 77.9 | 138.8 | 57.1 | 277.1 | 5/48 | 7450 |
| 2816 MiB reserve | 3.3 | 65.9 | 141.0 | 56.8 | 267.1 | 6/48 | 7481 |
| Independent pairs | 3.1 | 65.9 | 138.5 | 56.6 | 264.1 | 6/48 | 7444 |
| Compiled decoder, cold | 3.1 | 63.7 | 140.4 | 110.1 | 317.3 | 6/48 | 7310 |
| Compiled decoder, cached | 3.3 | 66.8 | 145.2 | 47.2 | 262.6 | 5/48 | 7792 |

## Findings

- Detector release reclaimed 646 MiB after scanning. Whole-video runs release it; segment rendering and synthetic speed-test scans do not.
- Reserve tuning and pair batching had small, inconclusive throughput effects. All 300 decoded RGB frames matched exactly across detector-release, reduced-reserve and paired-track outputs.
- Encoder: compiled FP16 4.400 s/window vs eager channels-last 5.288 and cuDNN autotuned eager 5.223. Comfy Kitchen full-window convolution failed with an unsupported CUTLASS MMA operation; promoted Triton convolution took 35.818 vs 4.297 seconds/window. Keep existing encoder.
- Conv VAE decoder: eager 3.345 vs compiled 2.471 s/window (26.1% less time), peak allocated VRAM 3.62 vs 3.15 GiB. First compile took 70.525 s. Raw decoded output PSNR 67.78 dB; end-to-end compressed output PSNR 42.43 dB (worst frame 40.86 dB). Not bit-identical; visual comparison and broader validation remain outstanding. Cached compose was 16.6% faster, but overall improvement was only 1.5 s versus paired tracks.
- Native transformer profiler: 10/48 resident blocks for both isolated runs; aggregate GPU durations 4650.1 vs 4642.3 ms after tuning. Rotation/quantization aggregate 254.48 vs 240.98 ms. Wide unmodulated quantization at 8192 x 16384: 3.075 vs 2.118 ms with 16 rather than 8 warps, exact codes/scales. End-to-end speedup is not established. Leave modulation, epilogue and other architectures unchanged.

## Opt-in flags (PowerShell)

```powershell
$env:JASNA_LTX_ACTIVATION_RESERVE_MIB = '2816' # Default 3072; allowed 2560-3072
$env:JASNA_LTX_BATCH_INDEPENDENT = '1' # Pair same-shape single-window tracks, local transformer only
$env:JASNA_LTX_COMPILE_CONV_DECODER = '1' # Convolutional decoder only, requires compilation support
```

These are experimental switches, not a recommended universal configuration. All remain off by default (reserve defaults to 3072 MiB). The 16-warp wide quantizer is selected only on SM 7.5. Compiled decoder implementation subsequently adjusted to use an unbound function to avoid retaining the decoder through a reference cycle; benchmark timings above precede that cleanup.

Native Windows extension compilation includes /Zc:preprocessor for CUDA 13.5. Local weights, videos, logs and temporary scripts are excluded from source control.


## Adaptive temporal windows and mixed-precision VAE tail

The public 300-frame clip has 14 single-window tracks with 256 real track frames
and 1694 padded model frames. Adaptive windows (minimum 33, lengths 1+8k)
reduce model frames to 486; 13 windows use 33 frames and one uses 57.
Multi-window tracks keep their original 121-frame context. The flag is limited
to local whole-video flat restoration. The output differs from the original
full-context model; this is a quality/performance tradeoff.

| Run | Scan | Encode | Denoise | Compose | Total | Resident |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Integrated adaptive, cold | 3.4 | 159.6 | 50.4 | 18.0 | 231.4 | |
| Integrated adaptive, warm | 3.1 | 21.5 | 55.8 | 20.7 | 101.2 | 3/48 |
| Mixed encoder, cold | 3.1 | 131.4 | 49.8 | 17.3 | 201.5 | 6/48 |
| Mixed encoder, warm | 3.1 | 20.1 | 48.2 | 17.4 | 88.8 | 7/48 |
| Mixed encoder + compiled decoder, cold decoder | 3.2 | 20.4 | 49.3 | 153.0 | 226.0 | 7/48 |
| Mixed encoder + compiled decoder, warm | 3.2 | 20.1 | 52.5 | 18.0 | 93.9 | 6/48 |

Mixed tail keeps residuals, normalization and statistics in FP32; scaled FP16
convolutions use a conservative weight L1 bound, a normalized input bound or
dynamic input magnitude, and power-of-two divisors. Only the local conv VAE
with its original encoder on SM 7.5 opts in. An explicit module replacement
and distinct compiled graph avoid misleading results from monkeypatching a
previously compiled graph. Non-finite output raises an error.

Standalone public clip results (median of three cached calls):

| Window | Existing compiled | Mixed compiled | Peak VRAM old/new | Latent PSNR vs compiled |
| --- | ---: | ---: | --- | ---: |
| 121 frames | 4.0912 s | 2.3985 s | 3.421 / 3.196 GiB | 79.52 dB |
| Independent 33 frames | 0.9912 s | 0.6408 s | 1.700 / 1.475 GiB | 79.28 dB |

Cold and warm integrated mixed encoder outputs are pixel-identical after
decoding all 300 frames. The mixed encoder output vs original adaptive output
has 42.02 dB PSNR (worst frame 40.35 dB) after lossy output encoding.
This is not a perceptual equivalence guarantee. Warm decode rates with
compilation enabled: 1.02 -> 0.77 s/window; overall short-clip time did not improve.
All flags remain opt-in.

Additional experiments not integrated:

- Fused SM75 CUTLASS INT8 GEMM + epilogue: exact in 35 shape/tile combinations;
  original 128x128 / 64x64 tile generally best, modest gains for wide matrices
  and a regression with fused GELU. Avoid another dependency for now.
- Combined QKV GEMM: exact but roughly 1% faster at 2560-8192 rows with
  80-256 MiB greater transient peak; contiguous output copies made it slower.
- Per-layer temporal convolution chunking: exact against eager inference;
  no encoder speed benefit. Decoder chunk16 cut peak 3.618 -> 3.270 GiB
  but increased time 3.796 -> 3.976 s.

Additional PowerShell switches:

```powershell
$env:JASNA_LTX_ADAPTIVE_WINDOWS = '1'
$env:JASNA_LTX_MIXED_ENCODER_TAIL = '1'
```

Regression suite after mixed encoder integration: 351 passed.
