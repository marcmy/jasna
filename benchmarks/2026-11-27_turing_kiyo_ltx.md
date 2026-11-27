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
