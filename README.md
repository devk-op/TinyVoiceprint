# TinyVoiceprint

Real-time, text-independent speaker verification on an STM32F407 (Cortex-M4F, 168 MHz).
The board listens through its on-board PDM microphone, enrols any new voice on the device,
and, while someone is speaking, decides every 0.4 s whether it is the enrolled voice.

- 139k-parameter time-channel-separable network, 8-bit integer convolutions checked
  bit-exact against a Python reference by a self-test at every boot
- 201 ms per 1.5 s window (293 ms under the live audio load), about 30 KB of RAM for the
  network and 151 KB of weights in flash
- The board also works as a USB audio microphone (UAC2) for collecting data

## Layout

| Path | What it is |
|---|---|
| `modules/pdm` | PDM to 16 kHz PCM (CIC + FIR decimator) |
| `modules/features` | Log-mel features, matched to a Python reference |
| `modules/voiceprint` | The integer network; `voiceprint_model.*` are generated |
| `modules/voice_match` | Speech gate, smoothing, enrolment and the match decision |
| `modules/voice_store` | Enrolled voiceprint stored in flash, tagged by model |
| `modules/usb_audio` | USB microphone (TinyUSB) |
| `modules/app` | FreeRTOS application and LEDs |
| `targets/stm32f407g_disc1` | Board support, OpenOCD config, flash and debug scripts |
| `tools/train` | Training, quantization, integer reference, evaluation |
| `tools/host` | Host tools: on-chip performance readout, recording, replay |

## Build and flash

Requires Bazel and an Arm GNU toolchain.

```
bazel test //modules/...
bazel build --config=stm32f407g_disc1 //modules/app
targets/stm32f407g_disc1/py/flash.py bazel-bin/modules/app/app
uv run tools/host/read_perf.py          # self-test, timing and match state over SWD
```

To enrol, press the blue user button and talk for about 15 seconds. Blue LED: the enrolled
voice is speaking.

## Model

`uv run tools/train/tcs_integer.py <model> --write` regenerates the firmware's model files
from a trained model. Datasets (LibriSpeech, Common Voice, VoxCeleb, MUSAN, RIRs) and trained
weights are not included; the training tools expect them under `data/`.

## License

Apache-2.0. See `LICENSE`.
