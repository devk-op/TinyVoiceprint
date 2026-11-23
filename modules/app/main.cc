// On-device voice match, with the board also acting as a USB microphone.
//
// Captures the onboard MP45DT02 PDM mic, decimates it to 16 kHz 16-bit PCM,
// and computes log-mel features. A small 8-bit network turns the newest
// 1.5 s of features into a voiceprint every 0.4 s and compares it with
// an enrolled one, all on this chip.
//
// To enroll: press the blue user button and speak for about 10 seconds
// (orange stays on while it listens). The enrolled voiceprint is saved in
// the last flash sector (modules/voice_store), so it survives reset,
// power-off and flashing new firmware, until the next enrollment or a
// clear. A voiceprint from a different model is ignored at boot.
//
// The PCM is also streamed to the host as a USB Audio Class 2 microphone
// ("device_ml mic") over the micro-USB connector CN5.
//
// There's no log path off the board (the ST-LINK on this board doesn't
// bridge USART2), so status is shown on the user LEDs, and in the structs
// below, which tools/host/read_perf.py reads over SWD:
//   blue   (PD15)  the enrolled voice is speaking
//   orange (PD13)  blinks at 1 Hz while running; solid while enrolling
//   green  (PD12)  USB enumerated by the host
//   red    (PD14)  error: mic failed to start, capture stalled, or blocks
//                  were dropped (stays on once set)

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <new>

#include "FreeRTOS.h"
#include "modules/features/log_mel.h"
#include "modules/features/log_mel_selftest.h"
#include "modules/mic/mp45dt02.h"
#include "modules/pdm/pdm_decimator.h"
#include "modules/usb_audio/usb_audio.h"
#include "modules/voice_match/voice_match.h"
#include "modules/voice_store/stm32_flash.h"
#include "modules/voice_store/voice_store.h"
#include "modules/voiceprint/voiceprint.h"
#include "modules/voiceprint/voiceprint_selftest.h"
#include "pw_log/log.h"
#include "stm32f4xx.h"
#include "task.h"

// Timing and self-test results, read from the host over SWD with
// tools/host/read_perf.py since there's no log path off the board. Plain
// 32-bit fields in a fixed order so the reader can decode a raw memory dump;
// keep the two in sync. Cycle counts are CPU cycles from the DWT counter.
extern "C" {
struct DeviceMlPerf {
  uint32_t selftest_frames;  // Should be 18 (golden::kNumFrames).
  float selftest_max_error;  // On-chip log-mel vs the Python reference.
  uint32_t selftest_cycles;  // For all selftest_frames frames.
  uint32_t cpu_hz;
  uint32_t blocks;  // 2 ms audio blocks processed.
  uint32_t frames;  // Log-mel frames produced.
  uint32_t decimate_min;  // Per 2 ms block.
  uint32_t decimate_max;
  uint32_t frame_min;  // Per LogMel::Process call that completed a frame.
  uint32_t frame_max;
  uint32_t decimate_sum_low;  // 64-bit totals, low word first.
  uint32_t decimate_sum_high;
  uint32_t features_sum_low;
  uint32_t features_sum_high;
  // Voiceprint network vs the integer reference (tools/train/export_c.py).
  uint32_t voiceprint_input_ok;  // 1 if the int8 input matched exactly.
  uint32_t voiceprint_conv_ok;  // 1 if the convolution output matched.
  float voiceprint_max_error;  // Largest voiceprint difference, per value.
  float voiceprint_similarity;  // vs the reference voiceprint; 1 = same.
  uint32_t voiceprint_cycles;  // One 1.5 s window, idle system.
};
DeviceMlPerf device_ml_perf = {};

// Live voice-match state. `command` and `threshold` may be written from the
// host over SWD (tools/host/read_perf.py --enroll / --clear / --threshold).
struct DeviceMlVoice {
  uint32_t command;  // 1: start enrollment, 2: clear it, 3: chip check.
  float threshold;  // Match threshold on the score.
  uint32_t result;  // Latest VoiceMatch::Result.
  float score;  // Latest smoothed score against the enrolled voice.
  uint32_t enrolled;
  uint32_t enroll_windows;  // Speech windows gathered in this enrollment.
  uint32_t steps;  // Decisions made.
  uint32_t speech_steps;  // Of those, how many ran the network.
  uint32_t match_steps;  // Of those, how many matched.
  uint32_t step_cycles_last;  // Wall-clock cycles of the last network run.
  uint32_t step_cycles_max;
  uint32_t enroll_failures;
  // Energy measurement (see tools/host/read_perf.py --mode). 0: normal.
  // 1: listen only (mic, filters, features; the network never runs).
  // 2: network back to back on the newest window, speech or not.
  uint32_t mode;
  uint32_t network_runs;  // Completed network runs, all modes.
  // Flash copy of the enrollment: 0 none, 1 loaded at boot, 2 saved after
  // enrolling, 3 cleared, 4 a flash write failed.
  uint32_t stored;
  uint32_t store_free_slots;  // Saves left before the sector is compacted.
};
volatile DeviceMlVoice device_ml_voice = {};

// Chip-vs-reference check (tools/host/chip_check.py). The host writes
// kCheckSamples int16 PCM samples to the start of the otherwise unused CCM
// RAM and sets device_ml_voice.command to 3; the match task computes log-mel
// features and runs the network on them exactly as on live audio, then
// publishes hashes of the int8 input and the convolution output (same FNV-1a
// as the boot self-test) and the voiceprint, and increments `done`.
struct DeviceMlCheck {
  uint32_t done;
  uint32_t frames;  // Feature frames produced; should be kFrames.
  uint32_t input_hash;
  uint32_t conv_hash;
  uint32_t conv_bytes;
  uint32_t cycles;  // Features and network.
  uint32_t input_address;  // For dumping the int8 input on a mismatch.
  uint32_t conv_address;
  float embedding[device_ml::voiceprint::Voiceprint::kEmbeddingDim];
};
volatile DeviceMlCheck device_ml_check = {};
}

namespace {

using device_ml::features::LogMel;
using device_ml::mic::Mp45dt02;
using device_ml::pdm::PdmDecimator;
using device_ml::voice_match::VoiceMatch;

constexpr uint16_t kLedGreen = GPIO_PIN_12;
constexpr uint16_t kLedOrange = GPIO_PIN_13;
constexpr uint16_t kLedRed = GPIO_PIN_14;
constexpr uint16_t kLedBlue = GPIO_PIN_15;

constexpr size_t kSamplesPerBlock =
    Mp45dt02::kBlockWords / PdmDecimator::kWordsPerSample;
constexpr uint32_t kBlocksPerSecond = 500;  // 2 ms blocks.

// Highest priority: each 2 ms block must be decimated before DMA wraps
// around and overwrites it.
constexpr UBaseType_t kAudioTaskPriority = configMAX_PRIORITIES - 1;
constexpr uint32_t kAudioStackWords = 1024;
// Lowest application priority: the network runs in whatever time the audio
// path leaves.
constexpr UBaseType_t kMatchTaskPriority = 1;
constexpr uint32_t kMatchStackWords = 1536;
// A decision every 40 frames (0.4 s); the network takes about 0.29 s of
// that alongside the audio path.
constexpr uint32_t kFramesPerDecision = 40;
constexpr uint16_t kUserButton = GPIO_PIN_0;  // PA0, high when pressed.
// High while the network runs, for a current profiler's logic input. PD11
// is unused on this board (header P2).
constexpr uint16_t kNetworkMarker = GPIO_PIN_11;

StackType_t audio_stack[kAudioStackWords];
StaticTask_t audio_task_buffer;
StackType_t match_stack[kMatchStackWords];
StaticTask_t match_task_buffer;
TaskHandle_t match_task = nullptr;
// Latest decision, passed to the host with the audio (see usb_audio::Write).
std::atomic<bool> voice_matched{false};

Mp45dt02 mic;
PdmDecimator decimator;
LogMel log_mel;
VoiceMatch voice_match;

void EnableCycleCounter() {
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

uint32_t Cycles() { return DWT->CYCCNT; }

void AddToSum(uint32_t& low, uint32_t& high, uint32_t value) {
  const uint32_t before = low;
  low += value;
  if (low < before) {
    ++high;
  }
}

// Confirms the features computed on this chip match the Python reference
// the model is trained against, and times them on an idle system.
void RunFeatureSelfTest() {
  const uint32_t start = Cycles();
  const device_ml::features::SelfTestResult result =
      device_ml::features::RunLogMelSelfTest(log_mel);
  device_ml_perf.selftest_cycles = Cycles() - start;
  device_ml_perf.selftest_frames = static_cast<uint32_t>(result.frames);
  device_ml_perf.selftest_max_error = result.max_abs_error;
}

// Confirms the network on this chip computes what the Python integer
// reference does, and times one run on an idle system.
void RunVoiceprintSelfTest() {
  const uint32_t start = Cycles();
  const device_ml::voiceprint::SelfTestResult result =
      device_ml::voiceprint::RunVoiceprintSelfTest(voice_match.voiceprint());
  device_ml_perf.voiceprint_cycles = Cycles() - start;
  device_ml_perf.voiceprint_input_ok = result.input_matches ? 1 : 0;
  device_ml_perf.voiceprint_conv_ok = result.conv_matches ? 1 : 0;
  device_ml_perf.voiceprint_max_error = result.max_abs_error;
  device_ml_perf.voiceprint_similarity = result.similarity;
}

void InitLeds() {
  __HAL_RCC_GPIOD_CLK_ENABLE();
  GPIO_InitTypeDef gpio = {};
  gpio.Pin = kLedGreen | kLedOrange | kLedRed | kLedBlue | kNetworkMarker;
  gpio.Mode = GPIO_MODE_OUTPUT_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOD, &gpio);

  // User button: the board pulls it low and the button drives it high.
  __HAL_RCC_GPIOA_CLK_ENABLE();
  gpio.Pin = kUserButton;
  gpio.Mode = GPIO_MODE_INPUT;
  HAL_GPIO_Init(GPIOA, &gpio);
}

device_ml::voice_store::VoiceStore& Store() {
  static device_ml::voice_store::Stm32Sector11 flash;
  static device_ml::voice_store::VoiceStore store(flash, device_ml::voice_store::CurrentModelTag());
  return store;
}

// Before audio starts: compacting the sector, when needed, stalls the CPU.
void RestoreEnrollment() {
  Store().Prepare();
  static float voice[VoiceMatch::kDim];  // Not on the audio task's small stack.
  if (Store().Load(voice)) {
    voice_match.SetEnrollment(voice);
    device_ml_voice.stored = 1;
    device_ml_voice.enrolled = 1;
  }
  device_ml_voice.store_free_slots = Store().free_slots();
}

void SetLed(uint16_t led, bool on) {
  HAL_GPIO_WritePin(GPIOD, led, on ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

// Makes a decision each time the audio task signals that another
// kFramesPerDecision frames have arrived.
// Runs the network on the PCM window the host left in CCM RAM. Only the match
// task touches the voiceprint network, so this is safe between decisions.
void RunChipCheck() {
  using device_ml::voiceprint::Voiceprint;
  constexpr uintptr_t kCcm = 0x10000000;  // 64 KB, core-only, unused otherwise.
  constexpr size_t kCheckSamples =
      LogMel::kFrameLength + (Voiceprint::kFrames - 1) * LogMel::kHop;
  constexpr size_t kPcmBytes = kCheckSamples * sizeof(int16_t);
  static_assert(kPcmBytes % 8 == 0);
  static_assert(kPcmBytes + sizeof(LogMel) <= 64 * 1024, "CCM is 64 KB");
  static LogMel* mel = new (reinterpret_cast<void*>(kCcm + kPcmBytes)) LogMel();
  mel->Reset();
  const auto* pcm = reinterpret_cast<const int16_t*>(kCcm);
  // Frames overwrite PCM that has already been consumed: frame n is written
  // at byte 160n once samples up to 400 + 160n have been read.
  auto* frames = reinterpret_cast<float*>(kCcm);
  size_t produced = 0;
  const uint32_t start = Cycles();
  mel->Process(pcm, kCheckSamples, [&](const LogMel::Frame& frame) {
    if (produced < Voiceprint::kFrames) {
      std::copy(frame.begin(), frame.end(), frames + produced * Voiceprint::kMels);
    }
    ++produced;
  });
  Voiceprint& voiceprint = voice_match.voiceprint();
  float embedding[Voiceprint::kEmbeddingDim];
  voiceprint.QuantizeInput(frames);
  device_ml_check.input_hash = device_ml::voice_store::Fnv1a(
      voiceprint.input(), Voiceprint::kFrames * Voiceprint::kMels);
  // Keep a copy of the int8 input in free CCM (after the 24 KB of frames), so
  // the host can compare it after the live pipeline has reused the buffer.
  constexpr uintptr_t kInputCopy = kCcm + 24 * 1024;
  static_assert(24 * 1024 + Voiceprint::kFrames * Voiceprint::kMels <= kPcmBytes);
  std::copy_n(voiceprint.input(), Voiceprint::kFrames * Voiceprint::kMels,
              reinterpret_cast<int8_t*>(kInputCopy));
  voiceprint.RunConvolutions();
  device_ml_check.conv_hash = device_ml::voice_store::Fnv1a(
      voiceprint.conv_output(), voiceprint.conv_output_bytes());
  voiceprint.PoolAndEmbed(embedding);
  device_ml_check.cycles = Cycles() - start;
  device_ml_check.frames = static_cast<uint32_t>(produced);
  device_ml_check.conv_bytes = static_cast<uint32_t>(voiceprint.conv_output_bytes());
  device_ml_check.input_address = kInputCopy;
  device_ml_check.conv_address = reinterpret_cast<uintptr_t>(voiceprint.conv_output());
  for (size_t i = 0; i < Voiceprint::kEmbeddingDim; ++i) {
    device_ml_check.embedding[i] = embedding[i];
  }
  device_ml_check.done = device_ml_check.done + 1;
}

void MatchTask(void*) {
  using Result = VoiceMatch::Result;
  for (;;) {
    if (device_ml_voice.mode == 2) {
      // Back to back: run the network as often as the CPU allows.
      HAL_GPIO_WritePin(GPIOD, kNetworkMarker, GPIO_PIN_SET);
      const uint32_t start = Cycles();
      voice_match.RunNetworkOnNewestWindow();
      device_ml_voice.step_cycles_last = Cycles() - start;
      HAL_GPIO_WritePin(GPIOD, kNetworkMarker, GPIO_PIN_RESET);
      device_ml_voice.network_runs = device_ml_voice.network_runs + 1;
      SetLed(kLedBlue, false);
      continue;
    }
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    if (device_ml_voice.mode == 1) {
      SetLed(kLedBlue, false);
      voice_matched = false;
      continue;  // Listen only.
    }

    const uint32_t command = device_ml_voice.command;
    if (command != 0) {
      device_ml_voice.command = 0;
      if (command == 1) {
        voice_match.StartEnrollment();
      } else if (command == 3) {
        RunChipCheck();
      } else if (command == 2) {
        voice_match.ClearEnrollment();
        device_ml_voice.stored = Store().Clear() ? 3 : 4;
        device_ml_voice.store_free_slots = Store().free_slots();
      }
    }
    voice_match.set_threshold(device_ml_voice.threshold);

    HAL_GPIO_WritePin(GPIOD, kNetworkMarker, GPIO_PIN_SET);
    const uint32_t start = Cycles();
    const Result result = voice_match.Step();
    const uint32_t cycles = Cycles() - start;
    HAL_GPIO_WritePin(GPIOD, kNetworkMarker, GPIO_PIN_RESET);

    device_ml_voice.result = static_cast<uint32_t>(result);
    device_ml_voice.score = voice_match.score();
    device_ml_voice.enrolled = voice_match.enrolled() ? 1 : 0;
    device_ml_voice.enroll_windows = voice_match.enroll_windows();
    device_ml_voice.steps = device_ml_voice.steps + 1;
    if (result == Result::kSpeech || result == Result::kOther ||
        result == Result::kMatch) {
      device_ml_voice.speech_steps = device_ml_voice.speech_steps + 1;
      device_ml_voice.network_runs = device_ml_voice.network_runs + 1;
      device_ml_voice.step_cycles_last = cycles;
      if (cycles > device_ml_voice.step_cycles_max) {
        device_ml_voice.step_cycles_max = cycles;
      }
    }
    if (result == Result::kMatch) {
      device_ml_voice.match_steps = device_ml_voice.match_steps + 1;
    }
    if (result == Result::kEnrolled) {
      // About 3 ms of flash programming; it stalls code fetches, but the
      // audio DMA keeps running.
      device_ml_voice.stored = Store().Save(voice_match.enrolled_voice()) ? 2 : 4;
      device_ml_voice.store_free_slots = Store().free_slots();
    }
    if (result == Result::kEnrollFailed) {
      device_ml_voice.enroll_failures = device_ml_voice.enroll_failures + 1;
    }
    voice_matched = result == Result::kMatch;
    SetLed(kLedBlue, result == Result::kMatch);
  }
}

void AudioTask(void*) {
  EnableCycleCounter();
  device_ml_perf.cpu_hz = SystemCoreClock;
  device_ml_perf.decimate_min = UINT32_MAX;
  device_ml_perf.frame_min = UINT32_MAX;
  RunFeatureSelfTest();
  RunVoiceprintSelfTest();
  RestoreEnrollment();

  device_ml::usb_audio::Start();

  if (pw::Status status = mic.Start(); !status.ok()) {
    PW_LOG_ERROR("mic: start failed: %s", status.str());
    SetLed(kLedRed, true);
    vTaskDelete(nullptr);
  }
  PW_LOG_INFO("mic: capturing at 16 kHz");

  std::array<int16_t, kSamplesPerBlock> pcm;
  uint32_t block_count = 0;
  uint32_t frames_since_decision = 0;
  bool button_was_down = false;
  for (;;) {
    const uint16_t* pdm = mic.WaitForBlock();
    if (pdm == nullptr) {
      SetLed(kLedRed, true);
      continue;
    }
    const uint32_t decimate_start = Cycles();
    decimator.Process(pdm, Mp45dt02::kBlockWords, pcm.data());
    const uint32_t features_start = Cycles();
    uint32_t new_frames = 0;
    log_mel.Process(pcm.data(), pcm.size(), [&new_frames](const LogMel::Frame& frame) {
      voice_match.AddFrame(frame.data());
      ++new_frames;
    });
    const uint32_t features_end = Cycles();

    DeviceMlPerf& perf = device_ml_perf;
    const uint32_t decimate_cycles = features_start - decimate_start;
    const uint32_t features_cycles = features_end - features_start;
    perf.decimate_min = std::min(perf.decimate_min, decimate_cycles);
    perf.decimate_max = std::max(perf.decimate_max, decimate_cycles);
    AddToSum(perf.decimate_sum_low, perf.decimate_sum_high, decimate_cycles);
    AddToSum(perf.features_sum_low, perf.features_sum_high, features_cycles);
    if (new_frames != 0) {
      perf.frame_min = std::min(perf.frame_min, features_cycles);
      perf.frame_max = std::max(perf.frame_max, features_cycles);
    }
    perf.frames += new_frames;
    ++perf.blocks;

    device_ml::usb_audio::Write(pcm.data(), pcm.size(), voice_matched);

    frames_since_decision += new_frames;
    if (frames_since_decision >= kFramesPerDecision) {
      frames_since_decision = 0;
      xTaskNotifyGive(match_task);
    }

    // Sampled every 2 ms; a press starts enrollment.
    const bool button_down =
        HAL_GPIO_ReadPin(GPIOA, kUserButton) == GPIO_PIN_SET;
    if (button_down && !button_was_down) {
      device_ml_voice.command = 1;
    }
    button_was_down = button_down;

    if (++block_count % (kBlocksPerSecond / 2) == 0) {
      if (voice_match.enrolling()) {
        SetLed(kLedOrange, true);
      } else {
        HAL_GPIO_TogglePin(GPIOD, kLedOrange);
      }
      SetLed(kLedGreen, device_ml::usb_audio::mounted());
      if (mic.dropped_blocks() != 0) {
        SetLed(kLedRed, true);
      }
    }
  }
}

}  // namespace

// Called from the target's boot code before the scheduler starts.
extern "C" void device_ml_AppInit() {
  InitLeds();
  device_ml_voice.threshold = VoiceMatch::kDefaultThreshold;
  match_task = xTaskCreateStatic(MatchTask,
                                 "match",
                                 kMatchStackWords,
                                 nullptr,
                                 kMatchTaskPriority,
                                 match_stack,
                                 &match_task_buffer);
  xTaskCreateStatic(AudioTask,
                    "audio",
                    kAudioStackWords,
                    nullptr,
                    kAudioTaskPriority,
                    audio_stack,
                    &audio_task_buffer);
}
