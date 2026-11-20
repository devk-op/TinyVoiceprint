#include "modules/usb_audio/usb_audio.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>

#include "FreeRTOS.h"
#include "modules/usb_audio/usb_audio_ids.h"
#include "stm32f4xx.h"
#include "task.h"

// TinyUSB's headers mix enum and integer operands in a conditional, which
// -Wextra rejects in C++. Silence that for TinyUSB only.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#include "tusb.h"
#pragma GCC diagnostic pop

namespace device_ml::usb_audio {
namespace {

// At or above configLIBRARY_MAX_SYSCALL_INTERRUPT_PRIORITY (5): TinyUSB's
// ISR posts to a FreeRTOS queue.
constexpr uint32_t kUsbIrqPriority = 6;

constexpr uint32_t kUsbTaskStackWords = 512;
// Below the audio task: control requests can wait a few ms, audio can't.
constexpr UBaseType_t kUsbTaskPriority = configMAX_PRIORITIES - 2;

// Host-adjustable input volume, in the UAC2 unit of 1/256 dB.
constexpr int16_t kVolumeStep = 256;  // 1 dB
constexpr int16_t kVolumeMin = -40 * kVolumeStep;
constexpr int16_t kVolumeMax = 20 * kVolumeStep;

StackType_t usb_task_stack[kUsbTaskStackWords];
StaticTask_t usb_task_buffer;

std::atomic<bool> is_streaming{false};

// Feature unit state, indexed by channel (0 = master, 1 = the mono
// channel). Only touched by the TinyUSB task; Write() reads the derived
// atomics below.
std::array<uint8_t, 2> mute = {0, 0};
std::array<int16_t, 2> volume = {0, 0};
std::atomic<bool> muted{false};
std::atomic<float> linear_gain{1.0f};

void UpdateGain() {
  muted = mute[0] || mute[1];
  const float db = static_cast<float>(volume[0] + volume[1]) / kVolumeStep;
  linear_gain = std::pow(10.0f, db / 20.0f);
}

void ConfigureHardware() {
  __HAL_RCC_GPIOA_CLK_ENABLE();

  GPIO_InitTypeDef gpio = {};
  gpio.Pin = GPIO_PIN_11 | GPIO_PIN_12;  // D-, D+
  gpio.Mode = GPIO_MODE_AF_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  gpio.Alternate = GPIO_AF10_OTG_FS;
  HAL_GPIO_Init(GPIOA, &gpio);

  gpio = {};
  gpio.Pin = GPIO_PIN_9;  // VBUS sense
  gpio.Mode = GPIO_MODE_INPUT;
  gpio.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &gpio);

  // The 48 MHz USB clock comes from PLLQ (336 MHz VCO / 7), set up in boot.
  __HAL_RCC_USB_OTG_FS_CLK_ENABLE();

  // CN5's VBUS is routed to PA9, so use it to detect the cable (B-device
  // session valid), as TinyUSB's own stm32f407disco board support does.
  USB_OTG_FS->GCCFG &= ~USB_OTG_GCCFG_NOVBUSSENS;
  USB_OTG_FS->GCCFG |= USB_OTG_GCCFG_VBUSBSEN;

  // TinyUSB enables the IRQ itself; it just needs a FreeRTOS-safe priority.
  HAL_NVIC_SetPriority(OTG_FS_IRQn, kUsbIrqPriority, 0);
}

void UsbTask(void*) {
  // Must run after the scheduler starts: the USB ISR uses RTOS queues.
  const tusb_rhport_init_t init = {
      .role = TUSB_ROLE_DEVICE,
      .speed = TUSB_SPEED_AUTO,
  };
  tusb_init(BOARD_TUD_RHPORT, &init);
  for (;;) {
    tud_task();  // Blocks until there are USB events to handle.
  }
}

}  // namespace

void Start() {
  ConfigureHardware();
  xTaskCreateStatic(UsbTask,
                    "usb",
                    kUsbTaskStackWords,
                    nullptr,
                    kUsbTaskPriority,
                    usb_task_stack,
                    &usb_task_buffer);
}

bool mounted() { return tud_mounted(); }

bool streaming() { return is_streaming; }

size_t Write(const int16_t* samples, size_t count, bool mark) {
  if (!is_streaming) {
    return 0;
  }
  const bool silence = muted;
  const float gain = linear_gain;

  std::array<int16_t, 64> scaled;
  size_t written = 0;
  while (written < count) {
    const size_t chunk = std::min(scaled.size(), count - written);
    for (size_t i = 0; i < chunk; ++i) {
      const float value = silence ? 0.0f : samples[written + i] * gain;
      const int16_t sample =
          static_cast<int16_t>(std::clamp(value, -32768.0f, 32767.0f));
      scaled[i] = static_cast<int16_t>((sample & ~1) | (mark ? 1 : 0));
    }
    const uint16_t bytes = static_cast<uint16_t>(chunk * sizeof(int16_t));
    const uint16_t queued = tud_audio_write(scaled.data(), bytes);
    written += queued / sizeof(int16_t);
    if (queued < bytes) {
      break;  // Queue full; drop the rest of this block.
    }
  }
  return written;
}

}  // namespace device_ml::usb_audio

// TinyUSB interrupt and class callbacks. These run in TinyUSB's task, except
// the IRQ handler.
namespace usb = device_ml::usb_audio;

extern "C" void OTG_FS_IRQHandler(void) {
  tusb_int_handler(BOARD_TUD_RHPORT, true);
}

extern "C" bool tud_audio_set_itf_cb(uint8_t rhport,
                                     const tusb_control_request_t* request) {
  (void)rhport;
  const uint8_t itf = TU_U16_LOW(request->wIndex);
  const uint8_t alt = TU_U16_LOW(request->wValue);
  if (itf == ITF_NUM_AUDIO_STREAMING) {
    usb::is_streaming = alt != 0;
  }
  return true;
}

extern "C" bool tud_audio_set_itf_close_ep_cb(
    uint8_t rhport, const tusb_control_request_t* request) {
  (void)rhport;
  if (TU_U16_LOW(request->wIndex) == ITF_NUM_AUDIO_STREAMING) {
    usb::is_streaming = false;
  }
  return true;
}

extern "C" bool tud_audio_get_req_entity_cb(
    uint8_t rhport, const tusb_control_request_t* request) {
  const uint8_t channel = TU_U16_LOW(request->wValue);
  const uint8_t control = TU_U16_HIGH(request->wValue);
  const uint8_t entity = TU_U16_HIGH(request->wIndex);

  switch (entity) {
    case UAC2_ENTITY_INPUT_TERMINAL:
      if (control == AUDIO20_TE_CTRL_CONNECTOR) {
        audio20_desc_channel_cluster_t cluster = {};
        cluster.bNrChannels = 1;
        return tud_audio_buffer_and_schedule_control_xfer(
            rhport, request, &cluster, sizeof(cluster));
      }
      return false;

    case UAC2_ENTITY_FEATURE_UNIT:
      if (channel >= usb::mute.size()) {
        return false;
      }
      if (control == AUDIO20_FU_CTRL_MUTE) {
        return tud_audio_buffer_and_schedule_control_xfer(
            rhport, request, &usb::mute[channel], 1);
      }
      if (control == AUDIO20_FU_CTRL_VOLUME) {
        if (request->bRequest == AUDIO20_CS_REQ_CUR) {
          audio20_control_cur_2_t cur = {};
          cur.bCur = usb::volume[channel];
          return tud_audio_buffer_and_schedule_control_xfer(
              rhport, request, &cur, sizeof(cur));
        }
        if (request->bRequest == AUDIO20_CS_REQ_RANGE) {
          audio20_control_range_2_n_t(1) range = {};
          range.wNumSubRanges = 1;
          range.subrange[0].bMin = usb::kVolumeMin;
          range.subrange[0].bMax = usb::kVolumeMax;
          range.subrange[0].bRes = usb::kVolumeStep;
          return tud_audio_buffer_and_schedule_control_xfer(
              rhport, request, &range, sizeof(range));
        }
      }
      return false;

    case UAC2_ENTITY_CLOCK_SOURCE:
      if (control == AUDIO20_CS_CTRL_SAM_FREQ) {
        if (request->bRequest == AUDIO20_CS_REQ_CUR) {
          audio20_control_cur_4_t cur = {};
          cur.bCur = CFG_TUD_AUDIO_FUNC_1_SAMPLE_RATE;
          return tud_audio_buffer_and_schedule_control_xfer(
              rhport, request, &cur, sizeof(cur));
        }
        if (request->bRequest == AUDIO20_CS_REQ_RANGE) {
          audio20_control_range_4_n_t(1) range = {};
          range.wNumSubRanges = 1;
          range.subrange[0].bMin = CFG_TUD_AUDIO_FUNC_1_SAMPLE_RATE;
          range.subrange[0].bMax = CFG_TUD_AUDIO_FUNC_1_SAMPLE_RATE;
          range.subrange[0].bRes = 0;
          return tud_audio_buffer_and_schedule_control_xfer(
              rhport, request, &range, sizeof(range));
        }
      }
      if (control == AUDIO20_CS_CTRL_CLK_VALID) {
        uint8_t valid = 1;
        return tud_audio_buffer_and_schedule_control_xfer(
            rhport, request, &valid, sizeof(valid));
      }
      return false;

    default:
      return false;
  }
}

extern "C" bool tud_audio_set_req_entity_cb(
    uint8_t rhport, const tusb_control_request_t* request, uint8_t* buffer) {
  (void)rhport;
  const uint8_t channel = TU_U16_LOW(request->wValue);
  const uint8_t control = TU_U16_HIGH(request->wValue);
  const uint8_t entity = TU_U16_HIGH(request->wIndex);

  if (entity != UAC2_ENTITY_FEATURE_UNIT ||
      request->bRequest != AUDIO20_CS_REQ_CUR ||
      channel >= usb::mute.size()) {
    return false;
  }
  if (control == AUDIO20_FU_CTRL_MUTE &&
      request->wLength == sizeof(audio20_control_cur_1_t)) {
    usb::mute[channel] =
        reinterpret_cast<const audio20_control_cur_1_t*>(buffer)->bCur;
    usb::UpdateGain();
    return true;
  }
  if (control == AUDIO20_FU_CTRL_VOLUME &&
      request->wLength == sizeof(audio20_control_cur_2_t)) {
    const int16_t requested =
        reinterpret_cast<const audio20_control_cur_2_t*>(buffer)->bCur;
    usb::volume[channel] =
        std::clamp(requested, usb::kVolumeMin, usb::kVolumeMax);
    usb::UpdateGain();
    return true;
  }
  return false;
}
