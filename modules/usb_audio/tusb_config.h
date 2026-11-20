// TinyUSB configuration: a full-speed USB Audio Class 2 device exposing one
// 16 kHz, 16-bit mono microphone on the STM32F407's OTG FS port.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define CFG_TUSB_MCU OPT_MCU_STM32F4
#define CFG_TUSB_OS OPT_OS_FREERTOS
#define CFG_TUSB_DEBUG 0

// OTG FS (PA11/PA12, connector CN5) is root hub port 0 on the F407.
#define BOARD_TUD_RHPORT 0
#define CFG_TUD_ENABLED 1
#define CFG_TUD_MAX_SPEED OPT_MODE_FULL_SPEED

#define CFG_TUSB_MEM_SECTION
#define CFG_TUSB_MEM_ALIGN __attribute__((aligned(4)))

#define CFG_TUD_ENDPOINT0_SIZE 64

#define CFG_TUD_AUDIO 1
#define CFG_TUD_CDC 0
#define CFG_TUD_MSC 0
#define CFG_TUD_HID 0
#define CFG_TUD_MIDI 0
#define CFG_TUD_VENDOR 0

// Audio stream: must match what //modules/pdm produces.
#define CFG_TUD_AUDIO_FUNC_1_SAMPLE_RATE 16000
#define CFG_TUD_AUDIO_FUNC_1_N_BYTES_PER_SAMPLE_TX 2
#define CFG_TUD_AUDIO_FUNC_1_N_CHANNELS_TX 1

#define CFG_TUD_AUDIO_ENABLE_EP_IN 1
#define CFG_TUD_AUDIO_EP_SZ_IN                                  \
  TUD_AUDIO_EP_SIZE(0,                                          \
                    CFG_TUD_AUDIO_FUNC_1_SAMPLE_RATE,           \
                    CFG_TUD_AUDIO_FUNC_1_N_BYTES_PER_SAMPLE_TX, \
                    CFG_TUD_AUDIO_FUNC_1_N_CHANNELS_TX)
#define CFG_TUD_AUDIO_FUNC_1_EP_IN_SZ_MAX CFG_TUD_AUDIO_EP_SZ_IN

// ~16 ms of audio. The mic delivers 2 ms blocks and USB drains 1 ms per
// frame; TinyUSB's IN flow control keeps this about half full, absorbing
// jitter and the slight clock difference between the board and the host.
#define CFG_TUD_AUDIO_FUNC_1_EP_IN_SW_BUF_SZ (16 * CFG_TUD_AUDIO_EP_SZ_IN)

#ifdef __cplusplus
}
#endif
