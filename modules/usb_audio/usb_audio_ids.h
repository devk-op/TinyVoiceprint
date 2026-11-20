// Interface numbers and audio entity IDs shared by the descriptors and the
// class request handlers. The entity IDs are fixed by TinyUSB's
// TUD_AUDIO20_MIC_ONE_CH_DESCRIPTOR.

#pragma once

enum {
  ITF_NUM_AUDIO_CONTROL = 0,
  ITF_NUM_AUDIO_STREAMING,
  ITF_NUM_TOTAL,
};

#define UAC2_ENTITY_INPUT_TERMINAL 0x01
#define UAC2_ENTITY_FEATURE_UNIT 0x02
#define UAC2_ENTITY_OUTPUT_TERMINAL 0x03
#define UAC2_ENTITY_CLOCK_SOURCE 0x04
