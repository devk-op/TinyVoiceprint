// USB descriptors for the device_ml microphone, adapted from TinyUSB's
// examples/device/audio_test. TinyUSB calls these callbacks during
// enumeration.

#include <stddef.h>
#include <stdint.h>

#include "modules/usb_audio/usb_audio_ids.h"
#include "stm32f4xx.h"
#include "tusb.h"

// TinyUSB's shared test VID. Fine for development; a product needs its own.
#define USB_VID 0xCafe
#define USB_PID 0x4D31

static const tusb_desc_device_t kDeviceDescriptor = {
    .bLength = sizeof(tusb_desc_device_t),
    .bDescriptorType = TUSB_DESC_DEVICE,
    .bcdUSB = 0x0200,
    // Audio uses an Interface Association Descriptor, which requires the
    // Misc/Common/IAD device class triple.
    .bDeviceClass = TUSB_CLASS_MISC,
    .bDeviceSubClass = MISC_SUBCLASS_COMMON,
    .bDeviceProtocol = MISC_PROTOCOL_IAD,
    .bMaxPacketSize0 = CFG_TUD_ENDPOINT0_SIZE,
    .idVendor = USB_VID,
    .idProduct = USB_PID,
    .bcdDevice = 0x0100,
    .iManufacturer = 1,
    .iProduct = 2,
    .iSerialNumber = 3,
    .bNumConfigurations = 1,
};

const uint8_t* tud_descriptor_device_cb(void) {
  return (const uint8_t*)&kDeviceDescriptor;
}

#define CONFIG_TOTAL_LEN (TUD_CONFIG_DESC_LEN + TUD_AUDIO20_MIC_ONE_CH_DESC_LEN)

#define EPNUM_AUDIO_IN 0x81

static const uint8_t kConfigurationDescriptor[] = {
    TUD_CONFIG_DESCRIPTOR(/*config_num=*/1,
                          ITF_NUM_TOTAL,
                          /*_stridx=*/0,
                          CONFIG_TOTAL_LEN,
                          /*_attribute=*/0x00,
                          /*_power_ma=*/100),
    TUD_AUDIO20_MIC_ONE_CH_DESCRIPTOR(
        /*_itfnum=*/ITF_NUM_AUDIO_CONTROL,
        /*_stridx=*/0,
        /*_nBytesPerSample=*/CFG_TUD_AUDIO_FUNC_1_N_BYTES_PER_SAMPLE_TX,
        /*_nBitsUsedPerSample=*/CFG_TUD_AUDIO_FUNC_1_N_BYTES_PER_SAMPLE_TX * 8,
        /*_epin=*/EPNUM_AUDIO_IN,
        /*_epsize=*/CFG_TUD_AUDIO_EP_SZ_IN),
};

const uint8_t* tud_descriptor_configuration_cb(uint8_t index) {
  (void)index;
  return kConfigurationDescriptor;
}

enum {
  STRID_LANGID = 0,
  STRID_MANUFACTURER,
  STRID_PRODUCT,
  STRID_SERIAL,
  STRID_COUNT,
};

static const char* const kStrings[STRID_COUNT] = {
    [STRID_MANUFACTURER] = "device_ml",
    [STRID_PRODUCT] = "device_ml mic",
};

// Serial number: the chip's 96-bit unique ID as 24 hex digits, so multiple
// boards are distinguishable on the host.
static size_t SerialNumber(uint16_t* out, size_t max_chars) {
  static const char kHex[] = "0123456789ABCDEF";
  const uint8_t* uid = (const uint8_t*)UID_BASE;
  size_t count = 0;
  for (size_t i = 0; i < 12 && count + 2 <= max_chars; ++i) {
    out[count++] = kHex[uid[i] >> 4];
    out[count++] = kHex[uid[i] & 0xF];
  }
  return count;
}

static uint16_t string_buffer[32 + 1];

const uint16_t* tud_descriptor_string_cb(uint8_t index, uint16_t langid) {
  (void)langid;
  const size_t max_chars =
      sizeof(string_buffer) / sizeof(string_buffer[0]) - 1;
  size_t count = 0;

  if (index == STRID_LANGID) {
    string_buffer[1] = 0x0409;  // English (US)
    count = 1;
  } else if (index == STRID_SERIAL) {
    count = SerialNumber(&string_buffer[1], max_chars);
  } else if (index < STRID_COUNT && kStrings[index] != NULL) {
    for (const char* c = kStrings[index]; *c != '\0' && count < max_chars;
         ++c) {
      string_buffer[1 + count++] = (uint16_t)*c;
    }
  } else {
    return NULL;
  }

  // First element: total length in bytes (incl. header) and descriptor type.
  string_buffer[0] = (uint16_t)((TUSB_DESC_STRING << 8) | (2 * count + 2));
  return string_buffer;
}
