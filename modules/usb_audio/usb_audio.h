// Presents the board to the host as a USB Audio Class 2 microphone
// ("device_ml mic", 16 kHz, 16-bit mono) on the micro-USB connector CN5.
// macOS, Linux and Windows use it with their built-in drivers.

#pragma once

#include <cstddef>
#include <cstdint>

namespace device_ml::usb_audio {

// Brings up the OTG FS pins and clock and starts the TinyUSB device task.
// Call once, after the FreeRTOS scheduler is running.
void Start();

// True once the host has enumerated and configured the device.
bool mounted();

// True while a host app has the mic open (UAC2 streaming alt setting 1).
bool streaming();

// Queues 16 kHz mono samples for the host, applying the host's mute and
// volume settings. Dropped when no host app is streaming, or if the queue is
// full. Returns the number of samples queued.
// `mark` is carried to the host in the lowest bit of every sample (set when
// true, cleared when false), which is inaudible. The firmware uses it to
// say "the enrolled voice is speaking"; tools/host/record.py --only-me
// reads it.
size_t Write(const int16_t* samples, size_t count, bool mark = false);

}  // namespace device_ml::usb_audio
