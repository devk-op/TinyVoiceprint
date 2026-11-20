#include "modules/accel/lis3dsh.h"

#include "stm32f4xx_hal.h"

namespace device_ml::accel {
namespace {

// Register map (LIS3DSH datasheet, DocID022405).
constexpr uint8_t kWhoAmI = 0x0F;
constexpr uint8_t kCtrlReg4 = 0x20;
constexpr uint8_t kCtrlReg5 = 0x24;
constexpr uint8_t kOutXL = 0x28;

constexpr uint8_t kWhoAmILis3dsh = 0x3F;
constexpr uint8_t kWhoAmILis302dl = 0x3B;

// CTRL_REG4: ODR = 100 Hz (0b0110 << 4), BDU = 1 (bit 3), Z/Y/X enable.
constexpr uint8_t kCtrlReg4Value = 0x6F;
// CTRL_REG5: +/-2g full scale, 800 Hz anti-aliasing filter, 4-wire SPI.
constexpr uint8_t kCtrlReg5Value = 0x00;

constexpr uint8_t kReadBit = 0x80;
constexpr uint32_t kSpiTimeoutMs = 10;

SPI_HandleTypeDef spi1;

void ChipSelect(bool active) {
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_3, active ? GPIO_PIN_RESET : GPIO_PIN_SET);
}

pw::Status ToStatus(HAL_StatusTypeDef status) {
  switch (status) {
    case HAL_OK:
      return pw::OkStatus();
    case HAL_TIMEOUT:
      return pw::Status::DeadlineExceeded();
    case HAL_BUSY:
      return pw::Status::Unavailable();
    default:
      return pw::Status::Internal();
  }
}

pw::Status InitSpi() {
  __HAL_RCC_SPI1_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();

  GPIO_InitTypeDef gpio = {};
  gpio.Pin = GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7;  // SCK, MISO, MOSI
  gpio.Mode = GPIO_MODE_AF_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_HIGH;
  gpio.Alternate = GPIO_AF5_SPI1;
  HAL_GPIO_Init(GPIOA, &gpio);

  gpio = {};
  gpio.Pin = GPIO_PIN_3;  // Chip select, active low
  gpio.Mode = GPIO_MODE_OUTPUT_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_HIGH;
  HAL_GPIO_Init(GPIOE, &gpio);
  ChipSelect(false);

  spi1.Instance = SPI1;
  spi1.Init.Mode = SPI_MODE_MASTER;
  spi1.Init.Direction = SPI_DIRECTION_2LINES;
  spi1.Init.DataSize = SPI_DATASIZE_8BIT;
  spi1.Init.CLKPolarity = SPI_POLARITY_HIGH;  // SPI mode 3
  spi1.Init.CLKPhase = SPI_PHASE_2EDGE;
  spi1.Init.NSS = SPI_NSS_SOFT;
  // APB2 = 84 MHz / 16 = 5.25 MHz, under the LIS3DSH's 10 MHz limit.
  spi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_16;
  spi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  spi1.Init.TIMode = SPI_TIMODE_DISABLE;
  spi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  spi1.Init.CRCPolynomial = 10;
  return ToStatus(HAL_SPI_Init(&spi1));
}

}  // namespace

pw::Status Lis3dsh::ReadRegister(uint8_t reg, uint8_t& value) {
  uint8_t tx[2] = {static_cast<uint8_t>(reg | kReadBit), 0x00};
  uint8_t rx[2] = {};
  ChipSelect(true);
  HAL_StatusTypeDef status =
      HAL_SPI_TransmitReceive(&spi1, tx, rx, sizeof(tx), kSpiTimeoutMs);
  ChipSelect(false);
  value = rx[1];
  return ToStatus(status);
}

pw::Status Lis3dsh::WriteRegister(uint8_t reg, uint8_t value) {
  uint8_t tx[2] = {reg, value};
  ChipSelect(true);
  HAL_StatusTypeDef status = HAL_SPI_Transmit(&spi1, tx, sizeof(tx), kSpiTimeoutMs);
  ChipSelect(false);
  return ToStatus(status);
}

pw::Status Lis3dsh::Init() {
  if (pw::Status status = InitSpi(); !status.ok()) {
    return status;
  }

  uint8_t who_am_i = 0;
  if (pw::Status status = ReadRegister(kWhoAmI, who_am_i); !status.ok()) {
    return status;
  }
  if (who_am_i == kWhoAmILis302dl) {
    chip_ = Chip::kLis302dl;
    return pw::Status::Unimplemented();  // Older MB997B board revision.
  }
  if (who_am_i != kWhoAmILis3dsh) {
    chip_ = Chip::kUnknown;
    return pw::Status::NotFound();
  }
  chip_ = Chip::kLis3dsh;

  if (pw::Status status = WriteRegister(kCtrlReg4, kCtrlReg4Value); !status.ok()) {
    return status;
  }
  return WriteRegister(kCtrlReg5, kCtrlReg5Value);
}

pw::Status Lis3dsh::Read(Sample& out) {
  if (chip_ != Chip::kLis3dsh) {
    return pw::Status::FailedPrecondition();
  }
  // OUT_X_L .. OUT_Z_H are six consecutive registers. Read them one at a
  // time so this does not depend on CTRL_REG6's address auto-increment.
  uint8_t bytes[6] = {};
  for (uint8_t i = 0; i < 6; ++i) {
    if (pw::Status status = ReadRegister(kOutXL + i, bytes[i]); !status.ok()) {
      return status;
    }
  }
  auto axis = [&](int lo) {
    return RawToMilliG(static_cast<int16_t>(bytes[lo] | (bytes[lo + 1] << 8)));
  };
  out = {axis(0), axis(2), axis(4)};
  return pw::OkStatus();
}

}  // namespace device_ml::accel
