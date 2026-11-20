#include "modules/mic/mp45dt02.h"

#include "FreeRTOS.h"
#include "stm32f4xx.h"
#include "task.h"

namespace device_ml::mic {
namespace {

// Above configLIBRARY_MAX_SYSCALL_INTERRUPT_PRIORITY (5) so the ISR may call
// FreeRTOS FromISR APIs.
constexpr uint32_t kDmaIrqPriority = 6;

// A block arrives every 2 ms; this long without one means capture stalled.
constexpr TickType_t kBlockTimeout = pdMS_TO_TICKS(20);

I2S_HandleTypeDef i2s2;
DMA_HandleTypeDef dma_rx;

// Two halves of kBlockWords each; DMA fills one while the task reads the
// other.
uint16_t dma_buffer[2 * Mp45dt02::kBlockWords];

TaskHandle_t waiting_task = nullptr;
volatile size_t ready_half = 0;

pw::Status ToStatus(HAL_StatusTypeDef status) {
  return status == HAL_OK ? pw::OkStatus() : pw::Status::Internal();
}

// The I2S peripheral clock must be an exact multiple of the 1.024 MHz PDM
// bit clock. PLLI2S input is HSE/PLLM = 1 MHz; x128 / 5 = 25.6 MHz, which
// HAL_I2S_Init divides by 25 (I2SDIV=12, ODD=1) to get exactly 1.024 MHz.
pw::Status ConfigureI2sClock() {
  RCC_PeriphCLKInitTypeDef clock = {};
  clock.PeriphClockSelection = RCC_PERIPHCLK_I2S;
  clock.PLLI2S.PLLI2SN = 128;
  clock.PLLI2S.PLLI2SR = 5;
  return ToStatus(HAL_RCCEx_PeriphCLKConfig(&clock));
}

void ConfigurePins() {
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();

  GPIO_InitTypeDef gpio = {};
  gpio.Mode = GPIO_MODE_AF_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_HIGH;
  gpio.Alternate = GPIO_AF5_SPI2;

  gpio.Pin = GPIO_PIN_10;  // I2S2_CK -> mic CLK
  HAL_GPIO_Init(GPIOB, &gpio);

  gpio.Pin = GPIO_PIN_3;  // I2S2_SD <- mic DOUT
  HAL_GPIO_Init(GPIOC, &gpio);
}

pw::Status ConfigureDma() {
  __HAL_RCC_DMA1_CLK_ENABLE();

  // SPI2_RX is DMA1 Stream 3, Channel 0 (RM0090 table 42).
  dma_rx.Instance = DMA1_Stream3;
  dma_rx.Init.Channel = DMA_CHANNEL_0;
  dma_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;
  dma_rx.Init.PeriphInc = DMA_PINC_DISABLE;
  dma_rx.Init.MemInc = DMA_MINC_ENABLE;
  dma_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_HALFWORD;
  dma_rx.Init.MemDataAlignment = DMA_MDATAALIGN_HALFWORD;
  dma_rx.Init.Mode = DMA_CIRCULAR;
  dma_rx.Init.Priority = DMA_PRIORITY_HIGH;
  dma_rx.Init.FIFOMode = DMA_FIFOMODE_DISABLE;
  if (HAL_DMA_Init(&dma_rx) != HAL_OK) {
    return pw::Status::Internal();
  }
  __HAL_LINKDMA(&i2s2, hdmarx, dma_rx);

  HAL_NVIC_SetPriority(DMA1_Stream3_IRQn, kDmaIrqPriority, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream3_IRQn);
  return pw::OkStatus();
}

// Matches ST's F4-Discovery BSP for this mic: I2S frames of 2 x 16 bits at
// 32 kHz give a 1.024 MHz bit clock, and every received bit is one PDM bit
// (the L/R split is meaningless here; it's one continuous bitstream).
pw::Status ConfigureI2s() {
  __HAL_RCC_SPI2_CLK_ENABLE();

  i2s2.Instance = SPI2;
  i2s2.Init.Mode = I2S_MODE_MASTER_RX;
  i2s2.Init.Standard = I2S_STANDARD_LSB;
  i2s2.Init.DataFormat = I2S_DATAFORMAT_16B;
  i2s2.Init.MCLKOutput = I2S_MCLKOUTPUT_DISABLE;
  i2s2.Init.AudioFreq = I2S_AUDIOFREQ_32K;
  i2s2.Init.CPOL = I2S_CPOL_HIGH;
  i2s2.Init.ClockSource = I2S_CLOCK_PLL;
  i2s2.Init.FullDuplexMode = I2S_FULLDUPLEXMODE_DISABLE;
  return ToStatus(HAL_I2S_Init(&i2s2));
}

void OnBlockReady(size_t half) {
  ready_half = half;
  BaseType_t higher_priority_woken = pdFALSE;
  if (waiting_task != nullptr) {
    vTaskNotifyGiveFromISR(waiting_task, &higher_priority_woken);
  }
  portYIELD_FROM_ISR(higher_priority_woken);
}

}  // namespace

pw::Status Mp45dt02::Start() {
  waiting_task = xTaskGetCurrentTaskHandle();

  if (pw::Status status = ConfigureI2sClock(); !status.ok()) {
    return status;
  }
  ConfigurePins();
  if (pw::Status status = ConfigureDma(); !status.ok()) {
    return status;
  }
  if (pw::Status status = ConfigureI2s(); !status.ok()) {
    return status;
  }
  return ToStatus(HAL_I2S_Receive_DMA(&i2s2, dma_buffer, 2 * kBlockWords));
}

const uint16_t* Mp45dt02::WaitForBlock() {
  const uint32_t pending = ulTaskNotifyTake(pdTRUE, kBlockTimeout);
  if (pending == 0) {
    return nullptr;
  }
  dropped_blocks_ += pending - 1;
  return &dma_buffer[ready_half * kBlockWords];
}

}  // namespace device_ml::mic

extern "C" void DMA1_Stream3_IRQHandler(void) {
  HAL_DMA_IRQHandler(&device_ml::mic::dma_rx);
}

// HAL's I2S DMA completion hooks are global; I2S2 is the only I2S in use.
extern "C" void HAL_I2S_RxHalfCpltCallback(I2S_HandleTypeDef* hi2s) {
  if (hi2s->Instance == SPI2) {
    device_ml::mic::OnBlockReady(0);
  }
}

extern "C" void HAL_I2S_RxCpltCallback(I2S_HandleTypeDef* hi2s) {
  if (hi2s->Instance == SPI2) {
    device_ml::mic::OnBlockReady(1);
  }
}
