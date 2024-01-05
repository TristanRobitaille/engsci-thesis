## Using the Coral Edge TPU as an embedded system
Gathering some findings about the possibility of using the Coral Edge TPU in an embedded system to act as a prototype of the full device.

### Coral Edge TPU
- [Overview](https://coral.ai/technology#performance)
- Has USB, PCIe and I2C comms. For comms with MCU, use USB.
- [Accelerator module](https://coral.ai/products/accelerator-module): Smallest Edge TPU product, this is the one you'll want to get for a custom MCU project (or desolder the Edge TPU from the [PCIe accelerator](https://coral.ai/products/pcie-accelerator)...). Includes Edge TPU and its power modules.
- Electrical level documentation for the Edge TPU is very sparse. I haven't module documentation for the Edge TPU itself, only for the accelerator module. Might want to contact them (since they projects are open-source, they might be willing to share the datasheet)

### [Coral Dev Board Micro](https://coral.ai/products/dev-board-micro)
- 30mm x 65mm USB-powered dev board w/ NXP i.MX RT1176 (Cortex-M7 and Cortex-M4) MCU + Edge TPU
- 64MB RAM, 128 MiB flash, camera/mic, LEDs/switches
- Coral recommends use of [FreeRTOS](https://coral.ai/docs/dev-board-micro/freertos/) and provides its [coralmicro](https://coral.ai/docs/reference/micro/) C++ library to interface with peripherals.
- It is [Arduino-compatible](https://coral.ai/docs/dev-board-micro/arduino/)
- [Schematics are open-source](https://github.com/google-coral/electricals/tree/master/dev_board_micro)

### [coralmicro](https://github.com/google-coral/coralmicro/tree/main)
- C++ library used to delegate model operations to Edge TPU
- Build system based on CMake

### [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- TensorFlow's library for running models on MCU. Supports ARM Cortex-M, ESP32 and others.

### Summary
- Easiest solution: Get the Coral Dev Board
- 2nd easiest solution: Make own PCB with similar components as the Dev Board Micro so library adaptation can be minimal. Can probably downsize MCU, remove peripherals.
- Most involved solution: Use `coralmicro` and `TensorFlow for Microcontrollers` (as highlighted [here](https://coral.ai/docs/edgetpu/inference/#microcontroller-systems)) on ARM Cortex-M or ESP32.
