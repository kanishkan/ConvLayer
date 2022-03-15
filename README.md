# ConvLayer Implementation on Embedded ASIP
5LIM0 - Lab3 assignment

## Folder structure
* `main.c` - ConvLayer C code
* `adf/base_tta.adf` - TTA reference design
* `data` - Input data to the network
    * `input.bin` - Inputs
    * `bias.bin` - Bias
    * `weight.bin` - Weight
    * `output.bin` - Reference output
* `Makefile` - Build system
    * `make tce` - Compiles C-code to TTA program (`tpef` format)
    * `make tcesim` - Compiles and run the program on TCE simulator (`ttasim`) and returns runtime.
    * `make energy` - Reports energy numbers for the design (energy per FU and instructions)
* `scripts\`
    * `tce_energy_model.py` - Script for analytical energy model
    * `tile_data.py` - Same script to tile the data
* `fpga\`
    * Scripts related fpga design

## Example flow

```bash
git clone https://github.com/kanishkan/convlayer.git
cd ConvLayer

# Compile and run the code on TCE simulator
make tcesim

$ Runtime: xxx cycles
```
