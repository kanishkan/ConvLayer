# Makefile for host and TCE build
# Optionally, you can test the code on host with `make test`

# Host dependency
HOST_CC = gcc

# TCE flags
TCE_CC = tcecc
TCC_FLAGS = -O3 --bubblefish2-scheduler

ADF_FILE := adf/base_tta.adf
IDF_PATH = adf/base_tta.idf
SYMBOL_NAMES=output,input,weight,bias

# Input data
INPUT_DATA=data/input.bin
WEIGHT_DATA=data/weight.bin
BIAS_DATA=data/bias.bin

# Expected output
OUTPUT_DATA=data/output.bin

SIM_CMD = 'load_data input $(INPUT_DATA); load_data weight $(WEIGHT_DATA); \
		  load_data bias $(BIAS_DATA); \
		  run; x /u b /n 51200 /f data/tce_output.bin output; info proc cycles; '

# Benchmark source
SRC := main.c

.PHONY : clean all host tce header tcesim vhdl compare

main.tpef: $(SRC) $(ADF_FILE)
	$(TCE_CC) $(TCC_FLAGS) -a $(ADF_FILE) -o main.tpef $(SRC) -k $(SYMBOL_NAMES)

tce: main.tpef

tcesim: main.tpef
	echo "Runtime: $$(echo quit | ttasim -a $(ADF_FILE) -p main.tpef -e $(SIM_CMD) | sed -n 2p) cycles"

compare: tcesim
	python3 ./scripts/compare.py $(OUTPUT_DATA) data/tce_output.bin

log.txt: main.tpef
	echo "$$(echo quit | ttasim -a $(ADF_FILE) -p main.tpef -e 'run; info proc stats;')\n" > log.txt

energy: log.txt ./scripts/tce_energy_model.py
	python3 ./scripts/tce_energy_model.py --adf $(ADF_FILE) --log log.txt

asm: main.tpef
	tcedisasm -o main.asm $(ADF_FILE) main.tpef

vhdl: tce
	rm -rf proge-out *.img
	generateprocessor -d onchip -f onchip -e tta_core -i $(IDF_PATH) -g AlmaIFIntegrator -o proge-out -p main.tpef $(ADF_FILE)
	generatebits --verbose -d -w 4 -e tta_core -p main.tpef -x proge-out $(ADF_FILE)

all: host tce

clean:
	@rm -rf *.tpef main weights.h proge-out *.asm *.img *.dot log.txt *.trace* *.S *.ll *.bc a.out data/tce_output.bin

