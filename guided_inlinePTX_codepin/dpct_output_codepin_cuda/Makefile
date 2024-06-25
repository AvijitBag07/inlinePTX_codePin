##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS=
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=
NVCC_LIBS=

# Include header file diretory:
INC_DIR = ../dpct_output/include/
COMMON_DIR = Common 

OUTPUT = inlinePTX_cuda

all:
	$(NVCC) Samples/2_Concepts_and_Techniques/inlinePTX/inlinePTX.cu -I $(INC_DIR) -I $(COMMON_DIR) -o $(OUTPUT)

clean:
	rm -rf $(OUTPUT) 
