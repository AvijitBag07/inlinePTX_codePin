##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=icpx
CC_FLAGS=-fsycl
CC_LIBS=


# Include header file diretory:
INC_DIR = ../dpct_output/include/
COMMON_DIR = Common 
OUTPUT = inlinePTX_sycl

all:
	$(CC) $(CC_FLAGS) Samples/2_Concepts_and_Techniques/inlinePTX/inlinePTX.dp.cpp -I $(INC_DIR) -I $(COMMON_DIR) -o $(OUTPUT)

clean:
	rm -rf $(OUTPUT) 
