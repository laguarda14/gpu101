EXE       = matmul
CPP       = g++
CPPFLAGS  = -Wall -O2 -std=c++17

NVCC      = nvcc
NVCCFLAGS = -arch=sm_86 -O2 -lineinfo -Xcompiler -Wall -Wno-deprecated-gpu-targets
LDFLAGS   = -lnvToolsExt
INC       = matmul.h

SRC_CU    = matmul.cu
SRC_CPP   = main.cpp
SRC       = $(SRC_CPP) $(SRC_CU)

.SUFFIXES:
.SUFFIXES: .cpp .cu .o

OBJ_CU    = $(SRC_CU:.cu=.o)
OBJ_CPP   = $(SRC_CPP:.cpp=.o)
OBJ       = $(OBJ_CPP) $(OBJ_CU)

.PHONY: all clean run

all: $(EXE)

%.o: %.cpp $(INC)
	$(CPP) $(CPPFLAGS) -c $< -o $@

%.o: %.cu $(INC)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(EXE): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJ) $(LFLAGS)

clean:
	rm -rf $(EXE) $(OBJ)

run: all
	./$(EXE)
