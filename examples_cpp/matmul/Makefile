EXE=a.out
CPP=g++
CPPFLAGS= -Wall -g -std=c++17

NVCC=nvcc
NVCCFLAGS= -arch=sm_52 -Wno-deprecated-gpu-targets
LDFLAGS = -lnvToolsExt

INC=matmul.h

SRC_CU=main.cu matmul.cu
SRC_CPP=
SRC=$(SRC_CPP) $(SRC_CU)

.SUFFIXES:
.SUFFIXES: .cpp .cu .o

OBJ_CU=$(SRC_CU:.cu=.o)
OBJ_CPP=$(SRC_CPP:.cpp=.o)
OBJ=$(OBJ_CPP) $(OBJ_CU)

all: $(EXE)

%.o: %.cpp
	$(CPP) $(CPPFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(EXE): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJ) $(LFLAGS)

$(OBJ): $(INC)

clean:
	rm -rf $(EXE) $(OBJ)
