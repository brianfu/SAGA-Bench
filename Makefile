# See LICENSE.txt for license details.

C = nvcc
CFLAGS = -Xcompiler -O2 -std=c++17 -w -pg
# CFLAGS = -g -G -Xcompiler -std=c++17 -w -D DEBUG=1 #For Debug

CXX = nvcc
CXXFLAGS = -Xcompiler -fopenmp -O2 -std=c++17 -w -pg
# CXXFLAGS = -g -G -Xcompiler -fopenmp -std=c++17 -w -D DEBUG=1 #For Debug

DYN_PREFIX := d_

DYN_DIR := src/dynamic
UTL_DIR := src/common
OBJ_DIR := obj
BIN_DIR := bin

DYN_SRC := $(wildcard $(DYN_DIR)/*.cc)
DYN_SRC += $(wildcard $(DYN_DIR)/*.c)
DYN_SRC += $(wildcard $(DYN_DIR)/*.cu)
DYN_SRC += $(wildcard $(UTL_DIR)/*.cc)
DYN_HDR := $(wildcard $(DYN_DIR)/*.h)
DYN_HDR += $(wildcard $(UTL_DIR)/*.h)

DYN_OBJ := $(addprefix $(OBJ_DIR)/$(DYN_PREFIX),$(notdir $(patsubst %.c,%.o,$(wildcard $(DYN_DIR)/*.c))))
DYN_OBJ += $(addprefix $(OBJ_DIR)/$(DYN_PREFIX),$(notdir $(patsubst %.cc,%.o,$(wildcard $(DYN_DIR)/*.cc))))
DYN_OBJ += $(addprefix $(OBJ_DIR)/$(DYN_PREFIX),$(notdir $(patsubst %.cu,%.o,$(wildcard $(DYN_DIR)/*.cu))))

.PHONY : all
all : $(BIN_DIR)/errorExtractor frontEnd 

$(BIN_DIR)/errorExtractor : errorExtractor.cc
	$(CXX) $(CXXFLAGS) $< -o $@

frontEnd : $(DYN_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(OBJ_DIR)/$(DYN_PREFIX)%.o : $(DYN_DIR)/%.cc $(DYN_HDR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/$(DYN_PREFIX)%.o : $(DYN_DIR)/%.c $(DYN_HDR)
	$(C) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/$(DYN_PREFIX)%.o : $(DYN_DIR)/%.cu $(DYN_HDR)
	$(C) $(CFLAGS) -c $< -o $@

.PHONY : clean

clean:
	rm -f frontEnd
	rm -f $(OBJ_DIR)/*.o
	rm -f $(BIN_DIR)/*