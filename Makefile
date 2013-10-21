CXX ?= g++
AR = ar
DEFS = 
INCLUDES = -I./include -I./submodular-flow/ -I./higher-order-energy/include -I./higher-order-energy/qpbo
OPT ?= -O3
CXX_FLAGS = $(OPT) -Wall -std=c++11 $(INCLUDES)
LD_FLAGS = 
LIBS = -lboost_serialization
TEST_DIR = ./test
LIB_DIR = ./lib

TEST_SRCS = $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJS = $(TEST_SRCS:.cpp=.o)
CORE_SRCS = submodular-flow/submodular-flow.cpp \
			submodular-flow/submodular-ibfs.cpp \
			submodular-flow/gen-random.cpp \
			submodular-flow/alpha-expansion.cpp \
			submodular-flow/spd2.cpp \
			submodular-flow/dgfm.cpp \
			submodular-flow/submodular-functions.cpp
CORE_OBJS = $(CORE_SRCS:.cpp=.o)

QPBO_DIR = ./higher-order-energy/qpbo
QPBO_SRCS = $(QPBO_DIR)/QPBO.cpp
QPBO_OBJS = $(QPBO_SRCS:.cpp=.o)

SF_LIB = $(LIB_DIR)/libsubmodular-flow.a

SRCS = $(CORE_SRCS) $(TEST_SRCS) $(QPBO_SRCS)
OBJS = $(SRCS:.cpp=.o)

.PHONY: all
all: $(SF_LIB) unit-test higher-order-experiment recover-crash add-noise submodular-flow experiment

.PHONY: submodular-flow
submodular-flow:
	$(MAKE) -C ./submodular-flow/

.PHONY: experiment
experiment: $(SF_LIB)
	$(MAKE) -C ./experiments/denoising/

unit-test: $(TEST_OBJS) $(SF_LIB)
	$(CXX) $(CXX_FLAGS) $(LD_FLAGS) -o $@ $(TEST_OBJS) $(SF_LIB) $(LIBS) -lboost_unit_test_framework 

higher-order-experiment: higher-order-experiment.o $(SF_LIB)
	$(CXX) $(CXX_FLAGS) $(LD_FLAGS) -o $@ higher-order-experiment.o $(SF_LIB) $(LIBS)

recover-crash: recover-crash.o $(SF_LIB)
	$(CXX) $(CXX_FLAGS) $(LD_FLAGS) -o $@ recover-crash.o $(SF_LIB) $(LIBS)

add-noise: add-noise.cpp
	$(CXX) $(CXX_FLAGS) $(LD_FLAGS) -o $@ add-noise.cpp $(LIBS) -lopencv_core -lopencv_imgproc -lopencv_highgui

$(SF_LIB): $(CORE_OBJS) $(QPBO_OBJS)
	mkdir -p $(LIB_DIR)
	$(AR) rcs $(SF_LIB) $(CORE_OBJS) $(QPBO_OBJS)

%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -MMD -o $@ -c $<
	@cp $*.d $*.P; \
	 sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
    	 -e '/^$$/ d' -e 's/$$/ :/' < $*.d >> $*.P; \
	 rm $*.d

-include $(SRCS:.cpp=.P)

.PHONY: clean
clean: 
	rm -rf $(OBJS)
	rm -rf *.o
	rm -rf *.P
	rm -rf *.d
	rm -rf *~
	rm -rf $(TEST_DIR)/*.o
	rm -rf $(TEST_DIR)/*.P
	rm -rf $(TEST_DIR)/*.d
	rm -rf $(TEST_DIR)/*~
	rm -rf $(QPBO_DIR)/*.o
	rm -rf $(QPBO_DIR)/*.P
	rm -rf $(QPBO_DIR)/*.d
	rm -rf $(QPBO_DIR)/*~
	rm -rf submodular-flow/*.o
	rm -rf submodular-flow/*.P
	rm -rf submodular-flow/*.d
	rm -rf submodular-flow/*~
	cd submodular-flow; $(MAKE) clean

.PHONY: distclean
distclean: clean
	rm -rf unit-test
	rm -rf higher-order-experiment
	rm -rf ./lib
	rm -rf recover-crash
	rm -rf add-noise
	cd submodular-flow; $(MAKE) distclean

.PHONY: check
check: ./unit-test
	./unit-test
