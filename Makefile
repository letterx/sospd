CXX ?= g++
DEFS = 
INCLUDES = -I. -I./higher-order-energy/include -I./higher-order-energy/qpbo
CXX_FLAGS = -O3 -Wall -std=c++11 $(INCLUDES)
LD_FLAGS = 
LIBS = 
TEST_DIR = ./test

TEST_SRCS = $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJS = $(TEST_SRCS:.cpp=.o)
CORE_SRCS = submodular-flow.cpp \
			gen-random.cpp
CORE_OBJS = $(CORE_SRCS:.cpp=.o)

QPBO_DIR = ./higher-order-energy/qpbo
QPBO_SRCS = $(QPBO_DIR)/QPBO.cpp
QPBO_OBJS = $(QPBO_SRCS:.cpp=.o)

SRCS = $(CORE_SRCS) $(TEST_SRCS) $(QPBO_SRCS)
OBJS = $(SRCS:.cpp=.o)

.PHONY: all
all: $(OBJS) unit-test higher-order-experiment

unit-test: $(CORE_OBJS) $(TEST_OBJS) $(QPBO_OBJS)
	$(CXX) $(CXX_FLAGS) $(LD_FLAGS) -o $@ $(CORE_OBJS) $(TEST_OBJS) $(QPBO_OBJS) $(LIBS) -lboost_unit_test_framework 

higher-order-experiment: higher-order-experiment.o $(CORE_OBJS) $(QPBO_OBJS)
	$(CXX) $(CXX_FLAGS) $(LD_FLAGS) -o $@ higher-order-experiment.o $(CORE_OBJS) $(QPBO_OBJS) $(LIBS)


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
	rm -rf $(TEST_DIR)/*.o
	rm -rf $(TEST_DIR)/*.P
	rm -rf $(TEST_DIR)/*.d
	rm -rf $(QPBO_DIR)/*.o
	rm -rf $(QPBO_DIR)/*.P
	rm -rf $(QPBO_DIR)/*.d

.PHONY: distclean
distclean: clean

