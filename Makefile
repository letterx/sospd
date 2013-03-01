CXX ?= g++
DEFS = 
INCLUDES = -I.
CXX_FLAGS = -g -Wall -std=c++11 $(INCLUDES)
LD_FLAGS = 
LIBS = 
TEST_DIR = ./test

TEST_SRCS = $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJS = $(TEST_SRCS:.cpp=.o)
CORE_SRCS = submodular-flow.cpp
CORE_OBJS = $(CORE_SRCS:.cpp=.o)

SRCS = $(CORE_SRCS) $(TEST_SRCS)
OBJS = $(SRCS:.cpp=.o)

.PHONY: all
all: $(OBJS) unit-test

unit-test: $(CORE_OBJS) $(TEST_OBJS)
	$(CXX) $(CXX_FLAGS) $(LD_FLAGS) -o $@ $(CORE_OBJS) $(TEST_OBJS) $(LIBS) -lboost_unit_test_framework 

%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -MMD -o $@ -c $<
	@cp $*.d $*.P; \
	 sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
    	 -e '/^$$/ d' -e 's/$$/ :/' < $*.d >> $*.P; \
	 rm $*.d

-include $(SRCS:.cpp=.P)

.PHONY: clean
clean: 
	rm $(OBJS)
	rm -rf *.o
	rm -rf *.P
	rm -rf *.d
	rm -rf $(TEST_DIR)/*.o
	rm -rf $(TEST_DIR)/*.P
	rm -rf $(TEST_DIR)/*.d

.PHONY: distclean
distclean: clean

