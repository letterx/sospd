CXX ?= g++
DEFS = 
BASE_DIR = ../../
INCLUDES = -I$(BASE_DIR)include -I$(BASE_DIR)/submodular-flow/ -I$(BASE_DIR)/higher-order-energy/include -I$(BASE_DIR)/higher-order-energy/qpbo
OPT ?= -O3
CXX_FLAGS = $(OPT) -Wall -std=c++11 $(INCLUDES)
LD_FLAGS = -L$(BASE_DIR)/lib/
LIBS = -lsubmodular-flow -lboost_serialization -lopencv_core -lopencv_imgproc -lopencv_highgui -lboost_program_options
SF_LIB = ../../lib/libsubmodular-flow.a

SRCS = stereo.cpp disparity-diff.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all
all: stereo disparity-diff

stereo: stereo.o $(SF_LIB)
	$(CXX) $(CXX_FLAGS) $(LD_FLAGS) -o $@ stereo.o $(LIBS)

disparity-diff: disparity-diff.o
	$(CXX) $(CXX_FLAGS) $(LD_FLAGS) -o $@ disparity-diff.o -lopencv_core -lopencv_highgui

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

.PHONY: distclean
distclean: clean
	rm -rf stereo
