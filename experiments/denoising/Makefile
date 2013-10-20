CXX ?= g++
DEFS = 
BASE_DIR = ../../
INCLUDES = -I$(BASE_DIR)/submodular-flow/ -I$(BASE_DIR)/higher-order-energy/include -I$(BASE_DIR)/higher-order-energy/qpbo
OPT ?= -O3
CXX_FLAGS = $(OPT) -Wall -std=c++11 $(INCLUDES)
LD_FLAGS = -L$(BASE_DIR)/lib/
LIBS = -lsubmodular-flow -lboost_serialization
SF_LIB = $(BASE_DIR)/lib/libsubmodular-flow.a

SRCS = denoise.cpp \
	   foe-cliques.cpp \
	   image.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all
all: $(SF_LIB) denoise

denoise: denoise.o $(SF_LIB)
	$(CXX) $(CXX_FLAGS) $(LD_FLAGS) -o $@ denoise.o $(LIBS)

$(SF_LIB):
	$(MAKE) -C $(BASE_DIR)

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
	rm -rf denoise
