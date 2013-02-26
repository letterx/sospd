CXX ?= g++
DEFS = 
INCLUDES = 
CXX_FLAGS = -g -Wall -std=c++11 $(INCLUDES)
LD_FLAGS = 
LIBS = 

SRCS = submodular-flow.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all
all: $(OBJS)

%.o: %.cxx
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

.PHONY: distclean
distclean: clean

