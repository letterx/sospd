CXX ?= g++
AR = ar
DEFS = 
OPT ?= -O3
CXX_FLAGS = $(OPT) -Wall -std=c++11 
LD_FLAGS = 
LIBS = -lboost_serialization -lopencv_core -lopencv_imgproc -lopencv_highgui
CORE_SRCS = submodular-flow.cpp \
			submodular-ibfs.cpp 
CORE_OBJS = $(CORE_SRCS:.cpp=.o)

.PHONY: all
all: ibfs_test sos_ibfs_test submodular-ibfs spd2 denoise_test

ibfs_test: ibfs_test.cpp ibfs.o
	$(CXX) $(CXX_FLAGS) -o $@ ibfs.o ibfs_test.cpp
	
sos_ibfs_test: sos_ibfs_test.cpp submodular-ibfs.o
	$(CXX) $(CXX_FLAGS) -o $@ submodular-ibfs.o sos_ibfs_test.cpp $(LIBS)

submodular-ibfs: submodular-ibfs.cpp submodular-ibfs.hpp
	$(CXX) $(CXX_FLAGS) -c submodular-ibfs.cpp $(LIBS)

spd2: spd2.cpp spd2.hpp clique.hpp submodular-ibfs.o
	$(CXX) $(CXX_FLAGS) -c spd2.cpp submodular-ibfs.o $(LIBS)	
	
denoise_test: denoise_test.cpp spd2.o submodular-ibfs.o 
	$(CXX) $(CXX_FLAGS) -o $@ denoise_test.cpp spd2.o submodular-ibfs.o $(LIBS)	

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
