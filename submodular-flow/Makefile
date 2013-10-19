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
SRCS = $(CORE_SRCS) \
	   ibfs.cpp \
	   sos_ibfs_test.cpp \
	   submodular-ibfs.cpp \
	   spd2.cpp \
       dgfm.cpp \
	   denoise_test.cpp \
	   submodular-functions.cpp

.PHONY: all
all: ibfs_test sos_ibfs_test denoise_test submodular_test

ibfs_test: ibfs_test.o ibfs.o
	$(CXX) $(CXX_FLAGS) -o $@ ibfs.o ibfs_test.o
	
sos_ibfs_test: sos_ibfs_test.o submodular-ibfs.o
	$(CXX) $(CXX_FLAGS) -o $@ submodular-ibfs.o sos_ibfs_test.o $(LIBS)

denoise_test: denoise_test.o dgfm.o submodular-ibfs.o submodular-functions.o 
	$(CXX) $(CXX_FLAGS) -o $@ denoise_test.o dgfm.o submodular-ibfs.o submodular-functions.o $(LIBS)	

submodular_test: submodular-functions.o submodular_test.o
	$(CXX) $(CXX_FLAGS) -o $@ submodular_test.o submodular-functions.o $(LIBS)

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
