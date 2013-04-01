# Makefile for sos-SVM-struct API, 03.10.06

#Call 'make' using the following line to make CYGWIN produce stand-alone Windows executables
#		make 'SFLAGS=-mno-cygwin'

#Use the following to compile under unix or cygwin
CC ?= gcc
CXX ?= g++
LD = $(CXX)

OPT ?= -g
CFLAGS =   $(SFLAGS) $(OPT) -ffast-math -Wall -std=c++11 -I../submodular-flow/ -I../higher-order-energy/qpbo/ -I../higher-order-energy/include/
LDFLAGS =  $(SFLAGS) $(OPT)  -L../lib 
LIBS = -lm -Wall -lopencv_core -lopencv_highgui -lopencv_imgproc -lsubmodular-flow -lboost_serialization -lboost_program_options
#CFLAGS =  $(SFLAGS) -pg -Wall
#LDFLAGS = $(SFLAGS) -pg -lm -Wall 

SVM_STRUCT_OBJS = svm_struct/svm_struct_learn.o \
			      svm_struct/svm_struct_common.o
SVM_LIGHT_OBJS = svm_light/svm_hideo.o \
				 svm_light/svm_learn.o \
				 svm_light/svm_common.o
SRCS = svm_struct_learn_custom.cpp \
	   svm_struct_api.cpp \
	   svm_c++.cpp \
	   interactive_seg_app.cpp \
	   gmm.cpp \
	   svm_struct_options.cpp \
	   stats.cpp
OBJS = $(SRCS:.cpp=.o)
LEARN_OBJS = $(SVM_STRUCT_OBJS) $(SVM_LIGHT_OBJS) $(OBJS) svm_struct/svm_struct_main.o
CLASSIFY_OBJS = $(SVM_STRUCT_OBJS) $(SVM_LIGHT_OBJS) $(OBJS) svm_struct/svm_struct_classify.o
PROCESS_STATS_OBJS = $(SVM_STRUCT_OBJS) $(SVM_LIGHT_OBJS) $(OBJS) process_stats.o

.PHONY:
all: svm_sos_learn svm_sos_classify process_stats

.PHONY: clean
clean: svm_light_clean svm_struct_clean
	rm -f *.o *.tcov *.d core gmon.out *.stackdump *.P
	cd ..; make clean

#-----------------------#
#----   SVM-light   ----#
#-----------------------#
svm_light_hideo_noexe: 
	cd svm_light; make svm_learn_hideo_noexe

svm_light_clean: 
	cd svm_light; make clean

#----------------------#
#----  STRUCT SVM  ----#
#----------------------#

svm_struct_noexe: 
	cd svm_struct; make svm_struct_noexe

svm_struct_clean: 
	cd svm_struct; make clean


#-------------------------#
#----  SVM sos API  ----#
#-------------------------#

svm_sos_classify: svm_light_hideo_noexe svm_struct_noexe $(OBJS) lib
	$(LD) $(LDFLAGS) $(CLASSIFY_OBJS) -o svm_sos_classify $(LIBS)

svm_sos_learn: svm_light_hideo_noexe svm_struct_noexe $(OBJS) lib
	$(LD) $(LDFLAGS) $(LEARN_OBJS) -o svm_sos_learn $(LIBS)

process_stats: $(OBJS) process_stats.o
	$(LD) $(LDFLAGS) $(PROCESS_STATS_OBJS) -o process_stats $(LIBS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -MMD -o $@ -c $<
	@cp $*.d $*.P; \
	 sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
    	 -e '/^$$/ d' -e 's/$$/ :/' < $*.d >> $*.P; \
	 rm $*.d

-include $(SRCS:.cpp=.P)

lib:
	cd ..; make
