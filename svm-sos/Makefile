# Makefile for sos-SVM-struct API, 03.10.06

#Call 'make' using the following line to make CYGWIN produce stand-alone Windows executables
#		make 'SFLAGS=-mno-cygwin'

#Use the following to compile under unix or cygwin
CC ?= gcc
CXX ?= g++
LD = $(CXX)

OPT ?= -g
CFLAGS =   $(SFLAGS) $(OPT) -fomit-frame-pointer -ffast-math -Wall -std=c++11 -I../submodular-flow/ -I../higher-order-energy/qpbo/ -I../higher-order-energy/include/
LDFLAGS =  $(SFLAGS) $(OPT) -lm -Wall -lopencv_core -lopencv_highgui -lopencv_imgproc -L../lib -lsubmodular-flow -lboost_serialization -lboost_program_options
#CFLAGS =  $(SFLAGS) -pg -Wall
#LDFLAGS = $(SFLAGS) -pg -lm -Wall 

all: svm_sos_learn svm_sos_classify

.PHONY: clean
clean: svm_light_clean svm_struct_clean
	rm -f *.o *.tcov *.d core gmon.out *.stackdump 
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

svm_sos_classify: svm_light_hideo_noexe svm_struct_noexe svm_struct_api.o svm_struct/svm_struct_classify.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o svm_c++.o gmm.o svm_struct_options.o lib 
	$(LD) svm_struct_api.o svm_struct/svm_struct_classify.o svm_light/svm_common.o svm_struct/svm_struct_common.o svm_c++.o gmm.o svm_struct_options.o -o svm_sos_classify $(LIBS) $(LDFLAGS)

svm_sos_learn: svm_light_hideo_noexe svm_struct_noexe svm_struct_api.o svm_struct_learn_custom.o svm_struct/svm_struct_learn.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o svm_c++.o gmm.o svm_struct_options.o lib
	$(LD) svm_struct/svm_struct_learn.o svm_struct_learn_custom.o svm_struct_api.o svm_light/svm_hideo.o svm_light/svm_learn.o svm_light/svm_common.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o svm_c++.o gmm.o svm_struct_options.o -o svm_sos_learn $(LIBS) $(LDFLAGS)


svm_struct_api.o: svm_struct_api.cpp svm_struct_api.h svm_struct_api_types.h svm_struct/svm_struct_common.h image_manip.hpp
	$(CXX) -c $(CFLAGS) svm_struct_api.cpp -o svm_struct_api.o

svm_struct_learn_custom.o: svm_struct_learn_custom.cpp svm_struct_api.h svm_light/svm_common.h svm_struct_api_types.h svm_struct/svm_struct_common.h
	$(CXX) -c $(CFLAGS) svm_struct_learn_custom.cpp -o svm_struct_learn_custom.o

svm_c++.o: svm_c++.cpp svm_c++.hpp image_manip.hpp gmm.o
	$(CXX) -c $(CFLAGS) svm_c++.cpp -o svm_c++.o

gmm.o: gmm.cpp gmm.hpp
	$(CXX) -c $(CFLAGS) gmm.cpp -o gmm.o

svm_struct_options.o: svm_struct_options.cpp svm_struct_options.hpp svm_struct_api_types.h
	$(CXX) -c $(CFLAGS) svm_struct_options.cpp -o svm_struct_options.o

lib:
	cd ..; make
