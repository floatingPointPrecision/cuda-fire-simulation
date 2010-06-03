#
# CS348B todo: Please set the following to point to the base directory of your pbrt installation
#
PBRT_BASEDIR=/home/yulou/pbrt-1.04

ARCH = $(shell uname)

LEXLIB = -lfl
DLLLIB = -ldl
ifeq ($(ARCH),Darwin)
  DLLLIB =
endif
ifeq ($(ARCH),OpenBSD)
  DLLLIB =
endif

EXRINCLUDEDIR=-I/usr/include/OpenEXR
EXRLIBDIR=-L/usr/lib

EXRLIBS=-lIlmImf -lImath -lIex -lIlmThread -lHalf -lz
ifeq ($(ARCH),Linux)
  EXRLIBS += -lpthread
endif

CC=gcc
CXX=g++
LD=$(CXX) $(OPT)
OPT=-O2
# OPT=-O2 -msse -mfpmath=sse
INCLUDE=-I. -I$(PBRT_BASEDIR)/core -I$(EXRINCLUDEDIR)
WARN=-Wall
CWD=$(shell pwd)
CXXFLAGS=$(OPT) $(INCLUDE) $(WARN) -fPIC
CCFLAGS=$(CXXFLAGS)
LIBS=$(LEXLIB) $(DLLLIB) -L$(EXRLIBDIR) $(EXRLIBS) -lm 

SHARED_LDFLAGS = -shared
LRT_LDFLAGS=-rdynamic $(OPT)

ifeq ($(ARCH), Darwin)
  OS_VERSION = $(shell uname -r)
  SHARED_LDFLAGS = -flat_namespace -undefined suppress -bundle -noprebind
  LRT_LDFLAGS=$(OPT) # -L/sw/lib
  INCLUDE += -I/sw/include
  WARN += -Wno-long-double
endif



FILES     = fire
OBJS      = $(FILES:=.o)
TARGET    = fire.so

OTHER_HEADERS := fire.h


.SUFFIXES: .cpp .o
.PHONY: clean
.SECONDARY: $(OBJS)


CORE_HEADERFILES = api.h camera.h color.h dynload.h film.h geometry.h \
                  kdtree.h light.h pbrt.h material.h mc.h mipmap.h octree.h \
                  paramset.h primitive.h reflection.h sampling.h scene.h \
                  shape.h texture.h timer.h tonemap.h transform.h transport.h \
                  volume.h 

CORE_HEADERS := $(addprefix $(PBRT_BASEDIR)/core/, $(CORE_HEADERFILES) )

default: $(TARGET)

%.so: $(OBJS) 
	$(LD) $(SHARED_LDFLAGS) $(OBJS) -o $@
	@mv $(TARGET) $(PBRT_BASEDIR)/bin

%.o: %.cpp $(CORE_HEADERS) $(OTHER_HEADERS)
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	rm -rf *.o $(TARGET)

