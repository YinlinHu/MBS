CCPP = g++

SSE_FLAG = -march=native -msse2 -DWITH_SSE

CFLAGS = -w -O3 $(SSE_FLAG) -Iinclude
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

SOURCES_CPP := $(shell find . -name '*.cpp')
OBJ := $(SOURCES_CPP:%.cpp=%.o)
HEADERS := $(shell find . -name '*.h')

all: MBS

.cpp.o:  %.cpp %.h
	$(CCPP) -c -o $@ $(CFLAGS) $(LDFLAGS) $+

MBS: $(HEADERS) $(OBJ)
	$(CCPP) -o $@ $(OBJ) $(LDFLAGS)

clean:
	rm -f $(OBJ) MBS

