CXX = mpic++

CXXFLAGS = -O2 -std=c++11

TARGET = matmul

SRCS = main.cc matmul.cc
OBJS = main.o matmul.o

# 預設目標
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

main.o: main.cc
	$(CXX) $(CXXFLAGS) -c main.cc

matmul.o: matmul.cc
	$(CXX) $(CXXFLAGS) -c matmul.cc

clean:
	rm -f $(OBJS) $(TARGET)
