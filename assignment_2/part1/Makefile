# Compiler and flags
CXX = g++
CXXFLAGS = -O3 -std=c++17 -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -mavx -mavx2

# Source file
SRC = pi.cpp

# Target file
TARGET = pi.out

# Compile the program
all: $(TARGET)

# Create the executable
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Clean rule to remove the executable
clean:
	rm -f $(TARGET)
