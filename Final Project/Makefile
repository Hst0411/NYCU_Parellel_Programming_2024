CXX = mpicxx
CXXFLAGS = -std=c++11 -O2 -Wall

main:
	$(CXX) $(CXXFLAGS) -o main -I include MPI_fptree.cpp MPI_main.cpp

clean:
	rm -rf main
