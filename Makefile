MPICXX=mpicxx
FLAGS=-std=c++11

all: test_nap_comm

test_nap_comm: test_nap_comm.cpp
	${MPICXX} ${FLAGS} test_nap_comm.cpp -o test_nap_comm

clean:
	rm test_nap_comm
