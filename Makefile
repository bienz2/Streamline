test: test_nap_comm.cpp
	${MPICXX} test_nap_comm.cpp -o test_nap_comm

clean:
	rm test_nap_comm
