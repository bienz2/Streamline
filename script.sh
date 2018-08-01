#/bin/bash

make clean
make
mpirun -n 16 valgrind ./test_nap_comm
