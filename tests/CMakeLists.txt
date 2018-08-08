add_executable(test_nap_comm test_nap_comm.cpp)
target_link_libraries(test_nap_comm ${MPI_LIBRARIES} googletest pthread )
add_test(RandomCommTest mpirun -n 16 ./test_nap_comm)
