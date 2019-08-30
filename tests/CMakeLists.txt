add_executable(test_nap_comm test_nap_comm.cpp)
target_link_libraries(test_nap_comm streamline ${MPI_LIBRARIES} googletest pthread )
add_test(RandomCommTest mpirun -n 16 ./test_nap_comm)

add_executable(test_nap_coll test_nap_coll.cpp)
target_link_libraries(test_nap_coll streamline ${MPI_LIBRARIES} googletest pthread )
add_test(RandomCollTest mpirun -n 16 ./test_nap_coll)
