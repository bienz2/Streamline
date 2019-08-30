include_directories(.)

add_subdirectory(allreduce)

add_library(streamline STATIC 
    ${allreduce_HEADERS} 
    ${allreduce_SOURCES})

target_link_libraries(streamline ${MPI_LIBRARIES} )

include_directories(".")
