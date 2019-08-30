# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
set(allreduce_SOURCES 
    allreduce/nap_allreduce.cpp
    PARENT_SCOPE
    )
set(allreduce_HEADERS
    allreduce/nap_allreduce.hpp
    PARENT_SCOPE
    )

