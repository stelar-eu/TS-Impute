cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(algorithms)

# Include dir
include_directories(/usr/local/include)

# Src
AUX_SOURCE_DIRECTORY(src SRC_FILES)

# Set Headers and src for both algorithms, algebra
set(ALGEBRA_SOURCE_DIR "src/Algebra")
set(ALGEBRA_INCLUDE_DIR "include/Algebra")

set(PROJECT_SOURCE_DIR "src/Algorithms")
set(PROJECT_INCLUDE_DIR "include/Algorithms")

# Source files
set(SOURCE_FILES
    ${PROJECT_INCLUDE_DIR}/IterativeSVD.h
    ${PROJECT_SOURCE_DIR}/IterativeSVD.cpp
)

set(ALGEBRA_FILES
  ${ALGEBRA_INCLUDE_DIR}/RSVD.h
  ${ALGEBRA_SOURCE_DIR}/RSVD.cpp
)

# Set up such that XCode organizes the files correctly
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES} ${ALGEBRA_FILES})

# Required library
find_library(ARMADILLO_LIB armadillo HINTS /usr/local/lib/ REQUIRED)

# Add library
add_library(algorithms SHARED ${SOURCE_FILES} ${ALGEBRA_FILES})

# Link
target_link_libraries(algorithms PUBLIC ${ARMADILLO_LIB})

# Include directories
target_include_directories(algorithms PRIVATE include/Algorithms)
target_include_directories(algebra PRIVATE include/Algebra)

# Install
install(TARGETS algorithms DESTINATION lib)

# Install the headers
install(FILES include/algorithms DESTINATION include/Algorithms)
install(FILES include/algebra DESTINATION include/Algebra)

# Create base directory
install(DIRECTORY include/Algorithms DESTINATION include/Algorithms)
install(DIRECTORY include/Algebra DESTINATION include/Algebra)