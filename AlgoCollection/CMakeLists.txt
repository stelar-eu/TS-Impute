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

include_directories(/usr/local/include)
include_directories("${CMAKE_SOURCE_DIR}/cpp/include/Algebra")
include_directories("${CMAKE_SOURCE_DIR}/cpp/include/Stats")
include_directories("${CMAKE_SOURCE_DIR}/cpp/include/Algorithms")
include_directories("${CMAKE_SOURCE_DIR}/python")
include_directories("${CMAKE_SOURCE_DIR}/python/carma")

file (GLOB SOURCE_FILES "cpp/src/Algebra/*.cpp" "cpp/src/Algorithms/*.cpp" "cpp/src/Stats/*.cpp")
file (GLOB HEADER_FILES "cpp/include/Algebra/*.h" "cpp/include/Algorithms/*.h" "cpp/include/Stats/*.h")
file (GLOB PYTHON_FILES "python/*.cpp" "python/*.h" "python/carma/*.h" "python/carma/carma/*.h")

# Set up such that XCode organizes the files
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES} ${HEADER_FILES} ${PYTHON_FILES} )

# Armadillo
find_library(ARMADILLO_LIB armadillo HINTS /usr/local/lib/ REQUIRED)

# mlpack
find_library(MLPACK_LIB mlpack HINTS /usr/local/lib/ REQUIRED)

# Pybind11
find_package(pybind11 REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED)

pybind11_add_module(algorithms 
	${SOURCE_FILES}
	${HEADER_FILES}
	${PYTHON_FILES}
)

target_link_libraries(algorithms PUBLIC ${MLPACK_LIB})
target_link_libraries(algorithms PUBLIC ${ARMADILLO_LIB})
target_link_libraries(algorithms PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS algorithms
  COMPONENT python3
  LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}"
  )
