cmake_minimum_required(VERSION 3.16)

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

AUX_SOURCE_DIRECTORY(include SRC_FILES)

# Source files
file (GLOB SOURCE_FILES "src/Algebra/*.cpp" "src/Algorithms/*.cpp" "src/Stats/*.cpp")
file (GLOB HEADER_FILES "include/Algebra/*.h" "include/Algorithms/*.h" "include/Stats/*.h")

# Set up such that XCode organizes the files correctly
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES} ${HEADER_FILES})

# Required library
find_library(ARMADILLO_LIB armadillo HINTS /usr/local/share/ REQUIRED)

# Add library
add_library(algorithms SHARED ${SOURCE_FILES} ${HEADER_FILES})

# Link
target_link_libraries(algorithms PUBLIC ${ARMADILLO_LIB})

# Include directories
target_include_directories(algorithms PRIVATE include/Algorithms)
target_include_directories(algebra PRIVATE  include/Algebra)
target_include_directories(stats PRIVATE include/Stats)

# Install
install(TARGETS algorithms DESTINATION lib)

# Create base directory
install(DIRECTORY include/Algorithms DESTINATION include/Algorithms)
install(DIRECTORY include/Algebra DESTINATION include/Algebra)
install(DIRECTORY include/Stats DESTINATION include/Stats)
