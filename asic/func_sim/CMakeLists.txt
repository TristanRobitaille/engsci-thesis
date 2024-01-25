cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Turn off Boost support
if(NOT DEFINED HIGHFIVE_USE_BOOST)
  set(HIGHFIVE_USE_BOOST Off)
endif()

project(func_sim)
file(GLOB FILES "src/*.cpp")
add_executable(func_sim ${FILES})
add_subdirectory(HighFive)
include_directories(include)
target_include_directories(func_sim PRIVATE)
target_compile_options(func_sim PRIVATE -Wall -g)
target_link_libraries(func_sim HighFive)

