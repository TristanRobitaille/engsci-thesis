cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(func_sim)
file(GLOB FILES "src/*.cpp")
add_executable(func_sim ${FILES})
add_subdirectory(HighFive)
add_subdirectory(rapidcsv)
include_directories(include)
target_include_directories(func_sim PRIVATE)
target_compile_options(func_sim PRIVATE -Wall -g)
target_link_libraries(func_sim HighFive)
