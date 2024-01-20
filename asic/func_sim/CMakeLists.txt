cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(func_sim)
file(GLOB FILES "*.cpp")
add_executable(func_sim ${FILES})
target_compile_options(func_sim PRIVATE -Wall -g)
