cmake_minimum_required(VERSION 3.25)
message ("Inside top cmake file, ${CMAKE_CURRENT_BINARY_DIR}")

project(Func_Sim)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./build)

if(DEFINED N_STO_INT_RES)
  add_definitions(-DN_STO_INT_RES=${N_STO_INT_RES})
else()
  add_definitions(-DN_STO_INT_RES=20)
endif()

if(DEFINED N_STO_PARAMS)
  add_definitions(-DN_STO_PARAMS=${N_STO_PARAMS})
else()
  add_definitions(-DN_STO_PARAMS=20)
endif()

file(GLOB FILES "src/*.cpp")

add_executable(Func_Sim ${FILES})

set(HIGHFIVE_FIND_HDF5 Off)
add_subdirectory(HighFive)
add_subdirectory(rapidcsv)

target_include_directories(Func_Sim PRIVATE HLS_Arb_Precision_Types/include)

include_directories(include)

target_include_directories(Func_Sim PRIVATE)
target_compile_options(Func_Sim PRIVATE -W -g -O3)
target_link_libraries(Func_Sim HighFive)
set_target_properties(Func_Sim PROPERTIES CMAKE_RUNTIME_OUTPUT_DIRECTORY ./build)
