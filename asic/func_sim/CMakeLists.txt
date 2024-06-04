cmake_minimum_required(VERSION 3.25)
message ("Inside top cmake file, ${CMAKE_CURRENT_BINARY_DIR}")

project(Func_Sim)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# option(BOOSTINSTALLED "Boost library location" "")
# if(NOT "${BOOSTINSTALLED}" STREQUAL "") 
#     set(BOOST_ROOT ${BOOSTINSTALLED} CACHE PATH "Boost library path (hint)" )
#     set(Boost_NO_SYSTEM_PATHS on CACHE BOOL "Do not search system for Boost" )
#     set(Boost_USE_STATIC_LIBS ON)
#     set(Boost_USE_MULTITHREADED ON)
#     set(Boost_USE_STATIC_RUNTIME OFF)
#     find_package(Boost 1.84.0 REQUIRED system filesystem)
# endif()

file(GLOB FILES "src/*.cpp")

add_executable(Func_Sim ${FILES})

set(HIGHFIVE_FIND_HDF5 Off)
add_subdirectory(HighFive)
add_subdirectory(rapidcsv)
include_directories(HLS_Arb_Precision_Types/include)

find_package(HDF5 REQUIRED)

include_directories(include)

target_include_directories(Func_Sim PRIVATE)
target_compile_options(Func_Sim PRIVATE -Wall -g)
target_link_libraries(Func_Sim HighFive HDF5::HDF5)
