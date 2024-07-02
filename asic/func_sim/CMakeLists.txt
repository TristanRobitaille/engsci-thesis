cmake_minimum_required(VERSION 3.25)
message ("Inside top cmake file, ${CMAKE_CURRENT_BINARY_DIR}")

project(Func_Sim)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(DEFINED N_STO)
  add_definitions(-DN_STO=${N_STO})
else()
  add_definitions(-DN_STO=20)
endif()

option(BOOSTINSTALLED "Boost library location" "")
if(DEFINED BOOSTINSTALLED)
  message (STATUS "Using local boost at location ${BOOSTINSTALLED}")
  set(BOOST_ROOT ${BOOSTINSTALLED} CACHE PATH "Boost library path (hint)" )
  set(Boost_NO_SYSTEM_PATHS on CACHE BOOL "Do not search system for Boost" )
  set(Boost_USE_STATIC_LIBS ON)
  set(Boost_USE_MULTITHREADED ON)
  set(Boost_USE_STATIC_RUNTIME OFF)
  find_package(Boost 1.82.0 REQUIRED system filesystem)
endif()

file(GLOB FILES "src/*.cpp")

add_executable(Func_Sim ${FILES})

set(HIGHFIVE_FIND_HDF5 Off)
add_subdirectory(HighFive)
add_subdirectory(rapidcsv)

target_include_directories(Func_Sim PRIVATE HLS_Arb_Precision_Types/include)

include_directories(include)

target_include_directories(Func_Sim PRIVATE)
target_compile_options(Func_Sim PRIVATE -Wall -g)
target_link_libraries(Func_Sim HighFive)