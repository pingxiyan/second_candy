cmake_minimum_required(VERSION 3.0)

set(TARGET_NAME second_candy)

if(NOT WIN32)
string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(Red         "${Esc}[31m")
set(Green       "${Esc}[32m")
set(Yellow      "${Esc}[33m")
set(Blue        "${Esc}[34m")
endif ()

MESSAGE(STATUS "${Blue}====================================${ColourReset}")
MESSAGE(STATUS "Start ${TARGET_NAME}")

project(${TARGET_NAME})

####################################
## to use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
####################################

add_subdirectory(test)


