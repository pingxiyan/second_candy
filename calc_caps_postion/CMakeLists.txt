cmake_minimum_required(VERSION 3.0)

set(TARGET_NAME calc_caps_postion)
project(${TARGET_NAME})

####################################
## to use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
####################################

find_package( OpenCV )
include_directories(${OpenCV_INCLUDE_DIR})

ADD_EXECUTABLE(${TARGET_NAME} main.cpp )
target_link_libraries(${TARGET_NAME} 
	${OpenCV_LIBS})