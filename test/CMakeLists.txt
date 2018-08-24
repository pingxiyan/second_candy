cmake_minimum_required(VERSION 3.0)

set(TARGET_NAME testapp)

PROJECT(${TARGET_NAME})



MESSAGE(STATUS "====================================")

MESSAGE(STATUS "Start cmake " ${PROJECT_NAME})

MESSAGE(STATUS "operation system is ${CMAKE_SYSTEM_NAME}") 


set (CMAKE_CXX_STANDARD 11) 



find_package( OpenCV )



include_directories(${OpenCL_INCLUDE_DIR} ./)

link_directories(./)



ADD_EXECUTABLE(${TARGET_NAME} main.cpp )

target_link_libraries(${TARGET_NAME} 
	${OpenCV_LIBS})

