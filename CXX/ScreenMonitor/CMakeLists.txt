string(REGEX REPLACE ".*/\(.*\)" "\\1" NAME ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCV)
aux_source_directory(. SRC_LIST)
add_executable(${NAME}_demo demo.cpp ${SRC_LIST})
add_library(${NAME} ${SRC_LIST})

