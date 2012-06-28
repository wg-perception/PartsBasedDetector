# MatlabIO CMakeLists.txt
cmake_minimum_required(VERSION 2.8)
project(PartsBasedDetector)
set(PROJECT_ROOT ${PROJECT_SOURCE_DIR})

# find the dependencies
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

# add Doxygen support
find_package(Doxygen)
configure_file(${PROJECT_SOURCE_DIR}/doc/Doxyfile.in ${PROJECT_SOURCE_DIR}/doc/Doxyfile @ONLY)
add_custom_target(doc ALL ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/doc/Doxyfile)

# add OpenMP support
find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")


# include the dependencies
include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories("include")

# GLOB the source files
file(GLOB SRC_FILES "src/*.cpp")
file(GLOB HDR_FILES "include/*.hpp")

# build as an executable
add_executable(${PROJECT_NAME} ${SRC_FILES})

#link
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${ZLIB_LIBRARIES})

# build the doxygen pdf
add_custom_command(COMMAND make 
    OUTPUT ${PROJECT_SOURCE_DIR}/doc/latex
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/doc/latex
)

# install the targets
install(FILES ${PROJECT_SOURCE_DIR}/build/${PROJECT_NAME} DESTINATION ${PROJECT_SOURCE_DIR}/bin)
install(FILES ${PROJECT_SOURCE_DIR}/doc/latex/refman.pdf DESTINATION ${PROJECT_SOURCE_DIR}/doc/latex/MatlabIO.pdf)
