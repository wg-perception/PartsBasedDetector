# PartsBasedDetector CMakeLists.txt
cmake_minimum_required(VERSION 2.8)

# -----------------------------------------------
# USER-DEFINED VARIABLES
option(BUILD_EXECUTABLE "Build as executable to test functionality"     OFF)
option(BUILD_DOC        "Build documentation with Doxygen"              ON)
option(USE_OPENMP       "Build with OpenMP support for multithreading"  ON)

# --------------------
# CATKIN
if (catkin_FOUND)
    project(object_recognition_by_parts)
    catkin_stack()
    catkin_project(object_recognition_by_parts
                   DEPENDS Boost OpenCV cvmatio
                   INCLUDE_DIRS include
    )
    #install targets for all things python
    catkin_python_setup()
    find_package(ecto REQUIRED)
    find_package(object_recognition_core REQUIRED)
else()
    project(PartsBasedDetector)
    # find cvmatio manually
    include_directories(../cvmatio/include)
    find_library(MatlabIO_LIBRARIES MatlabIO PATHS ../cvmatio/lib)
endif()


# --------------------
# COMMON
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING
        "Choose the type of build. Options are: None, Debug, Release, RelWithDebInfo, MinSizeRel"
    FORCE)
endif()

# find the dependencies
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

# add OpenMP support
if (USE_OPENMP)
    find_package(OpenMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
endif()

# include the dependencies
include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories("include")

# GLOB the source files
set(SRC_FILES   src/DepthConsistency.cpp src/DynamicProgram.cpp
                src/HOGFeatures.cpp src/Model.cpp src/Part.cpp
                src/PartsBasedDetector.cpp src/StereoCameraModel.cpp
                src/Visualize.cpp
)

# build as a standalone executable
if (BUILD_EXECUTABLE)
    set(SRC_FILES ${SRC_FILES} src/main.cpp)
    add_executable(${PROJECT_NAME} ${SRC_FILES})
else()
    add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
endif()

#link
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${ZLIB_LIBRARIES} ${MatlabIO_LIBRARIES})

# build the doxygen pdf
if(BUILD_DOC)
    find_package(Doxygen)
    configure_file(${PROJECT_SOURCE_DIR}/doc/Doxyfile.in ${PROJECT_SOURCE_DIR}/doc/Doxyfile @ONLY)
    add_custom_target(doc ALL ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/doc/Doxyfile)
endif()
