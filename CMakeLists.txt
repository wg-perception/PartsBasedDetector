project(PartsBasedDetector)
# PartsBasedDetector CMakeLists.txt
cmake_minimum_required(VERSION 2.8.5)

# -----------------------------------------------
# USER DEFINED VARIABLES
# -----------------------------------------------
option(BUILD_EXECUTABLE "Build as executable to test functionality"                     ON)
option(BUILD_DOC        "Build documentation with Doxygen"                              ON)
option(WITH_OPENMP      "Build with OpenMP support for multithreading"                  ON)
option(WITH_ECTO        "Build with ECTO bindings if building in a Catkin environment"  ON)
option(WITH_ROS         "Build with ROS bindings if building in a Catkin environment"   ON)

# -----------------------------------------------
# CATKIN
# -----------------------------------------------
find_package(catkin QUIET)
if (catkin_FOUND)
    set(WITH_CATKIN ON)
    set(BUILD_EXECUTABLE OFF)
    project(object_recognition_by_parts)
    # install targets for all things python
    if (WITH_ECTO)
        catkin_python_setup()
        find_package(ecto REQUIRED)
        find_package(ecto_pcl REQUIRED)
        find_package(object_recognition_core REQUIRED)
    endif()
    
    if (${catkin_VERSION} VERSION_GREATER "0.5.28")
        catkin_package(INCLUDE_DIRS include)
    else()
        catkin_stack()
        catkin_project(${PROJECT_NAME}
                       INCLUDE_DIRS include
    )
    endif()
    # deal with cvmatio
    find_package(cvmatio QUIET) 
    include_directories(SYSTEM ${cvmatio_INCLUDE_DIRS})
else()
    set(WITH_CATKIN OFF)
    set(WITH_ECTO   OFF)
    set(WITH_ROS    OFF)
    project(PartsBasedDetector)
    # find cvmatio manually
    include_directories(../cvmatio/include)
    find_library(cvmatio_LIBRARIES cvmatio PATHS ../cvmatio/lib)
endif()

# -----------------------------------------------
# COMPILER FLAGS
# -----------------------------------------------

# include all warning types
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-unused-parameter -pedantic")

# set the build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
        "Choose the type of build. Options are: None, Debug, Release, RelWithDebInfo, MinSizeRel"
    FORCE)
endif()

# add OpenMP support
if (WITH_OPENMP)
    find_package(OpenMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
endif()

# add vectorization support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -msse4.1")

# use highest level of optimization in Release mode
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O3")
endif ()

# -----------------------------------------------
# DEPENDENCIES
# -----------------------------------------------
# find the dependencies
include(cmake/FindEigen.cmake)
find_package(Boost COMPONENTS system filesystem signals REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})
add_definitions(-DEIGEN_USE_NEW_STDVECTOR
                -DEIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET)

# include the dependencies
include_directories(SYSTEM  ${OPENCV_INCLUDE_DIRS}
                            ${Boost_INCLUDE_DIRS}
)
include_directories(include)

# check if the MatlabIO dependency was successfully found
if (cvmatio_LIBRARIES)
    set(cvmatio_FOUND ON)
else()
    set(cvmatio_FOUND OFF)
endif()


# -----------------------------------------------
# SUBDIRECTORIES
# -----------------------------------------------
# add the main executable/library
add_subdirectory(src)

# add documentation
if(BUILD_DOC)
    find_package(Doxygen)
    configure_file(${PROJECT_SOURCE_DIR}/doc/Doxyfile.in ${PROJECT_SOURCE_DIR}/doc/Doxyfile @ONLY)
    add_custom_target(${PROJECT_NAME}_doc ALL ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/doc/Doxyfile)
endif()

# add tests
if(WITH_ECTO AND BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()

# add ecto cells
if (WITH_ECTO AND ${ecto_FOUND} AND ${object_recognition_core_FOUND})
    add_subdirectory(cells)
    add_subdirectory(python)
endif()

# add ROS bindings if need be
# does not build yet due to missing dependencies
#set(WITH_ROS OFF)
if (WITH_ROS)
    find_package(PCL REQUIRED)
    include_directories(${PCL_INCLUDE_DIRS})
    add_subdirectory(ros)
endif()


# -----------------------------------------------
# BUILD SUMARY
# -----------------------------------------------
message("")
message("---------------------------------------------")
message(${PROJECT_NAME} " Configuration Summary")
message("")
message("Build type:                    ${CMAKE_BUILD_TYPE}")
message("Building with Catkin:          ${WITH_CATKIN}")
message("Building with ECTO bindings:   ${WITH_ECTO}")
message("Building with ROS bindings:    ${WITH_ROS}")
message("Build with cvmatio bindings:   ${cvmatio_FOUND}")
message("Build with threading (OpenMP): ${WITH_OPENMP}")
message("Build as executable:           ${BUILD_EXECUTABLE}")
message("Build with documentation:      ${BUILD_DOC}")
message("---------------------------------------------")
message("")
