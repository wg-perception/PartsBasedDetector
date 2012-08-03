# ----------------------------------------------------------------------------
# With MatlabIO support
# ----------------------------------------------------------------------------
if (MatlabIO_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_MATLABIO")
    set(SRC_FILES   DepthConsistency.cpp 
                    DynamicProgram.cpp
                    FileStorageModel.cpp
                    HOGFeatures.cpp 
                    MatlabIOModel.cpp
                    nms.cpp
                    PartsBasedDetector.cpp 
                    SearchSpacePruning.cpp
                    StereoCameraModel.cpp
                    Visualize.cpp
    )

    # build as a standalone executable
    if (BUILD_EXECUTABLE)
        set(SRC_FILES ${SRC_FILES} main.cpp)
        add_executable(${PROJECT_NAME} ${SRC_FILES})
    else()
        add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
    endif()

    add_executable(ModelTransfer ModelTransfer.cpp FileStorageModel.cpp MatlabIOModel.cpp)
    target_link_libraries(ModelTransfer ${Boost_LIBRARIES}
                                        ${OpenCV_LIBS} 
                                        ${ZLIB_LIBRARIES} 
                                        ${MatlabIO_LIBRARIES}
    )

    target_link_libraries(${PROJECT_NAME}   ${Boost_LIBRARIES} 
                                            ${OpenCV_LIBS} 
                                            ${ZLIB_LIBRARIES} 
                                            ${MatlabIO_LIBRARIES}
    )

    #install
    install(TARGETS ${PROJECT_NAME} ModelTransfer   RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
                                                    LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib
    )

# ----------------------------------------------------------------------------
# Without MatlabIO support
# ----------------------------------------------------------------------------
else()
    set(SRC_FILES   DepthConsistency.cpp 
                    DynamicProgram.cpp
                    FileStorageModel.cpp
                    HOGFeatures.cpp 
                    PartsBasedDetector.cpp 
                    StereoCameraModel.cpp
                    Visualize.cpp
                    nms.cpp
    )

    # build as a standalone executable
    if (BUILD_EXECUTABLE)
        set(SRC_FILES ${SRC_FILES} main.cpp)
        add_executable(${PROJECT_NAME} ${SRC_FILES})
    else()
        add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
    endif()

    target_link_libraries(${PROJECT_NAME}   ${Boost_LIBRARIES} 
                                            ${OpenCV_LIBS} 
    )

    #install
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
                                    LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib

    )
endif()
