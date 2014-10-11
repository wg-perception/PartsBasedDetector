# -----------------------------------------------
# BUILD THE PARTS BASED DETECTOR FROM SOURCE
# -----------------------------------------------
set(SRC_FILES   DepthConsistency.cpp 
                DynamicProgram.cpp
                FileStorageModel.cpp
                HOGFeatures.cpp 
                SpatialConvolutionEngine.cpp
                FourierConvolutionEngine.cpp
                PartsBasedDetector.cpp 
                SearchSpacePruning.cpp
                StereoCameraModel.cpp
                Visualize.cpp
                nms.cpp
)

set(LIBS        ${Boost_LIBRARIES}
                ${OpenCV_LIBS}
)

# with cvmatio support
if (WITH_CVMATIO)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_MATLABIO")
    set(SRC_FILES ${SRC_FILES} MatlabIOModel.cpp)
    set(LIBS ${LIBS} ${ZLIB_LIBRARIES} ${cvmatio_LIBRARIES})
    add_executable(ModelTransfer ModelTransfer.cpp FileStorageModel.cpp MatlabIOModel.cpp)
    target_link_libraries(ModelTransfer ${LIBS})
    install(TARGETS ModelTransfer
            RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
    )
endif()

# as a library (always)
add_library(${PROJECT_NAME}_lib SHARED ${SRC_FILES})
target_link_libraries(${PROJECT_NAME}_lib ${LIBS})
set_target_properties(${PROJECT_NAME}_lib PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME}_lib 
        DESTINATION ${PROJECT_SOURCE_DIR}/lib
)

# as an executable
if (BUILD_EXECUTABLE)
    set(SRC_FILES demo.cpp)
    add_executable(${PROJECT_NAME}_bin ${SRC_FILES})
    target_link_libraries(${PROJECT_NAME}_bin ${LIBS} ${PROJECT_NAME}_lib)
    set_target_properties(${PROJECT_NAME}_bin PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
    install(TARGETS ${PROJECT_NAME}_bin
            RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
    )
endif()
