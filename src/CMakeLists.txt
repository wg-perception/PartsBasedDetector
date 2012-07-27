set(SRC_FILES   DepthConsistency.cpp 
                DynamicProgram.cpp
                HOGFeatures.cpp 
                Model.cpp
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

#link
#find_library(opencv_imgproc_LIBRARIES NAMES opencv_imgproc PATHS /u/hbristow/dev/lib NO_DEFAULT_PATH)
#find_library(opencv_core_LIBRARIES NAMES opencv_core PATHS /u/hbristow/dev/lib NO_DEFAULT_PATH)
#message(info "******************${opencv_imgproc_LIBRARIES} ${opencv_core_LIBRARIES}")


target_link_libraries(${PROJECT_NAME}   ${Boost_LIBRARIES} 
                                        ${OpenCV_LIBS} 
                                        ${ZLIB_LIBRARIES} 
                                        ${MatlabIO_LIBRARIES}
)

#install
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
                                LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib
)
