# -----------------------------------------------
# ROS NODE WRAPPER
# -----------------------------------------------

find_package(PCL)
find_package(catkin REQUIRED COMPONENTS catkin rosconsole cv_bridge image_geometry image_transport roscpp message_filters pcl_ros)
include_directories(${catkin_INCLUDE_DIRS})

# source --> build --> link --> install
set(SRC_FILES Node.cpp Messages.cpp)
add_executable(${PROJECT_NAME}_node ${SRC_FILES})
target_link_libraries(${PROJECT_NAME}_node  ${PROJECT_NAME}_lib 
                                            ${catkin_LIBRARIES} 
                                            ${OpenCV_LIBS} 
)
install(TARGETS ${PROJECT_NAME}_node
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
)
