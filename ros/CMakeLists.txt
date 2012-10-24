# -----------------------------------------------
# ROS NODE WRAPPER
# -----------------------------------------------

# if one Groovy
if (${catkin_VERSION} VERSION_GREATER "0.5.28")
find_package(catkin REQUIRED COMPONENTS catkin cv_bridge image_geometry image_transport roscpp message_filters)
include_directories(${catkin_INCLUDE_DIRS})
else()
# only Fuerte is supported otherwise
# manually get non-catkin packages
find_library(IMAGE_TRANSPORT_LIBS  image_transport PATHS /opt/ros/fuerte/stacks/image_common/image_transport/lib)
include_directories(/opt/ros/fuerte/stacks/image_common/image_transport/include)

find_library(IMAGE_GEOMETRY_LIBS image_geometry PATH /opt/ros/fuerte/stacks/vision_opencv/image_geometry/lib)
include_directories(/opt/ros/fuerte/stacks/vision_opencv/image_geometry/include)

find_library(CV_BRIDGE_LIBS cv_bridge PATH /opt/ros/fuerte/stacks/vision_opencv/cv_bridge/lib)
include_directories(/opt/ros/fuerte/stacks/vision_opencv/cv_bridge/include)

find_package(ROS REQUIRED COMPONENTS catkin cv_bridge image_geometry image_transport roscpp message_filters)
include_directories(${ROS_INCLUDE_DIRS})
endif()

# source --> build --> link --> install
set(SRC_FILES Node.cpp Messages.cpp)
add_executable(${PROJECT_NAME}_node ${SRC_FILES})
target_link_libraries(${PROJECT_NAME}_node  ${PROJECT_NAME}_lib 
                                            ${catkin_LIBRARIES} 
                                            ${ROS_LIBRARIES} 
                                            ${OpenCV_LIBS} 
                                            ${IMAGE_TRANSPORT_LIBS} 
                                            ${IMAGE_GEOMETRY_LIBS}
                                            ${CV_BRIDGE_LIBS}
)
install(TARGETS ${PROJECT_NAME}_node
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
)
