# -----------------------------------------------
# ROS NODE WRAPPER
# -----------------------------------------------

# automagically find the required packages
find_package(ROS REQUIRED COMPONENTS catkin roscpp message_filters)
include_directories(${ROS_INCLUDE_DIRS})

# manually get non-catkin packages
find_library(IMAGE_TRANSPORT_LIBS  image_transport PATHS /opt/ros/fuerte/stacks/image_common/image_transport/lib)
include_directories(/opt/ros/fuerte/stacks/image_common/image_transport/include)

find_library(IMAGE_GEOMETRY_LIBS image_geometry PATH /opt/ros/fuerte/stacks/vision_opencv/image_geometry/lib)
include_directories(/opt/ros/fuerte/stacks/vision_opencv/image_geometry/include)

find_library(CV_BRIDGE_LIBS cv_bridge PATH /opt/ros/fuerte/stacks/vision_opencv/cv_bridge/lib)
include_directories(/opt/ros/fuerte/stacks/vision_opencv/cv_bridge/include)

include_directories(/opt/ros/groovy/include)

# source --> build --> link --> install
set(SRC_FILES Node.cpp Messages.cpp)
add_executable(${PROJECT_NAME}_node ${SRC_FILES})
target_link_libraries(${PROJECT_NAME}_node  ${PROJECT_NAME}_lib 
                                            ${ROS_LIBRARIES} 
                                            ${OpenCV_LIBS} 
                                            ${IMAGE_TRANSPORT_LIBS} 
                                            ${IMAGE_GEOMETRY_LIBS}
                                            ${CV_BRIDGE_LIBS}
)
install(TARGETS ${PROJECT_NAME}_node
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
)
