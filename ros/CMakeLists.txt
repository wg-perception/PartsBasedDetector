# -----------------------------------------------
# ROS NODE WRAPPER
# -----------------------------------------------

# if one Groovy
if (${catkin_VERSION} VERSION_GREATER "0.5.28")
find_package(catkin REQUIRED COMPONENTS catkin rosconsole cv_bridge image_geometry image_transport roscpp message_filters pcl pcl_ros)
include_directories(${catkin_INCLUDE_DIRS})

find_library(IMAGE_GEOMETRY_LIBS image_geometry PATH /opt/ros/groovy/lib)
#include_directories(/opt/ros/groovy/include)
else()
# only Fuerte is supported otherwise
# manually get non-catkin packages
find_library(IMAGE_TRANSPORT_LIBS  image_transport PATHS /opt/ros/fuerte/stacks/image_common/image_transport/lib)
include_directories(/opt/ros/fuerte/stacks/image_common/image_transport/include)

find_library(IMAGE_GEOMETRY_LIBS image_geometry PATH /opt/ros/fuerte/stacks/vision_opencv/image_geometry/lib)
include_directories(/opt/ros/fuerte/stacks/vision_opencv/image_geometry/include)

find_library(CV_BRIDGE_LIBS cv_bridge PATH /opt/ros/fuerte/stacks/vision_opencv/cv_bridge/lib)
include_directories(/opt/ros/fuerte/stacks/vision_opencv/cv_bridge/include)

find_library(PCL_ROS_LIBS NAMES pcl_ros_filters pcl_ros_io pcl_ros_tf PATHS /opt/ros/fuerte/stacks/perception_pcl/pcl_ros/lib)
include_directories(/opt/ros/fuerte/stacks/perception_pcl/pcl_ros/include)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
set(PCL_LIBS ${PCL_LIBRARIES})

find_package(ROS REQUIRED COMPONENTS catkin roscpp message_filters)
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
                                            ${PCL_ROS_LIBS}
                                            ${PCL_LIBS}
)
install(TARGETS ${PROJECT_NAME}_node
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
)
