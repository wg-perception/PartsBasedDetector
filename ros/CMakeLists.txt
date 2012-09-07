# -----------------------------------------------
# ROS NODE WRAPPER
# -----------------------------------------------

# automagically find the required packages
find_package(ROS REQUIRED COMPONENTS catkin roscpp image_geometry)
include_directories(${ROS_INCLUDE_DIRS})

# manually get non-catkin packages
find_library(IMAGE_TRANSPORT_LIBS  image_transport PATHS /opt/ros/groovy/stacks/image_common/image_transport/lib)
include_directories(/opt/ros/groovy/stacks/image_common/image_transport/include)

# source --> build --> link --> install
set(SRC_FILES node.cpp)
add_executable(${PROJECT_NAME}_node ${SRC_FILES})
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME}_lib ${ROS_LIBRARIES} ${OpenCV_LIBS} ${IMAGE_TRANSPORT_LIBS})
install(TARGETS ${PROJECT_NAME}_node
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
)
