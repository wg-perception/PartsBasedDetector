ectomodule(object_recognition_by_parts_cells DESTINATION ${PROJECT_NAME}
                                             INSTALL
                                             module.cpp
                                             detect.cpp
)

find_package(object_recognition_core REQUIRED)
find_package(catkin REQUIRED COMPONENTS object_recognition_core ecto ecto_image_pipeline ecto_pcl)

include_directories(SYSTEM ${object_recognition_core_INCLUDE_DIRS}
                           ${catkin_INCLUDE_DIRS}
                           ${OpenCV_INCLUDE_DIRS}
                           ${PCL_INCLUDE_DIRS})

link_ecto(object_recognition_by_parts_cells     ${OpenCV_LIBRARIES}
                                                ${Boost_LIBRARIES}
                                                ${object_recognition_core_LIBRARIES}
                                                ${PCL_LIBRARIES}
                                                ${catkin_LIBRARIES}
                                                ${MatlabIO_LIBRARIES}
                                                ${PROJECT_NAME}_lib
)
