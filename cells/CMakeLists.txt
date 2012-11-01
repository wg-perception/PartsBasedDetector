ectomodule(object_recognition_by_parts_cells DESTINATION ${PROJECT_NAME}
                                             INSTALL
                                             module.cpp
                                             detect.cpp
)

find_package(object_recognition_core REQUIRED)
#find_package(catkin REQUIRED COMPONENTS catkin pcl)
find_package(PCL REQUIRED)
#TODO add non-groovy code

include_directories(SYSTEM ${object_recognition_core_INCLUDE_DIRS} 
                           ${ecto_INCLUDE_DIRS}
                           ${ecto_pcl_INCLUDE_DIRS}
                           ${OpenCV_INCLUDE_DIRS}
                           ${PCL_INCLUDE_DIRS}
)

link_ecto(object_recognition_by_parts_cells     ${OpenCV_LIBRARIES}
                                                ${Boost_LIBRARIES}
                                                ${object_recognition_core_LIBRARIES}
                                                ${MatlabIO_LIBRARIES}
                                                ${PCL_LIBRARIES}
                                                ${PROJECT_NAME}_lib
)
