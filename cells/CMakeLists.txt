set_ecto_install_package_name(${PROJECT_NAME})

ectomodule(object_recognition_by_parts_cells    module.cpp
                                                detect.cpp
)

find_package(object_recognition_core REQUIRED)

include_directories(SYSTEM ${object_recognition_core_INCLUDE_DIRS} 
                           ${OpenCV_INCLUDE_DIR}
)

link_ecto(object_recognition_by_parts_cells     ${OpenCV_LIBRARIES}
                                                ${Boost_LIBRARIES}
                                                ${object_recognition_core_LIBRARIES}
                                                ${MatlabIO_LIBRARIES}
                                                object_recognition_by_parts
)

install_ecto_module(object_recognition_by_parts_cells)
