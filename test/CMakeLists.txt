# object recognition by parts tests
find_package(object_recognition_core QUIET)
if (object_recognition_core_FOUND)
    enable_testing()
    object_recognition_core_detection_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/config_detection.by_parts)
    #object_recognition_core_training_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/config_training.by_parts)
endif()
