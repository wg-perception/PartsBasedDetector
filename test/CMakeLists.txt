# object recognition by parts tests
find_package(object_recognition_core QUIET)
if (object_recognition_core_FOUND)
    enable_testing()
    object_recognition_core_config_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/config_face.by_parts)
    object_recognition_core_config_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/config_person.by_parts)
    #object_recognition_core_config_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/config_training.by_parts)
    # publisher tests
    object_recognition_core_sink_test(Publisher "object_recognition_by_parts" "{}")
endif()
