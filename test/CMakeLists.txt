# LINE-MOD tests
find_package(object_recognition_core REQUIRED)
enable_testing()
object_recognition_core_detection_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/config_detection.by_parts)
#object_recognition_core_training_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/config_training.linemod)
