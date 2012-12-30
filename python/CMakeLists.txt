#install targets for all things python
if (${ROS_GROOVY_OR_ABOVE_FOUND})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/object_recognition_by_parts
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  COMPONENT main
  FILES_MATCHING PATTERN "*.py"
)
else()
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/object_recognition_by_parts
  DESTINATION ${PYTHON_PACKAGES_PATH}
  COMPONENT main
  FILES_MATCHING PATTERN "*.py"
)
endif()
