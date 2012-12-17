#install targets for all things python
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/object_recognition_by_parts
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  COMPONENT main
  FILES_MATCHING PATTERN "*.py"
)
