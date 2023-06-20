cmake_minimum_required(VERSION 2.8.3)
project(camera_package)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_package(
  CATKIN_DEPENDS 
	rospy
)
include_directories(
${catkin_INCLUDE_DIRS}
)

install(FILES
src/detector.py
DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY launch src
DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_install_python(PROGRAMS src/detector.py
DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
