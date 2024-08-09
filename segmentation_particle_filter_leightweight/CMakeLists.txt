cmake_minimum_required(VERSION 2.8.3)
project(segmentation_particle_filter_leightweight)
# set(CPP_PYTHON_MODULE spf_cpp)

# Additional included cmake files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message( ${CMAKE_INSTALL_PREFIX})

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
if(${CMAKE_VERSION} VERSION_LESS "3.1")
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else ()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif ()
else()
	set(CMAKE_CXX_STANDARD 11)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
)

## Python options -- choose 3.X please
#set(PYTHON_OPTIONS "2.X" "3.X")
#set(PYTHON_DESIRED_VERSION "3.X" CACHE STRING "Choose which python version to use, options are: ${PYTHON_OPTIONS}.")
#set_property(CACHE PYTHON_DESIRED_VERSION PROPERTY STRINGS ${PYTHON_OPTIONS})

# detect python as well as python libraries
#include("DetectPython")

find_package(OpenCV COMPONENTS core imgproc REQUIRED)
#set(Python_ADDITIONAL_VERSIONS ${PYTHON3_VERSION_MAJOR}.${PYTHON3_VERSION_MINOR})
#    find_package(Boost COMPONENTS python${PYTHON3_VERSION_MAJOR}${PYTHON3_VERSION_MINOR} QUIET)
#    if(NOT Boost_FOUND)
#       message(STATUS "Trying alternative Boost.Python component name, python-py<version>...")
#	     find_package(Boost COMPONENTS python-py${PYTHON3_VERSION_MAJOR}${PYTHON3_VERSION_MINOR} REQUIRED)
#    endif()



## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

#set(PYTHON_INCLUDE_DIRS ${PYTHON3_INCLUDE_DIR} ${PYTHON3_INCLUDE_DIR2} ${PYTHON3_NUMPY_INCLUDE_DIRS})
#set(PYTHON_NUMPY_INCLUDE_DIRS ${PYTHON3_NUMPY_INCLUDE_DIRS})
# set(PYTHON_LIBRARIES ${PYTHON3_LIBRARY})
#set(PYTHON_EXECUTABLE ${PYTHON3_EXECUTABLE})
#set(PYTHON_PACKAGES_PATH ${PYTHON3_PACKAGES_PATH})
#set(ARCHIVE_OUTPUT_NAME ${CPP_PYTHON_MODULE}_py3)
#set(ARCHIVE_OUTPUT_SUFFIX _py3)

#find_package_handle_standard_args(Python DEFAULT_MSG PYTHON_INCLUDE_DIRS PYTHON_NUMPY_INCLUDE_DIRS PYTHON_EXECUTABLE PYTHON_PACKAGES_PATH)
#if(NOT Python_FOUND)
#    message(SEND_ERROR "Not all requred components of Numpy/Python found.")
#endif()


################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
#add_message_files(
#  FILES
#  ParticleSet.msg
#)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
#generate_messages(
#  DEPENDENCIES
#  sensor_msgs
#)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#   INCLUDE_DIRS include
   LIBRARIES segmentation_particle_filter_leightweight
   DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
#include_directories(
# include
#  ${Boost_INCLUDE_DIRS}
#  ${catkin_INCLUDE_DIRS}
#  ${OpenCV_INCLUDE_DIRS}
#  ${PYTHON_INCLUDE_DIRS}
#)

## Declare a C++ library
#add_library(${PROJECT_NAME} SHARED
#  include/pyboostcvconverter/pyboostcvconverter.hpp
#  src/cpp_sources/pyboost_cv2_converter.cpp
#  src/cpp_sources/pyboost_cv3_converter.cpp
#  src/cpp_sources/pyboost_cv4_converter.cpp
#  src/cpp_sources/python_module.cpp
#  include/${PROJECT_NAME}/CustomGraphSegmenter.h
#  src/cpp_sources/CustomGraphSegmenter.cpp
#)

#target_include_directories(${PROJECT_NAME}  PUBLIC
#                            include/
#                             ${Boost_INCLUDE_DIRS}
#                             ${OpenCV_INCLUDE_DIRS}
#                             ${PYTHON_INCLUDE_DIRS}
#                             )

#add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


## Specify libraries to link a library or executable target against
#target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
#install(FILES
#  launch/spf.launch
  # myfile2
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
#)

########
## Python Module install
########



#set(__python_ext_suffix_var "EXT_SUFFIX")
#if(${PYTHON_DESIRED_VERSION} STREQUAL "2.X")
#    set(__python_ext_suffix_var "SO")
#endif()

#execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('${__python_ext_suffix_var}'))"
#        RESULT_VARIABLE PYTHON_${PROJECT_NAME}_PY_PROCESS
#        OUTPUT_VARIABLE ${PROJECT_NAME}_PY_SUFFIX
#        OUTPUT_STRIP_TRAILING_WHITESPACE)
#if(NOT ${PYTHON_${PROJECT_NAME}_PY_PROCESS} EQUAL 0)
#    set(${PROJECT_NAME}_PY_SUFFIX ".so")
#endif()


#set_target_properties(${PROJECT_NAME} PROPERTIES
#        ARCHIVE_OUTPUT_NAME ${ARCHIVE_OUTPUT_NAME}  # prevent name conflict for python2/3 outputs
#        PREFIX ""
#        OUTPUT_NAME ${CPP_PYTHON_MODULE}
#        SUFFIX ${${PROJECT_NAME}_PY_SUFFIX})



#if (MSVC AND NOT PYTHON_DEBUG_LIBRARIES)
#    set(PYTHON_INSTALL_CONFIGURATIONS CONFIGURATIONS Release)
#else ()
#    set(PYTHON_INSTALL_CONFIGURATIONS "")
#endif ()

#if (WIN32)
#    set(PYTHON_INSTALL_ARCHIVE "")
#else ()
#    set(PYTHON_INSTALL_ARCHIVE ARCHIVE DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python)
#endif ()


#install(TARGETS ${PROJECT_NAME}
#        ${PYTHON_INSTALL_CONFIGURATIONS}
#        RUNTIME DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python
#        LIBRARY DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python
#        ${PYTHON_INSTALL_ARCHIVE}
#        )

#execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import *; print(get_python_lib())"
#                RESULT_VARIABLE _process_out
 #               OUTPUT_VARIABLE FULL_PYTHON_DESTINATION_PATH
 #               OUTPUT_STRIP_TRAILING_WHITESPACE)

#add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
 #                  COMMAND ${CMAKE_COMMAND} -E copy
 #                      ${PY_LIB_FILE_PATH}
#                       ${FULL_PYTHON_DESTINATION_PATH}/${LIB_FILE_NAME}
#                   )


#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_segmentation_particle_filter.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
