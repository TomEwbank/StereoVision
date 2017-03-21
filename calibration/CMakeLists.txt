#add_executable(calib_stereo calib_stereo.cpp)
#target_link_libraries(calib_stereo ${OpenCV_LIBS})
add_executable(calib_intrinsic calib_intrinsic.cpp)
target_link_libraries(calib_intrinsic ${OpenCV_LIBS})
