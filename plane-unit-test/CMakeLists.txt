add_executable(plane-unit-test plane-unit-test.cpp)
target_link_libraries(plane-unit-test plane linearSystem sparsestereo fade2d ${OpenCV_LIBS})