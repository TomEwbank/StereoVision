add_executable(triangulation-example triangulation-example.cpp)
target_link_libraries(triangulation-example sparsestereo fade2d ${OpenCV_LIBS})
