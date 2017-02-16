add_executable(highPerfStereo highPerfStereo.cpp)
target_link_libraries(highPerfStereo plane linearSystem sparsestereo fade2d ${OpenCV_LIBS})