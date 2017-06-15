add_executable(highPerfStereo highPerfStereo.cpp highPerfStereoLib.cpp GroundTruth.cpp PerformanceEvaluator.cpp PerformanceEvaluator.h BallGroundTruth.cpp BallGroundTruth.h)
target_link_libraries(highPerfStereo plane linearSystem sparsestereo fade2d ${OpenCV_LIBS})

add_executable(mainStereo main.cpp highPerfStereoLib.cpp GroundTruth.cpp PerformanceEvaluator.cpp PerformanceEvaluator.h BallGroundTruth.cpp BallGroundTruth.h)
target_link_libraries(mainStereo plane linearSystem sparsestereo fade2d ${OpenCV_LIBS})