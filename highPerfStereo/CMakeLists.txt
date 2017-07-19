add_executable(highPerfStereo highPerfStereo.cpp highPerfStereoLib.cpp GroundTruth.cpp PerformanceEvaluator.cpp PerformanceEvaluator.h BallGroundTruth.cpp BallGroundTruth.h)
target_link_libraries(highPerfStereo plane linearSystem sparsestereo fade2d ${OpenCV_LIBS})

add_executable(mainStereo main.cpp highPerfStereoLib.cpp GroundTruth.cpp PerformanceEvaluator.cpp PerformanceEvaluator.h BallGroundTruth.cpp BallGroundTruth.h)
target_link_libraries(mainStereo plane linearSystem sparsestereo fade2d ${OpenCV_LIBS})

add_executable(ballErrorEstimation ballErrorEstimation.cpp highPerfStereoLib.cpp BallGroundTruth.cpp BallGroundTruth.h)
target_link_libraries(ballErrorEstimation plane linearSystem sparsestereo fade2d ${OpenCV_LIBS})

add_executable(planeErrorEstimation planeErrorEstimation.cpp highPerfStereoLib.cpp)
target_link_libraries(planeErrorEstimation plane linearSystem sparsestereo fade2d ${OpenCV_LIBS} ${MRPT_LIBS})

add_executable(errorCurve errorCurve.cpp highPerfStereoLib.cpp)
target_link_libraries(errorCurve plane linearSystem sparsestereo fade2d ${OpenCV_LIBS})