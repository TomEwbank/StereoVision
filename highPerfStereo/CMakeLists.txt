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

add_executable(singleStereo singleStereo.cpp highPerfStereoLib.cpp)
target_link_libraries(singleStereo plane linearSystem sparsestereo fade2d ${OpenCV_LIBS})

add_executable(stereoDiscretization stereoDiscretization.cpp)
target_link_libraries(stereoDiscretization ${OpenCV_LIBS})

add_executable(stereoExample stereoExample.cpp)
target_link_libraries(stereoExample highPerfStereoLib sparsestereo fade2d ${OpenCV_LIBS})