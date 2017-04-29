


### Function to prepend a string to all elements of a list
FUNCTION(PREPEND var prefix)
    SET(listVar "")
    FOREACH (f ${ARGN})
        LIST(APPEND listVar "${prefix}/${f}")
    ENDFOREACH (f)
    SET(${var} "${listVar}" PARENT_SCOPE)
ENDFUNCTION(PREPEND)

set(HSEG_SOURCE_FILES include/Image/Image.h include/helper/coordinate_helper.h include/helper/image_helper.h src/helper/image_helper.cpp include/helper/opencv_helper.h src/helper/opencv_helper.cpp src/Energy/EnergyFunction.cpp include/Energy/EnergyFunction.h include/Image/Coordinates.h src/Energy/Weights.cpp include/Energy/Weights.h include/helper/hash_helper.h src/Timer.cpp include/Timer.h src/Accuracy/ConfusionMatrix.cpp include/Accuracy/ConfusionMatrix.h include/Inference/InferenceIterator.h include/Inference/InferenceResult.h include/Inference/InferenceResultDetails.h src/Threading/ThreadPool.cpp include/Threading/ThreadPool.h include/typedefs.h src/Image/FeatureImage.cpp include/Image/FeatureImage.h include/Image/Feature.h src/Energy/LossAugmentedEnergyFunction.cpp include/Energy/LossAugmentedEnergyFunction.h include/Inference/Cluster.h include/helper/clustering_helper.h src/helper/clustering_helper.cpp include/Energy/IStepSizeRule.h src/Energy/DiminishingStepSizeRule.cpp include/Energy/DiminishingStepSizeRule.h src/Energy/AdamStepSizeRule.cpp include/Energy/AdamStepSizeRule.h)
set(HSEG_INCLUDE_DIRS ${HSEG_DIR}/include)
set(HSEG_INCLUDE_SYS_DIRS ${trw_s_INCLUDE_DIRS} ${properties_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PNG_INCLUDE_DIRS} ${MATIO_INCLUDE_DIRS} ${dense_crf_INCLUDE_DIRS})
set(HSEG_LIBS trw_s densecrf properties ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PNG_LIBRARIES} ${MATIO_LIBRARIES})
PREPEND(SOURCE_FILES_FULL_PATH ${HSEG_DIR} ${HSEG_SOURCE_FILES})

add_library(hseg STATIC ${SOURCE_FILES_FULL_PATH})
target_include_directories(hseg PUBLIC ${HSEG_INCLUDE_DIRS})
target_include_directories(hseg SYSTEM PUBLIC  ${HSEG_INCLUDE_SYS_DIRS})
target_link_libraries(hseg ${HSEG_LIBS})
set_target_properties(hseg PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Wpedantic")
if(THREADS_HAVE_PTHREAD_ARG)
    target_compile_options(PUBLIC hseg "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(hseg "${CMAKE_THREAD_LIBS_INIT}")
endif()