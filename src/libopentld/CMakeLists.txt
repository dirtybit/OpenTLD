
include_directories(imacq
	mftracker
	tld
	tld/detector
    ../3rdparty/cvblobs
	${OpenCV_INCLUDE_DIRS})


set(SRC_FILES
	imacq/ImAcq.cpp
	mftracker/BB.cpp
	mftracker/BBPredict.cpp
	mftracker/FBTrack.cpp
	mftracker/Lk.cpp
	mftracker/Median.cpp
	tld/Clustering.cpp
	tld/DetectionResult.cpp
	tld/detector/DetectorCascade.cpp
	tld/MedianFlowTracker.cpp
	tld/TLD.cpp
	tld/TLDUtil.cpp
	tld/detector/EnsembleClassifier.cpp
	tld/detector/ForegroundDetector.cpp
	tld/detector/NNClassifier.cpp
	tld/detector/VarianceFilter.cpp
	imacq/ImAcq.h
	mftracker/BB.h
	mftracker/BBPredict.h
	mftracker/FBTrack.h
	mftracker/Lk.h
	mftracker/Median.h
	tld/Clustering.h
	tld/DetectionResult.h
	tld/DetectorCascade.h
	tld/MedianFlowTracker.h
	tld/TLD.h
	tld/TLDUtil.h
	tld/Timing.h
	tld/detector/EnsembleClassifier.h
	tld/detector/ForegroundDetector.h
	tld/detector/IntegralImage.h
	tld/detector/NNClassifier.h
	tld/detector/NormalizedPatch.h
	tld/detector/VarianceFilter.h)


if(CUDA_ENABLED)
	include_directories(${CUDA_INCLUDE_DIRS})
	#set(CUDA_SRC_FILES opentld.cu)
	set(SRC_FILES ${SRC_FILES} ${CUDA_SRC_FILES})
	cuda_add_library(libopentld ${SRC_FILES})
else(CUDA_ENABLED)
	add_library(libopentld ${SRC_FILES})
endif(CUDA_ENABLED)

link_directories(${OpenCV_LIB_DIR})
target_link_libraries(libopentld ${OpenCV_LIBS})

set_target_properties(libopentld PROPERTIES OUTPUT_NAME opentld)
