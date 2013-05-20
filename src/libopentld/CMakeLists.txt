set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)

include_directories(imacq
	mftracker
	tld
	tld/detector
	tld/detector/cuda
	tld/detector/cuda/npp
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
	tld/IDetectorCascade.h
	tld/IEnsembleClassifier.h
	tld/INNClassifier.h
	tld/IVarianceFilter.h
	tld/MedianFlowTracker.h
	tld/TLD.h
	tld/TLDUtil.h
	tld/Timing.h
	tld/detector/DetectorCascade.h
	tld/detector/EnsembleClassifier.h
	tld/detector/ForegroundDetector.h
	tld/detector/IntegralImage.h
	tld/detector/NNClassifier.h
	tld/detector/NormalizedPatch.h
	tld/detector/VarianceFilter.h)


if(CUDA_ENABLED)
	include_directories(${CUDA_INCLUDE_DIRS} tld/detector/cuda)
	set(CUDA_SRC_FILES
		tld/detector/cuda/CuVarianceFilter.cpp
		tld/detector/cuda/CuDetectorCascade.cpp
		tld/detector/cuda/CuEnsembleClassifier.cpp
		tld/detector/cuda/CUDA.h
		tld/detector/cuda/CuVarianceFilter.h
		tld/detector/cuda/CuDetectorCascade.h
		tld/detector/cuda/CuEnsembleClassifier.h
		tld/detector/cuda/npp/NPP_staging.hpp
		tld/detector/cuda/npp/NCV.hpp
		tld/detector/cuda/varianceFilter.cu
		tld/detector/cuda/ensembleClassifier.cu
		tld/detector/cuda/npp/NPP_staging.cu
		tld/detector/cuda/npp/NCV.cu)
	set(SRC_FILES ${SRC_FILES} ${CUDA_SRC_FILES})
	cuda_add_library(libopentld ${SRC_FILES})
else(CUDA_ENABLED)
	add_library(libopentld ${SRC_FILES})
endif(CUDA_ENABLED)

link_directories(${OpenCV_LIB_DIR})
target_link_libraries(libopentld ${OpenCV_LIBS})

set_target_properties(libopentld PROPERTIES OUTPUT_NAME opentld)
