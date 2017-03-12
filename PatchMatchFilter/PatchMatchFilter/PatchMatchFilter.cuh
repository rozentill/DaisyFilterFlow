
#include "opencv2\opencv.hpp"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "vector_types.h"
#include "vector_functions.h"
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <set>
#include <string>
using std::string;
using std::set;
using std::max;
using std::min;

#include "Superpixels_Header.h"
#include "CommonDataStructure.cuh"

#define DOUBLE_MAX 1e10
#define USE_COLOR_FEATURE 1
#define USE_DEEP_FEATURE 0


class PatchMatchFilter
{
public:
	PatchMatchFilter(cv::Mat img1, cv::Mat img2);
	~PatchMatchFilter();

	//file
	string root;

	cv::Mat_<cv::Vec3b> imSrcOrigin, imRefOrigin;

	// the label of each pixel, for superpixel
	cv::Mat_<int> segLabelsSrc, segLabelsRef;
	cv::Mat_<cv::Vec4i> subRangeSrc, spRangeSrc;
	int spNumber, spSize, spNumOrSize;
	int numOfLabelsSrc;
	std::vector<cv::Mat_<cv::Vec3f>> subImageSrc;
	std::vector<std::vector<cv::Vec2i>> superpixelsListSrc;
	cv::Mat_<cv::Vec2i> repPixelsSrc;

	//iteration
	GraphStructure spGraphSrc;
	int iteration;

	//flow
	std::vector<std::vector<cv::Vec4f>> spFlowVisitedSrc;
	std::vector<int> spFlowVisitedNumberSrc;
	float *bestCost;
	int2 *bestFlow;

	//cost
	int channels;
	float *dataSrc, *dataRef;

	//filter
	int kernelSize, subRadius;

	void Initialization();
	void CreateAndOrganizeSuperpixels();
	void RunPatchMatchFilter();
	void RandomAssignRepresentativePixel(const std::vector<std::vector<cv::Vec2i>> &spPixelsList, int numOfLabels, cv::Mat_<cv::Vec2i> &rePixel);
	void GetSuperpixelsListFromSegment(const cv::Mat_<int> &segLabels, int numOfLabels, std::vector<std::vector<cv::Vec2i>> &spPixelsList);
	void InitiateBufferData();
	void BuildSuperpixelsPropagationGraph(const cv::Mat_<int> &refSegLabel, int numOfLabels, const cv::Mat_<cv::Vec3f> &refImg, GraphStructure &spGraph);
	void ImproveFlow(int py, int px, std::vector<int2> flowList);
	void ReconstructSrc();
	void ReconstructFlow();
	//for speed up
	//void AssociateLeftImageItselfToEstablishNonlocalPropagation(int sampleNum, int topK);
private:

};



