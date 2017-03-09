#include "opencv2\opencv.hpp"

class PatchMatchFilter
{
public:
	PatchMatchFilter();
	~PatchMatchFilter();

	float *data_src, *data_ref;

	// the label of each pixel
	cv::Mat_<int> segLabelsLeft, segLabelsRight;

private:

};

