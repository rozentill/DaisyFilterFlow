//#include "Program_Control_Header.h"
//
//#include "Image_Video_Capture_Header.h"
//#include "OpenCV_Tickcount_Header.h"
//#include "Scan_File_Header.h"

#include "opencv2\opencv.hpp"

#include <cstdio>


#include <Windows.h>

#include "PatchMatchFilter.cuh"
#include "Scan_File_Header.h"

#include <iostream>
#include <fstream>
using std::ifstream;


void CalcFlow(const cv::Mat &img1, const cv::Mat &img2, string root) {
	PatchMatchFilter pmf(img1, img2);
	pmf.root = root;
	pmf.Initialization();
	pmf.CreateAndOrganizeSuperpixels();
	pmf.RunPatchMatchFilter();
	pmf.ReconstructFlow();
	pmf.ReconstructSrc();
}
int main(int argc, char** argv){
	string f_src, f_ref, root;

	ScanFile::GUI_GetFileName(f_src);
	ScanFile::GUI_GetFileName(f_ref);
	root = "D:\\MSRA\\Code\\Project\\DaisyFilterFlow\\PatchMatchFilter\\PatchMatchFilter\\data\\bear\\";
	cv::Mat im_src, im_ref;

	im_src = cv::imread(f_src);
	im_ref = cv::imread(f_ref);

	cv::Mat flow;

	CalcFlow(im_src, im_ref, root);
	return 0;
}