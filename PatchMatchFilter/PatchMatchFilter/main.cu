//#include "Program_Control_Header.h"
//
//#include "Image_Video_Capture_Header.h"
//#include "OpenCV_Tickcount_Header.h"
//#include "Scan_File_Header.h"

#include "opencv2\opencv.hpp"

#include <cstdio>
#include <string>
using std::string;

#include <Windows.h>

#include "PatchMatchFilter.cuh"

#include <iostream>
#include <fstream>
using std::ifstream;


int CalcFlow(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &forwardFlow, int option = 0) {
	PatchMatchFilter pmf;


}
int main(int argc, char** argv){
	string f_src, f_ref;
	cv::Mat im_src, im_ref;

	im_src = cv::imread(f_src);
	im_ref = cv::imread(f_ref);

	cv::Mat flow;

	CalcFlow(im_src, im_ref, flow);
	return 0;
}