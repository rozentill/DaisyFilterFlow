#include "PatchMatchFilter.cuh"
PatchMatchFilter::PatchMatchFilter(cv::Mat img1, cv::Mat img2)
{
	imSrcOrigin = img1.clone();
	imRefOrigin = img2.clone();
}

PatchMatchFilter::~PatchMatchFilter()
{
}

void PatchMatchFilter::Initialization(){
	
	//for kernel
	kernelSize = 9;
	
	subRadius = 2 * kernelSize;

	//for superpixel
	spNumber = 300;
	spSize = 300;
	spNumOrSize = 1;

	//for cost
	channels = 3;

	//for iteration
	iteration = 40;

	//initialize
	bestCost = (float *)malloc(sizeof(float)*imSrcOrigin.cols*imSrcOrigin.rows);
	bestFlow = (int2 *)malloc(sizeof(int2)*imSrcOrigin.cols*imSrcOrigin.rows);
}


void PatchMatchFilter::GetSuperpixelsListFromSegment(const cv::Mat_<int> &segLabels, int numOfLabels, std::vector<std::vector<cv::Vec2i>> &spPixelsList)
{
	int iy, ix, height, width;
	height = segLabels.rows;
	width = segLabels.cols;

	spPixelsList.clear();
	spPixelsList.resize(numOfLabels);
	for (iy = 0; iy<numOfLabels; ++iy)
		spPixelsList[iy].clear();
	for (iy = 0; iy<height; ++iy)
	{
		for (ix = 0; ix<width; ++ix)
		{
			int tmpLabel = segLabels[iy][ix];
			spPixelsList[tmpLabel].push_back(cv::Vec2i(iy, ix));
		}
	}
}



void PatchMatchFilter::CreateAndOrganizeSuperpixels(){
	cv::Mat im_src = imSrcOrigin, im_ref = imRefOrigin;
	cv::Mat_<int> labelSrc;
	cv::Mat_<cv::Vec4i> subSrc;//sub-image
	cv::Mat_<cv::Vec4i> spSrc;//superpixel

	int numLabelSrc;

	clock_t start, end;
	start = clock();

	numLabelSrc = CreateSLICSegments(im_src, labelSrc, spNumber, spSize, spNumOrSize);
	GetSubImageRangeFromSegments(labelSrc, numLabelSrc, subRadius, subSrc, spSrc);

	subRangeSrc = subSrc.clone();
	spRangeSrc = spSrc.clone();

	numOfLabelsSrc = numLabelSrc;
	segLabelsSrc = labelSrc.clone();

	GetSuperpixelsListFromSegment(segLabelsSrc, numOfLabelsSrc, superpixelsListSrc);

	/*********************** draw superpixel ********************************/
	//char *WINDOW_SEGMENT_CONTOUR = "segement_contour";
	//char *WINDOW_SEGMENT_CONTOUR_RIGHT = "segement_contour_right";
	//cvNamedWindow(WINDOW_SEGMENT_CONTOUR, CV_WINDOW_AUTOSIZE);
	//cvNamedWindow(WINDOW_SEGMENT_CONTOUR_RIGHT, CV_WINDOW_AUTOSIZE);
	//	
	//cv::Mat_<cv::Vec3b> resImg;
	//DrawContoursAroundSegments(im_src, labelSrc, resImg);
	//cv::imshow(WINDOW_SEGMENT_CONTOUR, resImg);
	//cv::Mat_<cv::Vec3b> resImgRef;
	//DrawContoursAroundSegments(im_ref, labelRef, resImgRef);
	//cv::imshow(WINDOW_SEGMENT_CONTOUR_RIGHT, resImgRef);
	//cv::imwrite("sup_src.png", resImg);
	//cv::imwrite("sup_ref.png", resImgRef);

	//cvWaitKey();
	//cv::destroyAllWindows();
	/*************************************************************************/

	end = clock();
	std::cout << "Finished creating superpixels, time :" << (end - start) / CLOCKS_PER_SEC << "s.\n";
}

void PatchMatchFilter::RandomAssignRepresentativePixel(const std::vector<std::vector<cv::Vec2i>> &spPixelsList, int numOfLabels, cv::Mat_<cv::Vec2i> &rePixel){
	
	rePixel.create(numOfLabels, 1);
	cv::RNG rng;
	int iy;
	for (iy = 0; iy<numOfLabels; ++iy)
	{
		rePixel[iy][0] = spPixelsList[iy][rng.next() % spPixelsList[iy].size()];
	}
}

void PatchMatchFilter::InitiateBufferData(){
	subImageSrc.clear();
	subImageSrc.resize(numOfLabelsSrc);

	for (int iy = 0; iy<numOfLabelsSrc; ++iy)
	{
		int py, px;
		py = repPixelsSrc[iy][0][0];
		px = repPixelsSrc[iy][0][1];
		// extract sub-image from subrange
		int w = subRangeSrc[py][px][2] - subRangeSrc[py][px][0] + 1;
		int h = subRangeSrc[py][px][3] - subRangeSrc[py][px][1] + 1;
		int x = subRangeSrc[py][px][0];
		int y = subRangeSrc[py][px][1];

		subImageSrc[iy] = imSrcOrigin(cv::Rect(x, y, w, h)).clone();
	}
}

void PatchMatchFilter::BuildSuperpixelsPropagationGraph(const cv::Mat_<int> &refSegLabel, int numOfLabels, const cv::Mat_<cv::Vec3f> &refImg, GraphStructure &spGraph)
{
	spGraph.adjList.clear();
	spGraph.vertexNum = 0;
	// build superpixel connectivity graph
	spGraph.ReserveSpace(numOfLabels * 20);
	spGraph.SetVertexNum(numOfLabels);
	int iy, ix, height, width;
	height = refSegLabel.rows;
	width = refSegLabel.cols;
	for (iy = 0; iy<height; ++iy)
	{
		for (ix = 0; ix<width; ++ix)
		{
			int tmp1 = refSegLabel[iy][ix];
			if (iy > 0)
			{
				int tmp2 = refSegLabel[iy - 1][ix];
				if (tmp1 != tmp2)
				{
					spGraph.AddEdge(tmp1, tmp2);
					//spGraph.AddEdge(tmp2, tmp1);
				}
			}

			if (ix > 0)
			{
				int tmp2 = refSegLabel[iy][ix - 1];
				if (tmp1 != tmp2)
				{
					spGraph.AddEdge(tmp1, tmp2);
					//spGraph.AddEdge(tmp2, tmp1);
				}
			}
		}
	}
}

void PatchMatchFilter::RunPatchMatchFilter(){

	RandomAssignRepresentativePixel(superpixelsListSrc, numOfLabelsSrc, repPixelsSrc);
	
	InitiateBufferData();//initialize some buffer data as sub image

	int iy, ix;

	spFlowVisitedNumberSrc.resize(numOfLabelsSrc);
	for (iy = 0; iy<numOfLabelsSrc; ++iy) spFlowVisitedNumberSrc[iy] = 0;

	spFlowVisitedSrc.resize(numOfLabelsSrc);
	for (iy = 0; iy<numOfLabelsSrc; ++iy) spFlowVisitedSrc[iy].clear();

	std::cout << "Now initializing..." << std::endl;

	for (iy = 0; iy < imSrcOrigin.rows; iy++)
	{
		for (ix = 0; ix < imSrcOrigin.cols; ix++)
		{
			bestFlow[iy*imSrcOrigin.cols + ix] = make_int2(0,0);
			bestCost[iy*imSrcOrigin.cols + ix] = DOUBLE_MAX;

		}
	}

	BuildSuperpixelsPropagationGraph(segLabelsSrc, numOfLabelsSrc, imSrcOrigin, spGraphSrc);

	for (iy = 0; iy < numOfLabelsSrc; iy++)
	{
		int ry = repPixelsSrc[iy][0][0];
		int rx = repPixelsSrc[iy][0][1];
		
		ImproveFlow(ry, rx, std::vector<int2>(1, make_int2(0, 0)));
	}

	for (int iter = 0; iter < iteration; iter++)
	{
		RandomAssignRepresentativePixel(superpixelsListSrc, numOfLabelsSrc, repPixelsSrc);

		int ystart = 0, yend = numOfLabelsSrc, ychange = 1;
		if (iter % 2 == 1)
		{
			ystart = numOfLabelsSrc - 1; yend = -1; ychange = -1;
		}

		for (iy = ystart; iy < yend; iy+=ychange)
		{
			std::vector<int2> dListVec;
			dListVec.clear();

			int refY, refX;
			refY = repPixelsSrc[iy][0][0];
			refX = repPixelsSrc[iy][0][1];

			std::set<int>::iterator sIt;
			std::set<int> &sAdj = spGraphSrc.adjList[iy];

			/* Propagation from neighbor */
			for (sIt = sAdj.begin(); sIt != sAdj.end(); sIt++)
			{
				repPixelsSrc[*sIt][0] = superpixelsListSrc[*sIt][rand() % superpixelsListSrc[*sIt].size()];
				int ky, kx;
				ky = repPixelsSrc[*sIt][0][0];
				kx = repPixelsSrc[*sIt][0][1];

				int2 tmpFlow = bestFlow[ky*imSrcOrigin.cols+kx];

				dListVec.push_back(tmpFlow);
			}

			/* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
			const float randomRatio = 2.0;

			float mag = std::min<float>(imRefOrigin.cols, imRefOrigin.rows);
			int2 tmpFlow = bestFlow[refY*imSrcOrigin.cols + refX];

			for (; mag >= 1; mag/=randomRatio)
			{
				float deltaHorLabel = (float(rand()) / RAND_MAX - 0.5)*2.0*mag;
				float deltaVerLabel = (float(rand()) / RAND_MAX - 0.5)*2.0*mag;

				float tmpHorLabel = tmpFlow.x + deltaHorLabel;
				float tmpVerLabel = tmpFlow.y + deltaVerLabel;

				tmpHorLabel = floor(tmpHorLabel + 0.5);
				tmpVerLabel = floor(tmpVerLabel + 0.5);

				if (tmpHorLabel < 0 || tmpHorLabel > imRefOrigin.cols || tmpVerLabel < 0 || tmpVerLabel > imRefOrigin.rows) continue;

				dListVec.push_back(make_int2(tmpHorLabel, tmpVerLabel));
			}
			ImproveFlow(refY, refX, dListVec);
		}

	}
}

void PatchMatchFilter::ImproveFlow(int py, int px, std::vector<int2> flowList){
	int flowSize = flowList.size();
	int w = subRangeSrc[py][px][2] - subRangeSrc[py][px][0] + 1;
	int h = subRangeSrc[py][px][3] - subRangeSrc[py][px][1] + 1;
	int x = subRangeSrc[py][px][0];
	int y = subRangeSrc[py][px][1];

	int dy, dx, ry, rx, oy, ox, cy, cx, cf, cc, sy, sx;

	float costTmp, costTotal = 0;

	//raw cost
	cv::Mat_<float> rawCost;
	rawCost.create(h, w*flowSize);

	for (int cf = 0; cf < flowSize; cf++)
	{
		int2 fl = flowList[cf];
		dy = fl.y;
		dx = fl.x;
		cv::Mat_<float> localRc = rawCost(cv::Rect(cf*w, 0, w, h));

		for (oy = y, cy = 0; cy < h; oy++, cy++)
		{
			for (ox = x, cx = 0; cx < w; ox++, cx++)
			{
				costTotal = 0;
				ry = oy + dy;
				rx = ox + dx;

				(ry < 0) ? ry = 0 : NULL;
				(ry >= imRefOrigin.rows) ? ry = imRefOrigin.rows - 1 : NULL;
				(rx < 0) ? rx = 0 : NULL;
				(rx >= imRefOrigin.cols) ? rx = imRefOrigin.cols - 1 : NULL;

				for (cc = 0; cc < channels; cc++)
				{
#if USE_COLOR_FEATURE
					costTmp = imSrcOrigin[oy][ox][cc] - imRefOrigin[ry][rx][cc];
					costTotal += costTmp*costTmp;
#endif // USE_COLOR_FEATURE

				}
				localRc[cy][cx] = costTotal;
			}
		}
	}

	//filtered cost
	cv::Mat_<float> filteredCost(h, w*flowSize);
	rawCost.copyTo(filteredCost);

	int spw = spRangeSrc[py][px][2] - spRangeSrc[py][px][0] + 1;
	int sph = spRangeSrc[py][px][3] - spRangeSrc[py][px][1] + 1;
	int spx = spRangeSrc[py][px][0];
	int spy = spRangeSrc[py][px][1];

	for (cf = 0; cf < flowSize; cf++)
	{
		oy = spy;
		sy = spy - y;
		for (cy = 0; cy < sph; cy++, oy++, sy++)
		{
			ox = spx;
			sx = spx - x;
			for (cx = 0; cx < spw; cx++, ox++, sx++)
			{
				costTmp = filteredCost[sy][sx + cf*w];
				if (costTmp < bestCost[oy*imSrcOrigin.cols+ox])
				{
					bestCost[oy*imSrcOrigin.cols + ox] = costTmp;
					bestFlow[oy*imSrcOrigin.cols + ox].x = flowList[cf].x;
					bestFlow[oy*imSrcOrigin.cols + ox].y = flowList[cf].y;
					//std::cout << "Updating x : " << ox << ", y : " << oy << " with flow : (" << flowList[cf].x << ", " << flowList[cf].y << ") and cost : " << costTmp << std::endl;
				}
			}
		}
	}
	

}


void PatchMatchFilter::ReconstructFlow(){
	cv::Mat flow = imSrcOrigin.clone();
	float maxVerRange = std::max<float>(imSrcOrigin.rows, imRefOrigin.rows);
	float maxHorRange = std::max<float>(imSrcOrigin.cols, imRefOrigin.cols);

	for (int iy = 0; iy < imRefOrigin.rows; iy++)
	{
		for (int ix = 0; ix < imRefOrigin.cols; ix++)
		{
			flow.at<cv::Vec3b>(iy, ix)[0] = 0;
			flow.at<cv::Vec3b>(iy, ix)[1] = (uchar)(255 * (bestFlow[iy*imSrcOrigin.cols + ix].x + maxHorRange) / (2 * maxHorRange));
			flow.at<cv::Vec3b>(iy, ix)[2] = (uchar)(255 * (bestFlow[iy*imSrcOrigin.cols + ix].y + maxVerRange) / (2 * maxVerRange));

		}
	}

	cv::imwrite(root + "flow.png", flow);
}

void PatchMatchFilter::ReconstructSrc(){
	cv::Mat res = imSrcOrigin.clone();

	int2 fl;
	int ry, rx;
	for (int iy = 0; iy < imRefOrigin.rows; iy++)
	{
		for (int ix = 0; ix < imRefOrigin.cols; ix++)
		{
			fl = bestFlow[iy*imSrcOrigin.cols+ix];
			ry = iy + fl.y;
			rx = ix + fl.x;
			res.at<cv::Vec3b>(iy, ix)[0] = imRefOrigin[ry][rx][0];
			res.at<cv::Vec3b>(iy, ix)[1] = imRefOrigin[ry][rx][1];
			res.at<cv::Vec3b>(iy, ix)[2] = imRefOrigin[ry][rx][2];

		}
	}

	cv::imwrite(root + "result.png", res);
}