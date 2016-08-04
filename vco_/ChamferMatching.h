#pragma once
//#include <stdio.h>
//#include "opencv2/opencv.hpp"
#include "std_include.h"
#include "VCOParams.h"


class cChamferMatching
{
public:
	cChamferMatching();
	~cChamferMatching();

	static cv::Mat computeChamferMatch(cv::Mat bimg_t, cv::Mat bimg_tp1, cVCOParams p, int &t_x, int &t_y);
	static cv::Mat ChamferMatch(cv::Mat dtImg, cv::Mat tempImg);
	static cv::Point* GetAddressTable(cv::Mat tempImg, int stride, int &idxSz);
};

