#pragma once

#include "std_include.h"
#include "cParam.h"

#include <cstdlib>
#include <stdio.h>
#include <limits>
#include <time.h>

#include "TRW_S-v1.3/MRFEnergy.h"

class cMRF
{
public:
	cMRF();
	~cMRF();

	void computeCost(std::vector<std::vector<cv::Point>> cell_coors, std::vector<cv::Mat> cell_cands, std::vector<cv::Mat> cell_cands_dists, cParam p,
		cv::Mat *o_unaryCost, std::vector<cv::Mat> * o_pairwiseCost, cv::Mat* o_mapMat,
		std::vector<cv::Point> * o_all_coors, cv::Mat *o_all_cands, std::vector<std::vector<int>>  *o_cell_coors_to_all_coors);
	void GetTruncatedPairwiseCost(cv::Point coor1, cv::Point coor2, cv::Mat cands1, cv::Mat cands2, int dummy_pairwise_cost1, int dummy_pairwise_cost2, cv::Mat *o_mapMat);
	//void GetIntervalCost();
	//void unique(cv::Mat inputMat, std::vector<cv::Point> *o_uniqueSotrtPts, std::vector<int> *o_uniqueSotrtIdx);

	void mrf_trw_s(double *u, int uw, int uh, double **p, double* m, int nm, int mw, int mh, /*int in_Method, int in_iter_max, int in_min_iter,*/
		double *e, double **s);
};

