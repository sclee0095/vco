#pragma once

#include "std_include.h"
#include "cParam.h"
#include "cChamferMatching.h"
#include "cFastMarching.h"
#include "cFrangiFilter.h"
#include "cGeodesic.h"
#include "cVCO.h"
#include "cP2pMatching.h"


//#ifdef __cplusplus
//extern "C"{
//#endif


/*TESTDLL_NKJ void VesselCorrespondenceOptimization(cv::Mat img_t, cv::Mat img_tp1, cv::Mat bimg_t,
cParam p, std::string ave_path, int nextNum,
cv::Mat* bimg_tp1, cv::Mat* bimg_tp1_post_processed, int fidx_tp1, char* savePath);*/
TESTDLL_NKJ void VesselCorrespondenceOptimization(double* arr_img_t, double* arr_img_tp1, double* arr_bimg_t, int img_w, int img_h,
	cParam p, std::string ave_path, int nextNum,
	double** arr_bimg_tp1, double** arr_bimg_tp1_post_processed, int fidx_tp1, char* savePath, bool bVerbose = 0);
TESTDLL_NKJ void GrowVesselUsingFastMarching(cv::Mat ivessel, std::vector<cv::Point> lidx, double thre, cParam p,
	cv::Mat *o_new_bimg, std::vector<cv::Point> *o_new_lidx, std::vector<cv::Point> *o_app_lidx);
TESTDLL_NKJ void GetLineLength(std::vector<cv::Point> L, bool IS3D, double *o_ll);
TESTDLL_NKJ void getBoundaryDistance(cv::Mat I, bool IS3D, cv::Mat *o_BoundaryDistance);
TESTDLL_NKJ void maxDistancePoint(cv::Mat BoundaryDistance, cv::Mat I, bool IS3D, cv::Point *o_posD, double *o_maxD);
//TESTDLL_NKJ void maxDistancePoint(cv::Mat BoundaryDistance, cv::Mat I, bool IS3D, double *o_maxD);

//#ifdef __cplusplus
//}
//#endif