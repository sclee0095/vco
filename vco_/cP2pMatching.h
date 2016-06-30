#pragma once


#include "std_include.h"
#include "cParam.h"



class cP2pMatching
{
public:
	cP2pMatching(int size=24);
	~cP2pMatching();

	//void run(cv::Mat gt_bimg_t , std::vector<cv::Point> &J, cv::Mat &bJ, std::vector<std::vector<cv::Point>> &E, cv::Mat &mapMat);
	void run(cv::Mat img_t, cv::Mat img_tp1, cv::Mat gt_bimg_t, cParam p, int t_x, int t_y, cv::Mat ivessel_tp1, int fidx_tp1, char* savePath,
		bool bVerbose,
		std::vector<std::vector<cv::Point>> *o_E, std::vector<cv::Mat>* o_cell_cands, std::vector<cv::Mat>* o_cell_cands_dists, std::vector<cv::Point> *o_J);
	
	void MakeGraphFromImage(cv::Mat bimg, std::vector<cv::Point> &J, cv::Mat &bJ, std::vector<std::vector<cv::Point>> &E, cv::Mat &mapMat);
	void endp(cv::Mat &src, cv::Mat &dst);
	void skel(cv::Mat &src, cv::Mat &dst);
	void applylut_8(cv::Mat &src, cv::Mat &dst, cv::Mat& lut);
	void applylut_1(cv::Mat &src, cv::Mat &dst);
	void GetLutSkel(cv::Mat& Lut);
	void applylut_branch(cv::Mat &src, cv::Mat &dst);
	void branch(cv::Mat &src, cv::Mat &dst);
	void applylut_backcount4(cv::Mat &src, cv::Mat &dst);
	void backcount4(cv::Mat &src, cv::Mat &dst);
	void applylut_thin2(cv::Mat &src, cv::Mat &dst);
	void applylut_thin1(cv::Mat &src, cv::Mat &dst);
	void thin(cv::Mat &src, cv::Mat &dst);
	cv::Mat ExtractPatchWithZeroPadding(cv::Mat img, cv::Point patch_center, int patch_size);
	std::vector<cv::Point> bresenham(cv::Point xy1, cv::Point xy2);
	void GetCandidates(cv::Mat img_t, cv::Mat img_tp1, std::vector<cv::Point> E,
		cv::Point tran_vec, float* d_tp1, int d_tp1_numkeys, cv::Mat idx_img_tp1, cv::Mat ivessel_tp1, cParam p, cv::Mat cand_ivessel,
		cv::Mat &arr_cands, cv::Mat &arr_cands_dists, cv::Mat d_tp1_img);
	
	int patchSize;
	int halfPatchSize ;
	
};

