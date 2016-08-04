#pragma once

#include "std_include.h"
#include "ChamferMatching.h"
#include "FastMarching.h"
#include "FrangiFilter.h"
#include "Geodesic.h"
#include "VCOParams.h"
#include "P2pMatching.h"

#include <cstdlib>
#include <stdio.h>
#include <limits>
#include <time.h>

#include "TRW_S-v1.3/MRFEnergy.h"

#ifdef VCO_EXPORTS
#define VCO_DLL __declspec(dllexport)
#else
#define VCO_DLL __declspec(dllimport)
#endif

class VCO_DLL cVCO
{
public:
	cVCO(
		// frame t
		double* frm_t,
		// frame t - vessel centerline 
		double* frm_vc_t,
		// frame t+1
		double* frm_tp1,
		// array index of frame t+1
		int ftpidx,
		// frame configuration: width and height
		int frm_w, int frm_h,
		// options
		bool bVer = false, char *spath = NULL);
	~cVCO();

	// ** structure **//
	// int form vessel point
	struct ves_vec2i
	{
		int x;
		int y;
	};

	// float form vessel point
	struct ves_vec2f
	{
		float x;
		float y;

	};


	// information for vessel feature point
	// type is information of end or juntion point
	struct ves_feat_info
	{
		int x;
		int y;
		int type; // 0 is end feature point, 1 is junction feature point
	};

// MEMBER FUNCTIONS
	// *********** RELATED TO MRF OPTIMIZATION *********** //
	//	% coded by syshin in MATLAB (160130) 
	//  % converted to C++ by kjNoh (160600)
	// *** *** //
	// MAIN FUNCTION FOR OBTAINING OPTIMAL LABELS USING MRF 
	void GetLabelsByMRFOpt(
		// INPUTS
		std::vector<cv::Point> &all_coors,
		cv::Mat &all_cands,
		cv::Mat &all_cands_dists,
		std::vector<std::vector<int>> &v_segm_to_all_coors,
		int ves_segm_num, int cand_num,
		// OUTPUTS
		double &mrf_energy,
		double **mrf_labels
		);
	// *** computeMRFCost: compute costs for mrf t-links and n-links
	void computeMRFCost(
		// INPUTS
		std::vector<cv::Point> &all_coors,
		cv::Mat &all_cands,
		cv::Mat &all_cands_dists,
		std::vector<std::vector<int>> &v_segm_to_all_coors,
		int ves_segm_num, int cand_num,
		// OUTPUTS
		// unary costs: unary costs, nLabel*nNode
		cv::Mat *o_unaryCost,
		// pairwise costs: pairwise costs, nEdge * 1, each element is size of (nLabel*nLabel) 
		std::vector<cv::Mat> *o_pairwiseCost,
		// map matrix: mapping indices for 'pairwiseCost', 'mapMat(i.j) = k' means that
		//		       there is an edge between node i & j and its costs are in pairwiseCost{ k }
		cv::Mat* o_mapMat);
	// *** ConstUnaryCosts: construct unary cost matrix and compute unary costs *** //
	//		perform thresholding on distances to exclude outliers, 
	//		remove redundent candidates and sort by distance, and compute cost
	cv::Mat ConstUnaryCosts(
		// INPUTS
		std::vector<cv::Point> &all_coors,
		cv::Mat &all_cands,
		cv::Mat &all_cands_dists,
		int cand_num,
		double unary_thre,
		double unary_trun_thre,
		double dummy_unary_cost);
	// *** compute pairwise costs *** //
	void ConstPairwiseCosts(
		// INPUTS
		std::vector<cv::Point> &all_coors,
		cv::Mat &all_cands,
		std::vector<std::vector<int>> &v_segm_to_all_coors,
		int nNode, int ves_segm_num,
		double dummy_pairwise_cost1,
		double dummy_pairwise_cost2,
		// OUTPUTS
		std::vector<cv::Mat> &pairwiseCost,
		cv::Mat &mapMat);
	// *** compute truncted pairwise cost for specific node pair *** //
	void GetTruncatedPairwiseCost(
	// INPUTs
		cv::Point coor1, cv::Point coor2, 
		cv::Mat cands1, cv::Mat cands2, 
		int dummy_pairwise_cost1, int dummy_pairwise_cost2, 
	// OUTPUTs
		cv::Mat *o_mapMat);

	// convert correspondence mapping matrix mapMat into sparse mapMat
	void GetSparseCorrespondenceMapMatrix(cv::Mat &mapMat, 
		double **s_mapMat, int &nm);
	// convert pairwise cost to array pairwise cost
	void GetArrayPairwiseCost(std::vector<cv::Mat> &pairwiseCost, double ***arrayPairwiseCost);
	// call trw-s function in library to run optimization
	void mrf_trw_s(double *u, int uw, int uh, double **p, double* m, int nm, int mw, int mh, /*int in_Method, int in_iter_max, int in_min_iter,*/
		double *e, double **s);
	//void GetIntervalCost();
	//void unique(cv::Mat inputMat, std::vector<cv::Point> *o_uniqueSotrtPts, std::vector<int> *o_uniqueSotrtIdx);
	// *********** END FUNCTIONS RELATED TO MRF OPTIMIZATION *********** //


	// *********** FUNCTIONS RELATED TO VCO ALGORITHM *********** //
	//VCO_EXPORTS void VesselCorrespondenceOptimization(cv::Mat img_t, cv::Mat img_tp1, cv::Mat bimg_t,
	//	cVCOParams p, std::string ave_path, int nextNum,
	//	cv::Mat* bimg_tp1, cv::Mat* bimg_tp1_post_processed, int fidx_tp1, char* savePath)
	void VesselCorrespondenceOptimization(
		/*double** arr_bimg_tp1, double** arr_bimg_tp1_post_processed, cv::Mat* tp1_postProc, cv::Mat* tp1_nonPostProc*/);
	// global chamfer matching of frame t vessel centerlines to that of frame t+1
	void globalChamferMatching(
	// INPUTS
		// t_th frame binary centerline mask, 
		cv::Mat &bimg_t,
		// t+1_th frame
		cv::Mat &img_tp1,
		// t+1_th frame binary centerline mask (estimated)
		cv::Mat &bimg_tp1,
		// options
		bool b_use_gc, bool bVerbose, 	
	// OUTPUTS
		// matched vessel centerlines
		cv::Mat &gt_bimg_t,
		// displacement vector 
		int &t_x, int &t_y
		);
	// *** convert cell (vessel segment) based coordinates into containers that do not distinguish cells
	void ConstAllCoors(
		// INPUT
		// cell coordinates: cell for paths of each segment, ves_segm_num * 1 cell, ves_segm_num is the number of vessel segments
		//					 each segment has(nPT*d) values
		std::vector<std::vector<cv::Point>> &v_segm_pt_coors,
		// cell candidates: this contains candidate points per each (sampled) point
		std::vector<cv::Mat> &v_segm_pt_cands,
		// cell candidate distances: corresponding(unary) costs, ves_segm_num * 1 cell, ves_segm_num is the number of segments
		std::vector<cv::Mat> &v_segm_pt_cands_d,
		// OUTPUT
		std::vector<cv::Point> &all_coors,
		cv::Mat &all_cands,
		cv::Mat &all_cands_dists,
		std::vector<std::vector<int>> &v_segm_to_all_coors
		);


	void mkNewPath(
		// INPUTS
		cv::Mat m_frangi_vesselness_tp1, 
		std::vector<std::vector<int>> v_segm_to_all_coors, 
		cv::Mat all_cands,
		std::vector<std::vector<int>> all_joining_seg, 
		double* labels, 
		int ves_segm_num, 
		int num_all_joining_seg,

		//OUTPUTS
		std::vector<std::vector<cv::Point>> *newE, 
		std::vector<cv::Point> *all_v, 
		std::vector<cv::Point> *all_vessel_pt
		);


	//function[new_bimg, new_lidx, app_lidx] = GrowVesselUsingFastMarching(ivessel, lidx, thre)
	//% input
	//%
	//% ivessel : vesselness
	//% lidx : linear indices for vessels
	//% thre : threshold for 'ivessel', default 0.05
	//%
	//% output
	//%
	//% new_bimg : binary mask for a new vessel
	//% new_lidx : linear indices for a new vessels
	//% app_lidx : linear indices of appened parts
	//%
	//% coded by syshin(160305)
	cv::Mat postProcGrowVessel(cv::Mat img_tp1, cv::Mat m_frangi_vesselness_tp1, std::vector<cv::Point> all_vessel_pt,
		cVCOParams params, std::vector<std::vector<cv::Point>> *E);
	void GrowVesselUsingFastMarching(cv::Mat ivessel, std::vector<cv::Point> lidx, double thre, cVCOParams p,
		cv::Mat *o_new_bimg, std::vector<cv::Point> *o_new_lidx, std::vector<cv::Point> *o_app_lidx);
	
	void getBoundaryDistance(cv::Mat I, bool IS3D, cv::Mat *o_BoundaryDistance);
	void maxDistancePoint(cv::Mat BoundaryDistance, cv::Mat I, bool IS3D, cv::Point *o_posD, double *o_maxD);
	void GetLineLength(std::vector<cv::Point> L, bool IS3D, double *o_ll);
	void cvt2Arr(
		// INPUTS
		cv::Mat draw_bimg_tp1, 
		cv::Mat bimg_tp1_post_processed, 

		// OUTPUTS
		double **arr_tmp_bimg_tp1_nonPostproc, 
		double **arr_tmp_bimg_tp1_post_processed);

	
	// get to mask to cv::Mat form of post-post processing
	cv::Mat get_tp1_Mask();

	// get to mask to cv::Mat form of pre-post processing
	cv::Mat get_tp1_Mask_pp();

	// get to mask to double array point form of pre-post processing
	double* get_p_tp1_mask();
	
	// get to mask to double array point form of post-post processing
	double* get_p_tp1_mask_pp();

	// get to mask to unsigned char array point form of pre-post processing
	unsigned char* get_p_tp1_mask_8u();

	// get to mask to unsigned char array point form of post-post processing
	unsigned char* get_p_tp1_mask_pp_8u();

	// convert to all vectors form from segmente based vectors
	std::vector<cv::Point> makeSegvec2Allvec(
		//INPUT
		std::vector<std::vector<cv::Point>> segVec);

	std::vector<cv::Point> makeSegvec2Allvec(
		//INPUT
		std::vector<std::vector<cv::Point>> segVec,
		cv::Point tanslation
		);

	// convert to all vectors form from cv::Mat based form
	std::vector<cv::Point> makeSegvec2Allvec(
		//INPUT
		cv::Mat allMat);

	// make displacemente vectors
	std::vector<cv::Point> makeDisplacementVec(
		// INPUTS
		std::vector<cv::Point> pre, 
		std::vector<cv::Point> post
		);

	cv::Mat drawDisplacementeVec(cv::Mat img, std::vector<cv::Point> pre, std::vector<cv::Point> post, std::vector<cv::Point> dispVec);

	std::vector<std::vector<cv::Point>> eraseRepeatSegPts(std::vector<std::vector<cv::Point>> segVec,int nX, int nY);

	// get to feature points
	std::vector<ves_feat_info> getVesFeatPts();

	// set to feature points to our structure form
	void setVesFeatPts(
		// OUTPUTS
		std::vector<cv::Point> junction, 
		std::vector<cv::Point> end
		);

	// get to vessel segment vector points 2d array of pre-post processing
	std::vector<std::vector<cv::Point>> getVsegVpts2dArr();

	// get to vessel segment vector points 2d array of post-post processing
	std::vector<std::vector<cv::Point>> getVsegVpts2dArr_pp();

	// get to points that it is seleteced as uniform term of pre-post processing
	std::vector<cv::Point> get_t_vpts_arr();

	// get to points that it is seleteced as uniform term of post-post processing
	std::vector<cv::Point> get_tp1_vpts_arr_pp();

	// get to displacement vector to subtract tp1 frame points to t frame points
	std::vector<cv::Point> get_disp_vec_arr();


	// get to tp1 frame frangi filtered vesselness mask
	// get_tp1_FrangiVesselnessMask() function is returend cv::Mat form
	// get_tp1_p_FrangiVesselnessMask() function is retuned float pointer array form
	cv::Mat get_tp1_FrangiVesselnessMask();
	float* get_tp1_p_FrangiVesselnessMask();
	

	// *********** END FUNCTIONS RELATED TO VCO ALGORITHM *********** //

// MEMBER VARIABLES
// INPUTS
	// frame t
	double* arr_img_t;
	// frame t - vessel centerline 
	double* arr_bimg_t;
	// frame t+1
	double* arr_img_tp1;
	// array index of frame t+1
	int fidx_tp1;
	// frame configuration: width and height
	int img_w, img_h;
	
	// parameters
	cVCOParams params;
	bool bVerbose;
	char* savePath;

// OUTPUTS
	// Frangi filter results for frame t+1. 512x512, pixelwise float
	cv::Mat m_frangi_vesselness_tp1;
	float* m_p_frangi_vesselness_tp1;

	// vessel centerlines for frame t+1, in mask form, 512x512
	//cv::Mat m_tp1_vmask; // result of mask form to calculate post processing
	//cv::Mat m_tp1_vmask_pp; // result of mask form to previous post processing
	double *m_p_tp1_vmask; // result of array form to previous psot processing
	double *m_p_tp1_vmask_pp; // result of array form to calculate psot processing

	// vessel centerlines for frame t+1, in array form, of cv::Point s, int
	std::vector<std::vector<cv::Point>> m_tp1_vsegm_vpt_2darr; // stored each segmentes of previous to post processing
	std::vector<std::vector<cv::Point>> m_tp1_vsegm_vpt_2darr_pp;// stored each segmentes of post to post processing

	// vessel motion estimation, (displacement vectors for vessel centerline points), cv::Point2f array
	std::vector<cv::Point> m_t_vpt_arr;// candidates from t frame
	std::vector<cv::Point> m_tp1_vpt_arr;// candidates from tp1 frame
	std::vector<cv::Point> m_disp_vec_arr;// displacemente vector to subtract tp1 frame candidates vetors to t frame candidates vectors

	// vessel feature points for frame t+1, cv::Point array
	std::vector<ves_feat_info> m_feat_pts;

	// *** TODO *** //
	// vessel segmentation mask (512x512, pixelwise labels)

	// vessel pixel orientation, boundary




	

	
};

