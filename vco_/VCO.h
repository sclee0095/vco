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

// ** structure **//
// vessel feature point 
// type = information of end or juntion point
struct VCO_DLL ves_feat_pt
{
	int x;
	int y;
	int type; // 0 is end feature point, 1 is junction feature point
};

class VCO_DLL cVCO
{
	// ***** USER CALLABLE FUNCTIONS ***** //
public: 
	// *** TO RUN VCO: 
	//	1. USING CONSTRUCTOR
	//		- input frame t + vessel centerline of frame t, and frame t+1, frame index, frame size (width + height)
	//		- set options
	//	2. RUN FUNCTION VesselCorrespondenceOptimization()
	//		- GENERATED OUTPUTS: refer to line VCO OUTPUTS in MEMBER VARIABLES
	//  3. GET OUTPUT OF INTEREST USING GET FUNCTIONS
// CONSTRUCTOR & DESTRUCTOR
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
// MAIN VCO FUNCTION
	void VesselCorrespondenceOptimization();

// GET FUNCTIONS FOR OUTPUTS

	// coded by kjNoh (20160809)
	// * Frangi filtered vesselness mask for frame t+1
	// return cv::Mat form
	cv::Mat get_tp1_FrangiVesselnessMask();
	// return float pointer array form
	float* get_tp1_p_FrangiVesselnessMask();

	// coded by kjNoh (20160809)
	// get cv::Mat mask form of vessel centerline, after post processing
	cv::Mat get_tp1_vescl_mask();
	// get cv::Mat mask form of vessel centerline, before post processing
	cv::Mat get_tp1_vescl_mask_pp();
	// get unsigned char pointer mask form of vessel centerline, after post processing
	unsigned char* get_p_tp1_mask_8u();
	// get unsigned char pointer mask form of vessel centerline, before post processing
	unsigned char* get_p_tp1_mask_pp_8u();

	cv::Mat get_tp1_adjusted_vescl_mask_pp();

	// coded by kjNoh (20160809)
	// get feature points
	std::vector<ves_feat_pt> get_t_VesFeatPts();
	std::vector<ves_feat_pt> get_tp1_VesFeatPts();

	// coded by kjNoh (20160809)
	// find feature points for corresponding points of t frame
	std::vector<ves_feat_pt> find_tp1_features(std::vector<ves_feat_pt>  t_features, 
		std::vector<std::vector<cv::Point>> t_seg_vec,
		std::vector<std::vector<cv::Point>> tp1_seg_vec);
	std::vector<ves_feat_pt> find_tp1_features(std::vector<ves_feat_pt> t_features,
		std::vector<cv::Point> t_vseg,
		std::vector<cv::Point> tp1_vseg);

	// coded by kjNoh (20160809)
	// set feature points to our structure form
	std::vector<ves_feat_pt> setVesFeatPts(
		// INPUTS
		std::vector<cv::Point> junction, 
		std::vector<cv::Point> end,
		cv::Point tans
		);

	// coded by kjNoh (20160809)
	// get vessel segment vector points 2d array of pre-post processing
	std::vector<std::vector<cv::Point>> getVsegVpts2dArr();
	// get vessel segment vector points 2d array of post-post processing
	std::vector<std::vector<cv::Point>> getVsegVpts2dArr_pp();

	// coded by kjNoh (20160809)
	// get points that are seleteced as uniform term of pre-post processing
	std::vector<cv::Point> get_t_vpts_arr();
	// get points that are seleteced as uniform term of post-post processing
	std::vector<cv::Point> get_tp1_vpts_arr();
	// get displacement vector to subtract tp1 frame points to t frame points
	std::vector<cv::Point> get_disp_vec_arr();
	// draw displacement vector
	cv::Mat drawDisplacementeVec(cv::Mat img, std::vector<cv::Point> pre, std::vector<cv::Point> post, std::vector<cv::Point> dispVec);

	// coded by khNoh (20160814)
	// get information to linked each segmentation at after post processed t+1 frame
	// note : this is not tree. Just linked information to relationship at each segmentation
	std::vector<std::vector<std::vector<std::vector<int>>>> get_tp1_segm_linked_information();

	// *** 

	// *********** END FUNCTIONS RELATED TO VCO ALGORITHM *********** //

// MEMBER VARIABLES
// VCO INPUTS
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

// VCO OUTPUTS
	// Frangi filter results for frame t+1. 512x512, pixelwise float
	cv::Mat m_frangi_vesselness_tp1;
	float* m_p_frangi_vesselness_tp1;

	// vessel centerlines for frame t+1, in 2D array form, of [cv::Point]s, point coordinates are integers
	//  dim 1: vessel segments = vessel points between nodes (=feature points=end points+junctions)
	//	dim 2: vessel point coordinate array within each segment
	// * post processing = extension of vessel centerlines corresponding to contrast agent influx
	std::vector<std::vector<cv::Point>> m_tp1_vsegm_vpt_2darr; // before post processing
	std::vector<std::vector<cv::Point>> m_tp1_vsegm_vpt_2darr_pp; // after post processing

	// subsampled vessel centerline points for frame t and t+1, in 1D array form, of [cv::Point]s, point coordinates are integers
	//  = points used in VCO, 
	//	* no deliniation between points from different vessel segments, for easy input to MRF optimizer
	// uniformly subsampled points from frame t centerline
	std::vector<cv::Point> m_t_vpt_arr;
	// points corresponding to m_t_vpt_arr in frame t+1, 
	// * if point x and y are over 1000, no corresponding point (=dummy label) * //
	std::vector<cv::Point> m_tp1_vpt_arr;
	// vessel motion estimation = displacement vectors = m_tp1_vpt_arr - m_t_vpt_arr
	// * if point x and y are over 1000, no estimated motion (=dummy label) * //
	std::vector<cv::Point> m_disp_vec_arr;

	// vessel feature points for frame t+1, cv::Point array
	// * feature points = vessel segment end points + vessel junctions
	std::vector<ves_feat_pt> m_t_feat_pts;
	std::vector<ves_feat_pt> m_tp1_feat_pts; 

	// coded by khNoh (20160814)
	// information to linked each segmentation at after post processed t+1 frame
	std::vector<std::vector<std::vector<std::vector<int>>>> m_tp1_vsegm_linked_information;
	// boundary range is between end boundary and term of boundary range 
	int boundaryRange;

	// *** TODO *** //
	// vessel segmentation mask (512x512, pixelwise labels)

	// vessel pixel orientation, boundary
// VCO OUTPUTS


private:
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
	// *********** END FUNCTIONS RELATED TO MRF OPTIMIZATION *********** //


	// *********** FUNCTIONS RELATED TO VCO ALGORITHM *********** //
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

	// *** construct connected centerline path by reconnecting subsampled correspondence points
	void MakeConnectedCenterlineFromSubsampledPts(
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
		std::vector<cv::Point> *all_vessel_pt,
		std::vector<cv::Point> *tp1_vpts_all_subsample, 
		std::vector<std::vector<cv::Point>> *tp1_vpts_seg_subsample
		);

	// ** POST PROCESSING FUNCTIONS ** //

	// coded by syshin in MATLAB (160130) 
	// converted to C++ by kjNoh (160600)
	// post processing main function //
	// It can get center line mask to grown up or shrinked up 
	// output is center line mask. cv::Mat form(512x512)
	cv::Mat postProcGrowVessel(
		//INPUTS
		cv::Mat img_tp1, 
		cv::Mat m_frangi_vesselness_tp1,
		std::vector<cv::Point> all_vessel_pt,
		cVCOParams params,
		// INPUT AND OUTPUT
		std::vector<std::vector<cv::Point>> *E
		);

	// converted to C++ by kjNoh (160800)
	// generate new center line to grown using fastmarching method
	void GrowVesselUsingFastMarching(
		// INPUTS
		cv::Mat ivessel, 
		std::vector<cv::Point> lidx,
		double thre, 
		cVCOParams p,
		cv::Mat *o_new_bimg, 
		std::vector<cv::Point> *o_new_lidx, 
		std::vector<cv::Point> *o_app_lidx
		);
	// compute distance from boundary to seed points
	void getBoundaryDistance(cv::Mat I, bool IS3D, cv::Mat *o_BoundaryDistance);
	// get max distance point in boundary distance map
	void maxDistancePoint(cv::Mat BoundaryDistance, cv::Mat I, bool IS3D, cv::Point *o_posD, double *o_maxD);
	// compute length of input line(std::vector<cv::Point> L)
	void GetLineLength(std::vector<cv::Point> L, bool IS3D, double *o_ll);

	// ** END POST PROCESSING FUNCTIONS ** //

	// coded by kjNoh (160809)
	// convert to all vectors form from cv::Mat based form
	std::vector<cv::Point> makeSegvec2Allvec(
		std::vector<std::vector<cv::Point>> segVec
		);
	std::vector<cv::Point> makeSegvec2Allvec(
		//INPUTS
		std::vector<std::vector<cv::Point>> segVec, 
		cv::Point tanslation);
	// make displacemente vectors
	std::vector<cv::Point> makeDisplacementVec(
		// INPUTS
		std::vector<cv::Point> pre,
		std::vector<cv::Point> post
		);

	// coded by kjNoh (160814)
	// make information of linking relation at each segmentation
	std::vector<std::vector<std::vector<std::vector<int>>>>  linkedSeg(
		//INPUT
		std::vector<std::vector<cv::Point>> seg
		);
	// ***** END OF INTERNAL FUNCTIONS ***** //
};

