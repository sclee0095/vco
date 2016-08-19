#include "VCO.h"

typedef int mwSize;
typedef int mwIndex;




cVCO::cVCO(double* frm_t, double* frm_vc_t,
	double* frm_tp1, int ftpidx,
	int frm_w, int frm_h,
	// options
	bool bVer, char *spath)
	
{
	// frame t
	arr_img_t = frm_t;
	// frame t - vessel centerline 
	arr_bimg_t = frm_vc_t;
	// frame t+1
	arr_img_tp1 = frm_tp1;
	// array index of frame t+1
	fidx_tp1 = ftpidx;
	// frame configuration: width and height
	img_w = frm_w;  img_h = frm_h;
	// options
	bVerbose = bVer;
	savePath = spath;

	// parameters
	params = cVCOParams();

	m_t_vpt_arr = std::vector<cv::Point>();
	m_tp1_vpt_arr = std::vector<cv::Point>();
	m_disp_vec_arr = std::vector<cv::Point>();

	// coded by kjNoh (160814)
	// fixed boundary rage
	boundaryRange = 12;
}


cVCO::~cVCO()
{
}

// compute costs for mrf t-links and n-links
//	% coded by syshin in MATLAB (160130) 
//  % converted to C++ by kjNoh (160600)
void cVCO::computeMRFCost(
// INPUTS
	std::vector<cv::Point> &all_coors,
	cv::Mat &all_cands,
	cv::Mat &all_cands_dists,
	std::vector<std::vector<int>> &v_segm_to_all_coors,
	int ves_segm_num, int cand_num,
	// parameters
	//cVCOParams p,
// OUTPUTS
	// unary costs: unary costs, nLabel*nNode
	cv::Mat *o_unaryCost,
	// pairwise costs: pairwise costs, nEdge * 1, each element is size of (nLabel*nLabel) 
	std::vector<cv::Mat> * o_pairwiseCost, 
	// map matrix: mapping indices for 'pairwiseCost', 'mapMat(i.j) = k' means that
	//		       there is an edge between node i & j and its costs are in pairwiseCost{ k }
	cv::Mat* o_mapMat)
{
	// *** initialization *** //
	// default params
	int unary_thre = 800;
	int unary_trun_thre = 750;
	int dummy_unary_cost = unary_thre;
	int dummy_pairwise_cost1 = 75;
	int dummy_pairwise_cost2 = 4 * dummy_pairwise_cost1;
	double dist_sigma = params.sampling_period / 3.0;
	double alpha = 0.5;
	double beta = 0.5;

	// numbers of (vessel) segments, candidates, labels
	int nLabel = cand_num + 1;
	// *** end of initialization *** //

	// *** compute unary costs *** //
	//	perform thresholding on distances to exclude outliers, 
	//	remove redundent candidates and sort by distance, and compute cost
	cv::Mat unaryCost = ConstUnaryCosts(all_coors, all_cands, all_cands_dists,
		cand_num, unary_thre, unary_trun_thre, dummy_unary_cost);

	// *** compute pairwise costs *** //
	int nNode = all_coors.size();
	std::vector<cv::Mat> pairwiseCost;
	cv::Mat mapMat = cv::Mat::zeros(nNode, nNode, CV_64FC1);
	ConstPairwiseCosts(all_coors, all_cands, v_segm_to_all_coors,
		nNode, ves_segm_num, dummy_pairwise_cost1, dummy_pairwise_cost2,
		pairwiseCost, mapMat);

	*o_unaryCost = unaryCost;
	*o_pairwiseCost = pairwiseCost;
	*o_mapMat = mapMat;
}


// *** construct all_variables: containers that do not distinguish between vessel segments ***
void cVCO::ConstAllCoors(
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
	)
{
	int ves_segm_num = v_segm_pt_coors.size();

	for (int i = 0; i < v_segm_pt_coors[0].size(); i++) {
		all_coors.push_back(v_segm_pt_coors[0][i]);
	}

	v_segm_pt_cands[0].copyTo(all_cands);
	//std::vector<cv::Mat> all_cands
	//all_cands.push_back(v_segm_pt_cands[0]);

	v_segm_pt_cands_d[0].copyTo(all_cands_dists);
	all_cands_dists.convertTo(all_cands_dists, CV_64FC1);
	//all_cands_dists.push_back(v_segm_pt_cands_d[0]);

	for (int i = 0; i < v_segm_pt_coors[0].size(); i++)	{
		v_segm_to_all_coors[0].push_back(i + 1);
	}
	//v_segm_to_all_coors[0] = [1:size(v_segm_pt_coors{ 1 }, 1)]';

	for (int i = 1; i < ves_segm_num; i++) {
		std::vector<cv::Point> temp = v_segm_pt_coors[i];

		std::vector<int> is_in_all_coors(temp.size());
		for (int j = 0; j < temp.size(); j++) {
			is_in_all_coors[j] = 0;
			for (int k = 0; k < all_coors.size(); k++) {
				if (temp[j] == all_coors[k]) {
					is_in_all_coors[j] = 1;
					break;
				}
			}
		}
		//[is_in_all_coors, all_coors_idx] = ismember(temp, all_coors, 'rows');

		int cnt_zero = 0;
		for (int j = 0; j < temp.size(); j++) {
			if (!is_in_all_coors[j]) {
				all_coors.push_back(temp[j]);
				cnt_zero++;
			}
		}
		//all_coors = [all_coors; temp(~is_in_all_coors, :)];

		cv::Mat new_line = cv::Mat::zeros(cnt_zero, all_cands.cols, CV_32SC2);
		new_line = -1;
		//new_line = zeros(nnz(~is_in_all_coors), size(all_cands, 2));

		int cnt = 0;
		for (int j = 0; j < temp.size(); j++) {
			if (!is_in_all_coors[j]) {
				v_segm_pt_cands[i].row(j).copyTo(new_line.row(cnt));
				cnt++;
			}
		}
		//new_line.at<double>(:, 1 : cand_num) = v_segm_pt_cands{ i }(~is_in_all_coors, :);

		for (int j = 0; j < new_line.rows; j++)
			all_cands.push_back(new_line.row(j));
		//all_cands = [all_cands; new_line];

		new_line = cv::Mat::zeros(cnt_zero, all_cands.cols, CV_64FC1);
		new_line = INFINITY;
		//new_line = inf(nnz(~is_in_all_coors), size(all_cands_dists, 2));

		cnt = 0;
		for (int j = 0; j < temp.size(); j++) {
			if (!is_in_all_coors[j]) {
				v_segm_pt_cands_d[i].row(j).copyTo(new_line.row(cnt));
				cnt++;
			}
		}
		//new_line(:, 1 : cand_num) = v_segm_pt_cands_d{ i }(~is_in_all_coors, :);

		for (int j = 0; j < new_line.rows; j++) {
			all_cands_dists.push_back(new_line.row(j));
			//all_cands_dists.push_back((double*)new_line.row(j).data);
			//all_cands_dists.push_back(new_line.col(j));
		}
		//all_cands_dists = [all_cands_dists; new_line];

		std::vector<int> all_coors_idx(temp.size());
		for (int j = 0; j < temp.size(); j++) {
			all_coors_idx[j] = 0;
			for (int k = 0; k < all_coors.size(); k++) {
				if (temp[j] == all_coors[k]) {
					all_coors_idx[j] = k + 1;
					break;
				}
			}
		}
		//[is_in_all_coors, all_coors_idx] = ismember(temp, all_coors, 'rows');

		v_segm_to_all_coors[i] = (all_coors_idx);
		//v_segm_to_all_coors{ i } = all_coors_idx;
	}

	/*
	// TO VERIFY RESULTS
	int cand_num = v_segm_pt_cands[0].cols;
	temp = all_cands_dists(:, cand_num + 1 : size(all_cands_dists, 2));
	temp(temp == 0) = inf;
	all_cands_dists(:, cand_num + 1 : size(all_cands_dists, 2)) = temp;

	for (int y = 0; y < all_cands_dists.rows; y++) {
	for (int x = 0; x < all_cands_dists.cols; x++) {
		if (all_cands_dists.at<double>(y, x) >= 100000) {
			printf("%s  ", "inf");
		}
		else
			printf("%.1f  ", all_cands_dists.at<double>(y, x));
		}
		printf("\n");
	}
	printf("\n\n");
	*/
}


// *** ConstUnaryCosts: construct unary cost matrix and compute unary costs *** //
//		perform thresholding on distances to exclude outliers, 
//		remove redundent candidates and sort by distance, and compute cost
cv::Mat cVCO::ConstUnaryCosts(
// INPUTS
	std::vector<cv::Point> &all_coors,
	cv::Mat &all_cands,
	cv::Mat &all_cands_dists,
	int cand_num,
	double unary_thre, 
	double unary_trun_thre,
	double dummy_unary_cost)
// OUTPUT: cv::Mat unaryCost
{
	// *** perform thresholding on candidate distances to exclude outliers *** //
	int nNode = all_coors.size();
	for (int y = 0; y < all_cands_dists.rows; y++)
	for (int x = 0; x < all_cands_dists.cols; x++) {
		if (all_cands_dists.at<double>(y, x) > unary_thre) {
			all_cands_dists.at<double>(y, x) = INFINITY;
		}
	}
	//all_cands_dists(all_cands_dists > unary_thre) = inf;

	for (int y = 0; y < all_cands_dists.rows; y++)
	for (int x = 0; x < all_cands_dists.cols; x++) {
		if (all_cands_dists.at<double>(y, x) <= unary_thre & all_cands_dists.at<double>(y, x) > unary_trun_thre) {
			all_cands_dists.at<double>(y, x) = unary_trun_thre;
		}
	}
	//all_cands_dists(all_cands_dists <= unary_thre&all_cands_dists > unary_trun_thre) = unary_trun_thre;
	// ***

	// *** nullify outlier candidate coordinates
	for (int y = 0; y < all_cands.rows; y++)
	for (int x = 0; x < all_cands.cols; x++)	{
		if (all_cands_dists.at<double>(y, x) == INFINITY) {
			all_cands.at<cv::Point>(y, x) = cv::Point(-1, -1);
		}
	}
	//all_cands(all_cands_dists == inf) = 0;

	// *** sort candidates by distances and remove redundent candidates *** //
	cv::Mat temp_all_cands;
	all_cands.copyTo(temp_all_cands);
	cv::Mat temp_all_cands_dists;
	all_cands_dists.copyTo(temp_all_cands_dists);
	all_cands = cv::Mat::zeros(nNode, cand_num, CV_32SC2);
	all_cands = -1;
	all_cands_dists = cv::Mat::zeros(nNode, cand_num, CV_64FC1);
	all_cands_dists = INFINITY;

	for (int i = 0; i < nNode; i++) {
		std::vector<cv::Point> C;
		std::vector<int> v_idxC;
		std::vector<int> ia;

		cv::Mat cur_row_cand = temp_all_cands.row(i);

		for (int j = 0; j < cur_row_cand.cols; j++) {
			int cur_x = cur_row_cand.at<cv::Point>(j).x;
			int cur_y = cur_row_cand.at<cv::Point>(j).y;
			v_idxC.push_back(cur_y * 512 + cur_x);
		}
		std::sort(v_idxC.begin(), v_idxC.end());
		v_idxC.erase(std::unique(v_idxC.begin(), v_idxC.end()), v_idxC.end());

		for (int j = 0; j < v_idxC.size(); j++) {
			C.push_back(cv::Point(v_idxC[j] % 512, v_idxC[j] / 512));

			for (int k = 0; k < cur_row_cand.cols; k++) {
				if (C[j] == cur_row_cand.at<cv::Point>(k)) {
					ia.push_back(k);
					break;
				}
			}

		}
		//unique(temp_all_cands.row(i), &C, &ia);
		//[C, ia, ic] = unique(temp_all_cands(i, :));

		cv::Mat unique_cands;
		cv::Mat unique_cands_dists;
		if (C[0].x != -1) {
			unique_cands = cv::Mat(1, ia.size(), CV_32SC2);
			unique_cands_dists = cv::Mat(1, ia.size(), CV_64FC1);
			for (int j = 0; j < ia.size(); j++) {
				unique_cands.at<cv::Point>(0, j) = temp_all_cands.at<cv::Point>(i, ia[j]);
				unique_cands_dists.at<double>(0, j) = temp_all_cands_dists.at<double>(i, ia[j]);
			}
		}
		else {
			//unique_cands = temp_all_cands(i, ia(2:end));
			unique_cands = cv::Mat(1, ia.size() - 1, CV_32SC2);
			unique_cands_dists = cv::Mat(1, ia.size() - 1, CV_64FC1);

			for (int j = 1; j < ia.size(); j++) {
				unique_cands.at<cv::Point>(0, j - 1) = temp_all_cands.at<cv::Point>(i, ia[j]);
				unique_cands_dists.at<double>(0, j - 1) = temp_all_cands_dists.at<double>(i, ia[j]);
			}
			//unique_cands_dists = temp_all_cands_dists(i, ia(2:end));
		}
		for (int j = 0; j < unique_cands.cols; j++) {
			all_cands.at<cv::Point>(i, j) = unique_cands.at<cv::Point>(0, j);
		}
		for (int j = 0; j < unique_cands_dists.cols; j++) {
			all_cands_dists.at<double>(i, j) = unique_cands_dists.at<double>(0, j);
		}
		//all_cands(i, 1:length(unique_cands)) = unique_cands;
		//all_cands_dists(i, 1:length(unique_cands_dists)) = unique_cands_dists;
	}
	// *** END of candidate sorting and redundency check *** //

	cv::Mat unaryCost;
	all_cands_dists.copyTo(unaryCost);
	cv::transpose(unaryCost, unaryCost);
	cv::Mat dummy(1, unaryCost.cols, CV_64FC1);
	dummy = 1 * dummy_unary_cost;
	unaryCost.push_back(dummy);
	return unaryCost;

	// to verify computed unary values (MATLAB code)
	/*
	unaryCost = [all_cands_dists';dummy_unary_cost*ones(1,nNode)];
	for (int y = 0; y < temp_all_cands_dists.rows; y++)
	{
		for (int x = 0; x < temp_all_cands_dists.cols; x++)
		{
			if (temp_all_cands_dists.at<double>(y, x) >= 1000000)
			{
				printf("%s  ", "inf");
			}
			else
				printf("%.1f  ", temp_all_cands_dists.at<double>(y, x));
		}
		printf("\n");
	}
	printf("\n\n");
	*/
}

// *** compute pairwise costs *** //
void cVCO::ConstPairwiseCosts(
// INPUTS
	std::vector<cv::Point> &all_coors,
	cv::Mat &all_cands,
	std::vector<std::vector<int>> &v_segm_to_all_coors,
	int nNode, int ves_segm_num,
	double dummy_pairwise_cost1, 
	double dummy_pairwise_cost2,
// OUTPUTS
	std::vector<cv::Mat> &pairwiseCost,
	cv::Mat &mapMat)
{
	int nEdge = 0;
	int nIntraEdge = 0;
	/// for each segment
	for (int i = 0; i < ves_segm_num; i++)
	{
		std::vector<int> t_coors_to_all_coors = v_segm_to_all_coors[i];
		for (int j = 0; j < t_coors_to_all_coors.size() - 1; j++)
		{
			cv::Mat t_pCost1;
			GetTruncatedPairwiseCost(
				all_coors[t_coors_to_all_coors[j] - 1], 
				all_coors[t_coors_to_all_coors[j + 1] - 1],
				all_cands.row(t_coors_to_all_coors[j] - 1), 
				all_cands.row(t_coors_to_all_coors[j + 1] - 1), 
				dummy_pairwise_cost1, dummy_pairwise_cost2, &t_pCost1);

			cv::Mat t_pCost = t_pCost1;

			nEdge = nEdge + 1;
			nIntraEdge = nIntraEdge + 1;

			pairwiseCost.push_back(t_pCost);
			mapMat.at<double>(t_coors_to_all_coors[j] - 1, t_coors_to_all_coors[j + 1] - 1) = nEdge;
		}
	}
}

// *** compute truncted pairwise cost for specific node pair *** //
void cVCO::GetTruncatedPairwiseCost(
// INPUT	
	cv::Point coor1, cv::Point coor2, 
	cv::Mat cands1, cv::Mat cands2, 
	int dummy_pairwise_cost1, int dummy_pairwise_cost2, 
// OUTPUT
	cv::Mat *o_mapMat)
//function cost_mat = GetTruncatedPairwiseCost(coor1, coor2, cands1, cands2, dummy_pairwise_cost1, dummy_pairwise_cost2)
{

	int nY = 512; int nX = 512;
	int cand_num = cands1.cols;
	int nLabel = cand_num + 1;
	std::vector<cv::Point> cands1_xxyy, cands2_xxyy;
	for (int i = 0; i < cands1.cols; i++) {
		cands1_xxyy.push_back(cands1.at<cv::Point>(0,i));
	}
	for (int i = 0; i < cands2.cols; i++) {
		cands2_xxyy.push_back(cands2.at<cv::Point>(0, i));
	}
	//[cands1_yy, cands1_xx] = ind2sub([nY, nX], cands1);
	//[cands2_yy, cands2_xx] = ind2sub([nY, nX], cands2);

	std::vector<double> diff1_y(cands1_xxyy.size()), diff1_x(cands1_xxyy.size()), diff2_y(cands1_xxyy.size()), diff2_x(cands1_xxyy.size());
	for (int i = 0; i < cands1_xxyy.size(); i++) {
		diff1_y[i] = cands1_xxyy[i].y - coor1.y;
		diff1_x[i] = cands1_xxyy[i].x - coor1.x;
		diff2_y[i] = cands2_xxyy[i].y - coor2.y;
		diff2_x[i] = cands2_xxyy[i].x - coor2.x;
	}
	//diff1_y = cands1_yy - coor1(1);
	//diff1_x = cands1_xx - coor1(2);
	//diff2_y = cands2_yy - coor2(1);
	//diff2_x = cands2_xx - coor2(2);

	cv::Mat repmat1(1, cand_num, CV_64FC1); cv::Mat repmat2(1, cand_num, CV_64FC1);
	for (int i = 0; i < repmat1.cols; i++) {
		repmat1.at<double>(i) = diff2_y[i];
		repmat2.at<double>(i) = diff1_y[i];
	}

	//for (int x = 0; x < repmat1.cols; x++)
	//{
	//	printf("%0.1f ", repmat1.at<double>(0, x));
	//}
	//printf("\n\n");

	for (int i = 0; i < cand_num-1; i++)	{
		repmat1.push_back(repmat1.row(0));
		repmat2.push_back(repmat2.row(0));
	}
	//repmat1.copyTo(repmat2);

	cv::transpose(repmat2, repmat2);

	cv::Mat diff_y = repmat1 - repmat2;
	//for (int y = 0; y < diff_y.rows; y++)
	//{
	//	for (int x = 0; x < diff_y.cols; x++)
	//	{
	//		printf("%0.1f ", diff_y.at<double>(y,x));
	//	}
	//	printf("\n");
	//}
	//printf("\n\n");
	//diff_y = repmat(diff2_y, cand_num, 1) - repmat(diff1_y',1,cand_num);

	repmat1 = cv::Mat(1, cand_num, CV_64FC1);
	repmat2 = cv::Mat(1, cand_num, CV_64FC1);
	for (int i = 0; i < repmat1.cols; i++) {
		repmat1.at<double>(i) = diff2_x[i];
		repmat2.at<double>(i) = diff1_x[i];
	}

	for (int i = 0; i < cand_num - 1; i++) {
		repmat1.push_back(repmat1.row(0));
		repmat2.push_back(repmat2.row(0));
	}

	//repmat1.copyTo(repmat2);

	cv::transpose(repmat2, repmat2);

	cv::Mat diff_x = repmat1 - repmat2;
	//diff_x = repmat(diff2_x, cand_num, 1) - repmat(diff1_x',1,cand_num);

	cv::Mat diff(cand_num, cand_num, CV_64FC1);
	for (int y = 0; y < diff_x.rows; y++)
	for (int x = 0; x < diff_x.cols; x++)
	{
		diff.at<double>(y,x) = std::sqrt(diff_x.at<double>(y, x)*diff_x.at<double>(y, x) + diff_y.at<double>(y, x)*diff_y.at<double>(y, x));
	}
	//diff = sqrt(diff_x. ^ 2 + diff_y. ^ 2);

	cv::Mat cost_mat(nLabel, nLabel, CV_64FC1);
	cost_mat = 1 * dummy_pairwise_cost1;
	//cost_mat = dummy_pairwise_cost1*ones(nLabel, nLabel);
	diff = diff * 10;
	
	for (int y = 0; y < diff.rows; y++)
	for (int x = 0; x < diff.cols; x++) {
		if (diff.at<double>(y, x) > dummy_pairwise_cost2)
		{
			diff.at<double>(y, x) = dummy_pairwise_cost2;
		}
	}
	//diff(diff > dummy_pairwise_cost2) = dummy_pairwise_cost2;

	for (int i = 0; i < cands1.cols; i++) {
		if (cands1.at<cv::Point>(i).x == -1) {
			for (int j = 0; j < diff.rows; j++) {
				diff.at<double>(i,j) = INFINITY;
			}
		}
	}

	for (int i = 0; i < cands2.cols; i++) {
		if (cands2.at<cv::Point>(i).x == -1) {
			for (int j = 0; j < diff.rows; j++) {
				diff.at<double>(j, i) = INFINITY;
			}
		}
	}


	for (int y = 0; y < cand_num; y++)
	for (int x = 0; x < cand_num; x++) {
		cost_mat.at<double>(y, x) = diff.at<double>(y, x);
	}
	//diff(cands1 == 0, :) = inf; % needless
	//diff(:, cands2 == 0) = inf; % needless
	//cost_mat(1:cand_num, 1 : cand_num) = diff;

	*o_mapMat = cost_mat;
}

// convert correspondence mapping matrix mapMat into sparse mapMat
void cVCO::GetSparseCorrespondenceMapMatrix(cv::Mat &mapMat, double **s_mapMat, int &nm)
{
	nm = 0;
	for (int y = 0; y < mapMat.rows; y++) {
		for (int x = 0; x < mapMat.cols; x++) {
			if (mapMat.at<double>(y, x)) {
				nm++;
			}
		}
	}

	double *sparse_mapMat = new double[nm * 3];
	int nonZeroCnt = 0;
	for (int y = 0; y < mapMat.rows; y++)
	for (int x = 0; x < mapMat.cols; x++) {
		if (mapMat.at<double>(y, x) && !nonZeroCnt) {
			sparse_mapMat[nonZeroCnt * 3 + 0] = x;
			sparse_mapMat[nonZeroCnt * 3 + 1] = y;
			sparse_mapMat[nonZeroCnt * 3 + 2] = mapMat.at<double>(y, x);
			nonZeroCnt++;
		}
		else if (mapMat.at<double>(y, x)) {
			sparse_mapMat[nonZeroCnt * 3 + 0] = x;
			sparse_mapMat[nonZeroCnt * 3 + 1] = y;
			sparse_mapMat[nonZeroCnt * 3 + 2] = mapMat.at<double>(y, x);
			nonZeroCnt++;
		}
	}
	*s_mapMat = sparse_mapMat;
	//mapMat = sparse(mapMat);	
}

// convert pairwise cost to array pairwise cost
void cVCO::GetArrayPairwiseCost(std::vector<cv::Mat> &pairwiseCost, double ***arrPairwiseCost)
{
	double **arrayPairwiseCost = new double*[pairwiseCost.size()];
	for (int i = 0; i < pairwiseCost.size(); i++) {
		arrayPairwiseCost[i] = new double[pairwiseCost[i].rows*pairwiseCost[i].cols];

		for (int x = 0; x < pairwiseCost[i].cols; x++)
		for (int y = 0; y < pairwiseCost[i].rows; y++) {
			arrayPairwiseCost[i][x* pairwiseCost[i].cols + y] = pairwiseCost[i].at<double>(y, x);
		}
	}
	*arrPairwiseCost = arrayPairwiseCost;
}

void cVCO::mrf_trw_s(double *u, int uw, int uh, double **p,
	double* m, int nm, int mw, int mh,
	/*int in_Method, int in_iter_max, int in_min_iter,*/
	double *e, double **s)
{
	//prepare default options
	MRFEnergy<TypeGeneral>::Options options;
	options.m_eps = 1e-2;
	options.m_iterMax = 20;
	options.m_printIter = 5;
	options.m_printMinIter = 10;
	int verbosityLevel = 0;
	int method = 0;

	int numNodes = uh;
	int numLabels = uw;

	double* termW = u;

	//create MRF object
	MRFEnergy<TypeGeneral>* mrf;
	MRFEnergy<TypeGeneral>::NodeId* nodes;
	TypeGeneral::REAL energy, lowerBound;

	TypeGeneral::REAL *D = new TypeGeneral::REAL[numLabels];
	TypeGeneral::REAL *P = new TypeGeneral::REAL[numLabels * numLabels];
	for (int i = 0; i < numLabels * numLabels; ++i)
		P[i] = 0;

	mrf = new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize());
	nodes = new MRFEnergy<TypeGeneral>::NodeId[numNodes];

	// construct energy
	// add unary terms
	for (int i = 0; i < numNodes; ++i) {
		nodes[i] = mrf->AddNode(TypeGeneral::LocalSize(numLabels), 
								TypeGeneral::NodeData(termW + i * numLabels));
	}

	//add pairwise terms
	for (mwIndex c = 0; c < nm; ++c) {
		int y = m[c * 3 + 1];
		int x = m[c * 3 + 0];
		int edge_idx = m[c * 3 + 2];

		double* pCost = p[edge_idx - 1];

		//add matrix that is specified by user
		for (int i = 0; i < numLabels; ++i)
		for (int j = 0; j < numLabels; ++j)
			P[j + numLabels * i] = pCost[j + numLabels * i];

		mrf->AddEdge(nodes[y], nodes[x], TypeGeneral::EdgeData(TypeGeneral::GENERAL, P));
	}

	/////////////////////// TRW-S algorithm //////////////////////
	if (verbosityLevel < 2)
		options.m_printMinIter = options.m_iterMax + 2;

	clock_t tStart = clock();

	if (method == 0) //TRW-S
	{
		// Function below is optional - it may help if, for example, nodes are added in a random order
		//mrf->SetAutomaticOrdering();
		lowerBound = 0;
		mrf->Minimize_TRW_S(options, lowerBound, energy);

		if (verbosityLevel >= 1)
			printf("TRW-S finished. Time: %f\n", (clock() - tStart) * 1.0 / CLOCKS_PER_SEC);
	}
	else
	{
		// Function below is optional - it may help if, for example, nodes are added in a random order
		//mrf->SetAutomaticOrdering();

		mrf->Minimize_BP(options, energy);
		lowerBound = std::numeric_limits<double>::signaling_NaN();

		if (verbosityLevel >= 1)
			printf("BP finished. Time: %f\n", (clock() - tStart) * 1.0 / CLOCKS_PER_SEC);
	}

	//output the best energy value
	*e = (double)energy;
	//output the best solution
	double* segment = new double[numNodes];
	for (int i = 0; i < numNodes; ++i)
		segment[i] = (double)(mrf->GetSolution(nodes[i])) + 1;
	*s = segment;

	//output the best lower bound
	//if (lbOutPtr != NULL)	{
	//	*lbOutPtr = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
	//	*(double*)mxGetData(*lbOutPtr) = (double)lowerBound;
	//}

	// done
	delete[] nodes;
	delete mrf;
	delete[] D;
	delete[] P;
}


void cVCO::GetLabelsByMRFOpt(
// INPUTS
	std::vector<cv::Point> &all_coors,
	cv::Mat &all_cands,
	cv::Mat &all_cands_dists,
	std::vector<std::vector<int>> &v_segm_to_all_coors,
	int ves_segm_num, int cand_num, 
// OUTPUTS
	double &mrf_energy,
	double **mrf_labels
)
{
	cv::Mat unaryCost;
	std::vector<cv::Mat> pairwiseCost;
	cv::Mat mapMat;
	computeMRFCost(all_coors, all_cands, all_cands_dists, v_segm_to_all_coors, ves_segm_num, cand_num, 
		&unaryCost, &pairwiseCost, &mapMat);
	cv::transpose(unaryCost, unaryCost);


	// convert outputs to appropriate forms
	int nm;
	double* sparse_mapMat;
	GetSparseCorrespondenceMapMatrix(mapMat, &sparse_mapMat, nm);

	double **arrayPairwiseCost = new double*[pairwiseCost.size()];
	GetArrayPairwiseCost(pairwiseCost, &arrayPairwiseCost);

	// perform optimization
	double energy;
	double *labels;
	mrf_trw_s(((double*)unaryCost.data), unaryCost.cols, unaryCost.rows,
		arrayPairwiseCost, sparse_mapMat, nm, mapMat.cols, mapMat.rows, &energy, &labels);

	for (int i = 0; i < pairwiseCost.size(); i++) {
		delete[] arrayPairwiseCost[i];
	}
	delete[] sparse_mapMat;

	mrf_energy = energy;
	*mrf_labels = labels;
}

//VCO_EXPORTS void VesselCorrespondenceOptimization(cv::Mat img_t, cv::Mat img_tp1, cv::Mat bimg_t,
//	cVCOParams p, std::string ave_path, int nextNum,
//	cv::Mat* bimg_tp1, cv::Mat* bimg_tp1_post_processed, int fidx_tp1, char* savePath)
void cVCO::VesselCorrespondenceOptimization(
	/*double** arr_bimg_tp1, double** arr_bimg_tp1_post_processed, cv::Mat *tp1_postProc, cv::Mat *tp1_nonPostProc*/)
{
	char str[200];

	// *** initialize t-frame - original frame *** //
	cv::Mat img_t(img_h, img_w, CV_64FC1, arr_img_t);
	img_t.convertTo(img_t, CV_8UC1);
	int nY = img_t.rows;
	int nX = img_t.cols;


	// *** perform Frangi filtering for t+1-frame *** //
	cv::Mat img_tp1(img_h, img_w, CV_64FC1, arr_img_tp1);
	img_tp1.convertTo(img_tp1, CV_8UC1);


	cFrangiFilter frangi;
	cv::Mat tmp_frangi_vesselness_tp1;
	
	m_frangi_vesselness_tp1 = frangi.frangi(img_tp1);
	
	m_p_frangi_vesselness_tp1 = new float[m_frangi_vesselness_tp1.rows*m_frangi_vesselness_tp1.cols];
	m_p_frangi_vesselness_tp1 = ((float*)m_frangi_vesselness_tp1.data);

	// *** END Frangi filtering *** //

	// *** generate binary centerline images for t-frame and t+1-frame*** //


	cP2pMatching p2p(18);
	// binary img of 't' frame
	cv::Mat bimg_t(img_h, img_w, CV_64FC1, arr_bimg_t);
	bimg_t.convertTo(bimg_t, CV_8UC1);
	p2p.thin(bimg_t, bimg_t);
	// binary img of 't+1' frame
	cv::Mat bimg_tp1;
	cv::Mat frangi_vesselness_tp1_32f;
	m_frangi_vesselness_tp1.convertTo(frangi_vesselness_tp1_32f, CV_32FC1);
	cv::threshold(frangi_vesselness_tp1_32f, bimg_tp1, params.thre_ivessel, 255, 0);
	frangi_vesselness_tp1_32f.release();
	bimg_tp1.convertTo(bimg_tp1, CV_8UC1);
	p2p.thin(bimg_tp1, bimg_tp1);
	// *** END of binary centerline images generation *** //

	// *** global chamfer matching *** //
	cv::Mat gc_bimg_t;
	int gc_t_x, gc_t_y;
	globalChamferMatching(bimg_t, img_tp1, bimg_tp1, params.use_global_chamfer, bVerbose,
		gc_bimg_t, gc_t_x, gc_t_y);
	// *** END of global chamfer matching *** //

	

	// *** point correspondence candidate searching *** //
	std::vector<std::vector<cv::Point>> v_segm_pt_coors;
	std::vector<cv::Mat> v_segm_pt_cands;
	std::vector<cv::Mat> v_segm_pt_cands_d;
	std::vector<cv::Point> J, end;
	//cv::Mat bJ;
	p2p.thin(gc_bimg_t, gc_bimg_t);
	p2p.run(img_t, img_tp1, gc_bimg_t, params, gc_t_x, gc_t_y, m_frangi_vesselness_tp1,
		fidx_tp1, savePath, bVerbose, &v_segm_pt_coors, &v_segm_pt_cands, &v_segm_pt_cands_d, &J, &end);

	m_t_feat_pts = setVesFeatPts(J, end, cv::Point(gc_t_x, gc_t_y));


	m_t_vpt_arr = makeSegvec2Allvec(v_segm_pt_coors,cv::Point(gc_t_x,gc_t_y));

	// * number of vessel segments
	int ves_segm_num = v_segm_pt_coors.size();
	// * cand_num = number of candidates for specific vessel point, FIXED TO params.n_cands * //
	int cand_num = v_segm_pt_cands[0].cols;
	// * construct all_variables: containers that do not distinguish between vessel segments *
	// copy, check consistency and establish index correspondences 
	// between	v_segm_pt_coors and all_coors, 
	//			v_segm_pt_cands and all_cands, 
	//			and v_segm_pt_cands_d and all_cands_dists 
	std::vector<cv::Point> all_coors;
	cv::Mat all_cands, all_cands_dists;
	std::vector<std::vector<int>> v_segm_to_all_coors(ves_segm_num);
	ConstAllCoors(v_segm_pt_coors, v_segm_pt_cands, v_segm_pt_cands_d,
		all_coors, all_cands, all_cands_dists, v_segm_to_all_coors);
	// *** END of point correspondence candidate searching *** //



	// *** determine optimal correspondnece points by MRF optimization *** //
	double energy, *labels;
	GetLabelsByMRFOpt(
		// INPUTS
		all_coors, all_cands, all_cands_dists, v_segm_to_all_coors, ves_segm_num, cand_num,
		// OUTPUTS
		energy, &labels);
	// *** END of MRF optimization *** //

	// *** connect all output vco points of the t+1_th frame *** //
	// * get all "feature" (bifurcations + crossings + end) point indices * //
	cv::Mat bJ_all_pts = cv::Mat::zeros(all_coors.size(), 1, CV_8UC1);
	for (int j = 0; j < ves_segm_num; j++) {
		for (int k = 0; k < J.size(); k++) {
			if (v_segm_pt_coors[j][0] == J[k]) {
				bJ_all_pts.at<uchar>(v_segm_to_all_coors[j][0] - 1) = true;
				break;
			}
		}

		int n_j_segm = v_segm_pt_coors[j].size();
		int n_j_c2a = v_segm_to_all_coors[j].size();
		for (int k = 0; k < J.size(); k++)	{
			if (v_segm_pt_coors[j][n_j_segm - 1] == J[k]) {
				bJ_all_pts.at<uchar>(v_segm_to_all_coors[j][n_j_c2a - 1] - 1) = true;
				break;
			}
		}

	}
	std::vector<cv::Point> bJ_idx;
	cv::findNonZero(bJ_all_pts, bJ_idx);

	// * check if there are dummy labels for "feature" points, 
	//   for a dummy labeled feature point, find alternative feature point for corresponding vessel segments * //
	std::vector<std::vector<int>> all_joining_seg;
	int num_all_joining_seg = 0;
	for (int j = 0; j < bJ_idx.size(); j++) {
		// if the j_th feature point has been assigned a dummy label
		if (labels[bJ_idx[j].y * bJ_all_pts.cols + bJ_idx[j].x] == params.n_all_cands + 1) {
			// find segments joining at this junction
			std::vector<int> joining_seg;
			// find index of current "feature" point with dummy label WITHIN v_segm_pt_coors
			for (int k = 0; k < ves_segm_num; k++) {
				int idx = -1;
				for (int a = 0; a < v_segm_to_all_coors[k].size(); a++) {
					if (v_segm_to_all_coors[k][a] - 1 == 
						(bJ_idx[j].y * bJ_all_pts.cols + bJ_idx[j].x)) {
						idx = a;
					}
				}
				
				// find next point within same vessel segment that has not been assigned dummy label
				if (idx != -1) {
					int meet_pt = -1;
					if (idx == 0) {
						for (int a = 1; a < v_segm_to_all_coors[k].size(); a++) {
							if (labels[v_segm_to_all_coors[k][a] - 1] != params.n_all_cands + 1) {
								meet_pt = a;
								break;
							}
						}
						//meet_pt = find(labels(v_segm_to_all_coors{ k })~= params.n_all_cands + 1, 1, 'first');
					}
					else {
						for (int a = v_segm_to_all_coors[k].size() - 2; a >= 0; a--) {
							if (labels[v_segm_to_all_coors[k][a] - 1] != params.n_all_cands + 1) {
								meet_pt = a;
								break;
							}
						}
						//meet_pt = find(labels(v_segm_to_all_coors{ k })~= params.n_all_cands + 1, 1, 'last');
					}

					if (meet_pt == -1) {
						continue;
					}
					joining_seg.push_back(k);
					joining_seg.push_back(meet_pt);
					//joining_seg = [joining_seg; [k, meet_pt]];
				}
			}

			all_joining_seg.push_back(joining_seg);
			//all_joining_junctions[num_all_joining_seg] = bJ_idx(j);
			num_all_joining_seg = num_all_joining_seg + 1;
		}
	}
	// * END find next points for dummy labeled "feature" points 

	std::vector<std::vector<cv::Point>> newE;
	std::vector<cv::Point> all_v;
	std::vector<cv::Point> vpts_all_subsample;
	std::vector<cv::Point> all_vessel_pt;
	std::vector<cv::Point> cor_line_pt;
	std::vector<std::vector<cv::Point>> vpts_seg_subsample;

	// make new path in tp1 frame
	MakeConnectedCenterlineFromSubsampledPts(m_frangi_vesselness_tp1,v_segm_to_all_coors,all_cands,all_joining_seg,labels,ves_segm_num,num_all_joining_seg,
		&newE, &all_v, &all_vessel_pt, &vpts_all_subsample, &vpts_seg_subsample);


	// stored tp1 feature points
	m_tp1_feat_pts = find_tp1_features(m_t_feat_pts,
		m_t_vpt_arr,
		vpts_all_subsample);



	//stored previous segment vector 
	m_tp1_vsegm_vpt_2darr = newE;

	// stored all candidates
	m_tp1_vpt_arr = vpts_all_subsample;

	// stroed displacement vectors
	m_disp_vec_arr = makeDisplacementVec(m_t_vpt_arr, m_tp1_vpt_arr);

	if (bVerbose)
	{
		cv::Mat drawDisplacemete = drawDisplacementeVec(img_tp1, m_t_vpt_arr, m_tp1_vpt_arr, m_disp_vec_arr);
		sprintf(str, "%s%d-th_frame_displacemente.png", savePath, fidx_tp1);
		cv::imwrite(str, drawDisplacemete);
	}
		

	cv::Mat draw_bimg_tp1(nY, nX, CV_8UC1);
	draw_bimg_tp1 = 0;

	for (int avp = 0; avp < all_vessel_pt.size(); avp++) {
		draw_bimg_tp1.at<uchar>(all_vessel_pt[avp]) = 255;
	}

	if (bVerbose) {
		sprintf(str, "%s%d-th_frame_final_b.png", savePath, fidx_tp1);
		cv::imwrite(str, draw_bimg_tp1);
	}

	if (bVerbose) {
		cv::Mat final_canvas_img;
		cv::cvtColor(img_tp1, final_canvas_img, CV_GRAY2BGR);

		final_canvas_img.setTo(cv::Scalar(255, 0, 0), draw_bimg_tp1);
		sprintf(str, "%s%d-th_frame_final_rgb.png", savePath, fidx_tp1);
		cv::imwrite(str, final_canvas_img);

		for (int avp = 0; avp < all_v.size(); avp++)
		{
			final_canvas_img.at<uchar>(all_v[avp].y, all_v[avp].x * 3 + 0) = 0;
			final_canvas_img.at<uchar>(all_v[avp].y, all_v[avp].x * 3 + 1) = 0;
			final_canvas_img.at<uchar>(all_v[avp].y, all_v[avp].x * 3 + 2) = 255;
		}

		sprintf(str, "%s%d-th_frame_final_rgb_node.png", savePath, fidx_tp1);
		cv::imwrite(str, final_canvas_img);
	}



	///////////// post - processing///////////////

	cv::Mat tp1_post_processed = postProcGrowVessel(img_tp1, m_frangi_vesselness_tp1, all_vessel_pt, params,&newE);

	m_tp1_vsegm_vpt_2darr_pp = newE;


	m_tp1_vsegm_linked_information = linkedSeg(newE);

	if (bVerbose)
	{

		cv::Mat draw(img_h, img_w, CV_8UC3);
		draw = 0;

		cv::Mat cur_seg_draw(img_h, img_w, CV_8UC3);
		cv::Mat cur_seg_sp_link_draw(img_h, img_w, CV_8UC3);
		cv::Mat cur_seg_ep_link_draw(img_h, img_w, CV_8UC3);

		srand(time(NULL));

		for (int i = 0; i < newE.size(); i++)
		{
			cur_seg_draw = 0;
			int r = rand() % 256;
			int g = rand() % 256;
			int b = rand() % 256;
			for (int j = 0; j < newE[i].size(); j++)
			{
				draw.at<uchar>(newE[i][j].y, newE[i][j].x * 3 + 0) = b;
				draw.at<uchar>(newE[i][j].y, newE[i][j].x * 3 + 1) = g;
				draw.at<uchar>(newE[i][j].y, newE[i][j].x * 3 + 2) = r;
			}

		}

		sprintf(str, "%s%d-th_frame_postprocessed_final_seg_graph.png", savePath, fidx_tp1);
		cv::imwrite(str, draw);
	}

	return;
}


void cVCO::globalChamferMatching(
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
	)
{
	cChamferMatching chamf;

	if (b_use_gc) {
		gt_bimg_t = chamf.computeChamferMatch(bimg_t, bimg_tp1, params, t_x, t_y);
	}
	else {
		gt_bimg_t = bimg_t;
		t_x = 0; t_y = 0;
	}
	// * VERIFICATION: draw results to verify global chamfer matching results * //
	if (bVerbose) {
		cv::Mat gc_canvas_img(img_tp1.rows, img_tp1.cols, CV_8UC3);
		if (img_tp1.channels() == 1)
			cv::cvtColor(img_tp1, gc_canvas_img, CV_GRAY2BGR);
		else if (img_tp1.channels() == 3)
			img_tp1.copyTo(gc_canvas_img);

		for (int y = 0; y < gc_canvas_img.rows; y++)
		for (int x = 0; x < gc_canvas_img.cols; x++) {
			if (gt_bimg_t.at<uchar>(y, x)) {
				gc_canvas_img.at<uchar>(y, x * 3 + 0) = 0;
				gc_canvas_img.at<uchar>(y, x * 3 + 1) = 0;
				gc_canvas_img.at<uchar>(y, x * 3 + 2) = 255;
			}
			else if (bimg_t.at<uchar>(y, x)) {
				gc_canvas_img.at<uchar>(y, x * 3 + 0) = 0;
				gc_canvas_img.at<uchar>(y, x * 3 + 1) = 255;
				gc_canvas_img.at<uchar>(y, x * 3 + 2) = 0;
			}
		}
		char str[200];
		sprintf(str, "%s%d-th_frame_gc_rgb.png", savePath, fidx_tp1);
		cv::imwrite(str, gc_canvas_img);
	}
	//return gt_bimg_t;
}

void cVCO::MakeConnectedCenterlineFromSubsampledPts(cv::Mat m_frangi_vesselness_tp1, std::vector<std::vector<int>> v_segm_to_all_coors, cv::Mat all_cands, 
	std::vector<std::vector<int>> all_joining_seg, double* labels, int ves_segm_num, int num_all_joining_seg,
	std::vector<std::vector<cv::Point>> *newE, std::vector<cv::Point> *all_v, std::vector<cv::Point> *all_vessel_pt,
	std::vector<cv::Point> *tp1_vpts_all_subsample, std::vector<std::vector<cv::Point>> *tp1_vpts_seg_subsample)
{
	int nX = m_frangi_vesselness_tp1.cols;
	int nY = m_frangi_vesselness_tp1.rows;

	cv::Mat trs;
	m_frangi_vesselness_tp1.copyTo(trs);
	trs.convertTo(trs, CV_64FC1);
	cv::transpose(trs, trs);
	trs.setTo(1e-10, trs < 1e-10);

	cv::Mat H_mat = cv::Mat::zeros(img_h, img_w, CV_64FC1);
	cv::Mat S_mat = cv::Mat::ones(img_h, img_w, CV_64FC1);
	cv::Mat D_mat(img_h, img_w, CV_64FC1);
	D_mat = 1e9;
	cv::Mat Q_mat = -cv::Mat::ones(img_h, img_w, CV_64FC1);
	cv::Mat PD_mat = -cv::Mat::ones(img_h, img_w, CV_64FC1);

	cFastMarching fmm;
	for (int j = 0; j < ves_segm_num; j++) {
		std::vector<cv::Point> temp_v;
		std::vector<cv::Point> temp_vpts;
		cv::Mat temp = cv::Mat::zeros(nY, nX, CV_64FC1);
		std::vector<int> t_seg = v_segm_to_all_coors[j];
		int len_t_seg = t_seg.size();
		cv::Point st_pt = cv::Point(-1, -1);
		cv::Point ed_pt = cv::Point(-1, -1);
		std::vector<cv::Point> cum_seg_path;
		bool is_first = true;

		for (int k = 0; k < len_t_seg; k++) {
			int t_idx = t_seg[k] - 1;
			double t_label = labels[t_idx] - 1;
			if (t_label >= params.n_all_cands)
			{
				temp_vpts.push_back(cv::Point(-1,-1));
			}
			if (t_label < params.n_all_cands) {
				if (st_pt.x == -1) {
					st_pt = all_cands.at<cv::Point>(t_idx, (int)t_label);

					temp_v.push_back(st_pt);
					temp_vpts.push_back(st_pt);
				}
				else {
					ed_pt = all_cands.at<cv::Point>(t_idx, (int)t_label);
					temp_v.push_back(ed_pt);
					temp_vpts.push_back(ed_pt);
				}
			}
			else {
				ed_pt = cv::Point(-1, -1);
			}
			if (st_pt.x != -1 && ed_pt.x != -1) {
				int st_pt_y = st_pt.y; int st_pt_x = st_pt.x;
				int ed_pt_y = ed_pt.y; int ed_pt_x = ed_pt.x;

				// geodesic path
				double pfm_end_points[] = { ed_pt_y, ed_pt_x };
				double pfm_start_points[] = { st_pt_y, st_pt_x };
				double nb_iter_max = std::min(params.pfm_nb_iter_max,
					(1.2*std::max(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols)*
					std::max(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols)));

				double *D, *S, *H;

				S_mat = 1;
				D_mat = 1e9;
				Q_mat = -1;
				PD_mat = -1;
				fmm.fast_marching(((double*)trs.data), m_frangi_vesselness_tp1.cols, m_frangi_vesselness_tp1.rows, pfm_start_points, 1, pfm_end_points, 1, nb_iter_max,
					(double*)H_mat.data, (double*)S_mat.data, (double*)D_mat.data, (double*)Q_mat.data, (double*)PD_mat.data,
					&D, &S);

				std::vector<cv::Point> geo_path;
				cv::Mat D_mat(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols, CV_64FC1, D);
				cv::transpose(D_mat, D_mat);

				fmm.compute_discrete_geodesic(D_mat, cv::Point(pfm_end_points[1], pfm_end_points[0]), &geo_path);


				if (is_first) {
					for (int a = geo_path.size()-1; a >= 0; a--)
						cum_seg_path.push_back(geo_path[a]);

					is_first = false;
				}
				else {
					for (int a = geo_path.size()-1; a >= 0; a--)
						cum_seg_path.push_back(geo_path[a]);
				}

				st_pt = ed_pt;
				ed_pt = cv::Point(-1, -1);
			}
		}

		if (st_pt == cv::Point(-1, -1) || ed_pt == cv::Point(-1, -1))
		{
			if (st_pt == cv::Point(-1, -1) && ed_pt == cv::Point(-1, -1))
			{
				bool bFind = false;
				for (int k = 0; k < v_segm_to_all_coors.size(); k++)
				{
					if (k == j)
						continue;
					for (int m = 0; m < v_segm_to_all_coors[k].size(); m++)
					{
						if (v_segm_to_all_coors[k][m] == t_seg[0])
						{
							if (m == 0)
							{

								int t_idx = v_segm_to_all_coors[k][m + 1] - 1;
								int t_label = labels[v_segm_to_all_coors[k][m + 1] - 1] - 1;

								if (t_label >= params.n_all_cands)
									continue;

								st_pt = all_cands.at<cv::Point>(t_idx, (int)t_label);

								if (temp_v.size() < 2)
								{
									temp_v.clear();
									temp_v.push_back(st_pt);
								}
								else
								{
									temp_v[0] = (st_pt);
								}

								temp_vpts[0] = (st_pt);

								bFind = true;
								break;
							}
							else
							{
								int t_idx = v_segm_to_all_coors[k][m - 1] - 1;
								int t_label = labels[v_segm_to_all_coors[k][m - 1] - 1] - 1;

								if (t_label >= params.n_all_cands)
									continue;

								st_pt = all_cands.at<cv::Point>(t_idx, (int)t_label);


								if (temp_v.size() < 2)
								{
									temp_v.clear();
									temp_v.push_back(st_pt);
								}
								else
								{
									temp_v[0] = (st_pt);
								}

								temp_vpts[0] = (st_pt);

								bFind = true;
								break;
							}

						}

					}
					if (bFind)
						break;
				}

				bFind = false;
				for (int k = 0; k < v_segm_to_all_coors.size(); k++)
				{
					if (k == j)
						continue;
					for (int m = 0; m < v_segm_to_all_coors[k].size(); m++)
					{
						if (v_segm_to_all_coors[k][m] == t_seg[t_seg.size() - 1])
						{
							if (m == 0)
							{

								int t_idx = v_segm_to_all_coors[k][m + 1] - 1;
								int t_label = labels[v_segm_to_all_coors[k][m + 1] - 1] - 1;

								if (t_label >= params.n_all_cands)
									continue;

								ed_pt = all_cands.at<cv::Point>(t_idx, (int)t_label);


								if (temp_v.size() < 2)
								{
									temp_v.push_back(ed_pt);
								}
								else
								{
									temp_v[temp_v.size() - 1] = (ed_pt);
								}

								
								temp_vpts[temp_v.size() - 1] = (ed_pt);

								bFind = true;
								break;
							}
							else
							{
								int t_idx = v_segm_to_all_coors[k][m - 1] - 1;
								int t_label = labels[v_segm_to_all_coors[k][m - 1] - 1] - 1;

								if (t_label >= params.n_all_cands)
									continue;

								ed_pt = all_cands.at<cv::Point>(t_idx, (int)t_label);


								if (temp_v.size() < 2)
								{
									temp_v.push_back(ed_pt);
								}
								else
								{
									temp_v[temp_v.size() - 1] = (ed_pt);
								}

								temp_vpts[temp_v.size() - 1] = (ed_pt);


								bFind = true;
								break;
							}

						}

					}
					if (bFind)
						break;
				}


				if (st_pt != cv::Point(-1, -1) && ed_pt != cv::Point(-1, -1) )
				{
					int st_pt_y = st_pt.y; int st_pt_x = st_pt.x;
					int ed_pt_y = ed_pt.y; int ed_pt_x = ed_pt.x;
					double pfm_end_points[] = { ed_pt_y, ed_pt_x };
					double pfm_start_points[] = { st_pt_y, st_pt_x };
					double nb_iter_max = std::min(params.pfm_nb_iter_max,
						(1.2*std::max(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols)*
						std::max(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols)));

					double *D, *S, *H;

					S_mat = 1;
					D_mat = 1e9;
					Q_mat = -1;
					PD_mat = -1;
					fmm.fast_marching(((double*)trs.data), m_frangi_vesselness_tp1.cols, m_frangi_vesselness_tp1.rows, pfm_start_points, 1, pfm_end_points, 1, nb_iter_max,
						(double*)H_mat.data, (double*)S_mat.data, (double*)D_mat.data, (double*)Q_mat.data, (double*)PD_mat.data,
						&D, &S);

					std::vector<cv::Point> geo_path;
					cv::Mat D_mat(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols, CV_64FC1, D);
					cv::transpose(D_mat, D_mat);

					fmm.compute_discrete_geodesic(D_mat, cv::Point(pfm_end_points[1], pfm_end_points[0]), &geo_path);



					for (int a = 0; a < geo_path.size(); a++)
					{
						cum_seg_path.insert(cum_seg_path.begin(), geo_path[a]);
					}

				}


			}
			else if (st_pt == cv::Point(-1, -1))
			{
				bool bFind = false;
				for (int k = 0; k < v_segm_to_all_coors.size(); k++)
				{
					if (k == j)
						continue;
					for (int m = 0; m < v_segm_to_all_coors[k].size(); m++)
					{
						if (v_segm_to_all_coors[k][m] == t_seg[0])
						{
							if (m == 0)
							{
								int t_idx = v_segm_to_all_coors[k][m+1] - 1;
								int t_label = labels[v_segm_to_all_coors[k][m + 1] - 1] - 1;

								if (t_label >= params.n_all_cands)
									continue;

								st_pt = all_cands.at<cv::Point>(t_idx, (int)t_label);

								if (temp_v.size() < 2)
								{
									temp_v.insert(temp_v.begin(), st_pt);
								}
								else
								{
									temp_v[0] = (st_pt);
								}
								temp_vpts[0] = (st_pt);

								bFind = true;
								break;
							}
							else
							{
								int t_idx = v_segm_to_all_coors[k][m -1] - 1;
								int t_label = labels[v_segm_to_all_coors[k][m -1] - 1] - 1;

								if (t_label >= params.n_all_cands)
									continue;

								st_pt = all_cands.at<cv::Point>(t_idx, (int)t_label);

								if (temp_v.size() < 2)
								{
									temp_v.insert(temp_v.begin(), st_pt);
								}
								else
								{
									temp_v[0] = (st_pt);
								}
								temp_vpts[0] = (st_pt);

								bFind = true;
								break;
							}

						}
						
					}
					if (bFind)
						break;
				}


				if (st_pt != cv::Point(-1, -1))
				{
					int st_pt_y = st_pt.y; int st_pt_x = st_pt.x;
					int ed_pt_y = ed_pt.y; int ed_pt_x = ed_pt.x;

					double pfm_end_points[] = { ed_pt_y, ed_pt_x };
					double pfm_start_points[] = { st_pt_y, st_pt_x };
					double nb_iter_max = std::min(params.pfm_nb_iter_max,
						(1.2*std::max(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols)*
						std::max(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols)));

					double *D, *S, *H;

					S_mat = 1;
					D_mat = 1e9;
					Q_mat = -1;
					PD_mat = -1;
					fmm.fast_marching(((double*)trs.data), m_frangi_vesselness_tp1.cols, m_frangi_vesselness_tp1.rows, pfm_start_points, 1, pfm_end_points, 1, nb_iter_max,
						(double*)H_mat.data, (double*)S_mat.data, (double*)D_mat.data, (double*)Q_mat.data, (double*)PD_mat.data,
						&D, &S);

					std::vector<cv::Point> geo_path;
					cv::Mat D_mat(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols, CV_64FC1, D);
					cv::transpose(D_mat, D_mat);

					fmm.compute_discrete_geodesic(D_mat, cv::Point(pfm_end_points[1], pfm_end_points[0]), &geo_path);

					for (int a = 0; a < geo_path.size(); a++)
					{
						cum_seg_path.insert(cum_seg_path.begin(), geo_path[a]);
					}

				}

				
			}
			else if (ed_pt == cv::Point(-1, -1))
			{
				bool bFind = false;
				for (int k = 0; k < v_segm_to_all_coors.size(); k++)
				{
					if (k == j)
						continue;
					for (int m = 0; m < v_segm_to_all_coors[k].size(); m++)
					{
						if (v_segm_to_all_coors[k][m] == t_seg[t_seg.size()-1])
						{
							if (m == 0)
							{
								int t_idx = v_segm_to_all_coors[k][m + 1] - 1;
								int t_label = labels[v_segm_to_all_coors[k][m + 1] - 1] - 1;

								if (t_label >= params.n_all_cands)
									continue;

								ed_pt = all_cands.at<cv::Point>(t_idx, (int)t_label);

								if (temp_v.size() < 2)
								{
									temp_v.push_back(ed_pt);
								}
								else
								{
									temp_v[temp_v.size() - 1] = (ed_pt);
								}

								temp_vpts[temp_v.size() - 1] = (ed_pt);

								bFind = true;
								break;
							}
							else
							{
								int t_idx = v_segm_to_all_coors[k][m - 1] - 1;
								int t_label = labels[v_segm_to_all_coors[k][m - 1] - 1] - 1;

								if (t_label >= params.n_all_cands)
									continue;

								ed_pt = all_cands.at<cv::Point>(t_idx, (int)t_label);

								if (temp_v.size() < 2)
								{
									temp_v.push_back(ed_pt);
								}
								else
								{
									temp_v[temp_v.size() - 1] = (ed_pt);
								}

								temp_vpts[temp_v.size() - 1] = (ed_pt);


								bFind = true;
								break;
							}

						}

					}
					if (bFind)
						break;
				}

				if (ed_pt != cv::Point(-1, -1))
				{
					int st_pt_y = st_pt.y; int st_pt_x = st_pt.x;
					int ed_pt_y = ed_pt.y; int ed_pt_x = ed_pt.x;

					// geodesic path
					double pfm_end_points[] = { ed_pt_y, ed_pt_x };
					double pfm_start_points[] = { st_pt_y, st_pt_x };
					double nb_iter_max = std::min(params.pfm_nb_iter_max,
						(1.2*std::max(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols)*
						std::max(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols)));

					double *D, *S, *H;


					S_mat = 1;
					D_mat = 1e9;
					Q_mat = -1;
					PD_mat = -1;
					fmm.fast_marching(((double*)trs.data), m_frangi_vesselness_tp1.cols, m_frangi_vesselness_tp1.rows, pfm_start_points, 1, pfm_end_points, 1, nb_iter_max,
						(double*)H_mat.data, (double*)S_mat.data, (double*)D_mat.data, (double*)Q_mat.data, (double*)PD_mat.data,
						&D, &S);

					std::vector<cv::Point> geo_path;
					cv::Mat D_mat(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols, CV_64FC1, D);
					cv::transpose(D_mat, D_mat);

					fmm.compute_discrete_geodesic(D_mat, cv::Point(pfm_end_points[1], pfm_end_points[0]), &geo_path);


					if (geo_path.size())
					for (int a = geo_path.size() - 1; a >= 0; a--)
						cum_seg_path.push_back(geo_path[a]);


				}

			}
		}


		newE->push_back(cum_seg_path);
		for (int k = 0; k < temp_vpts.size(); k++)
		{
			tp1_vpts_all_subsample->push_back(temp_vpts[k]);
			if (temp_vpts[k] == cv::Point(-1, -1))
			{
				int a = 0;
			}
		}
		tp1_vpts_seg_subsample->push_back(temp_vpts);
		if (cum_seg_path.size()) {
			for (int k = 0; k < temp_v.size(); k++)
			{
				all_v->push_back(temp_v[k]);
			}
			
			for (int k = 0; k < cum_seg_path.size(); k++) {
				all_vessel_pt->push_back(cum_seg_path[k]);
				temp.at<double>(cum_seg_path[k]) = true;
			}

		}
	}

	// drawing for junctions labeled as 'dummy'
	for (int j = 0; j < num_all_joining_seg; j++)
	{
		std::vector<int> joining_seg = all_joining_seg[j];
		int n_joining_seg = joining_seg.size();
		std::vector<cv::Point> cum_path;
		for (int k = 0; k < n_joining_seg / 2; k++)
		{
			int t_idx = v_segm_to_all_coors[joining_seg[k * 2 + 0]][joining_seg[k * 2 + 1]] - 1;
			int t_label = labels[t_idx] - 1;
			cv::Point t_coor1 = all_cands.at<cv::Point>(t_idx, t_label);

			int st_pt_y = t_coor1.y;
			int st_pt_x = t_coor1.x;

			for (int m = k; m < n_joining_seg / 2; m++)
			{
				int t_idx = v_segm_to_all_coors[joining_seg[m * 2 + 0]][joining_seg[m * 2 + 1]] - 1;
				int t_label = labels[t_idx] - 1;
				cv::Point t_coor2 = all_cands.at<cv::Point>(t_idx, t_label);

				int ed_pt_y = t_coor2.y;
				int ed_pt_x = t_coor2.x;

				// geodesic path

				double pfm_end_points[] = { ed_pt_y, ed_pt_x };
				double pfm_start_points[] = { st_pt_y, st_pt_x };


				double nb_iter_max = std::min(params.pfm_nb_iter_max, 1.2*std::max(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols)*std::max(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols));

				double *D, *S;
				S_mat = 1;
				D_mat = 1e9;
				Q_mat = -1;
				PD_mat = -1;
				fmm.fast_marching(((double*)trs.data), m_frangi_vesselness_tp1.cols, m_frangi_vesselness_tp1.rows, pfm_start_points, 1, pfm_end_points, 1, nb_iter_max,
					(double*)H_mat.data, (double*)S_mat.data, (double*)D_mat.data, (double*)Q_mat.data, (double*)PD_mat.data,
					&D, &S);

				std::vector<cv::Point> geo_path;
				cv::Mat D_mat(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols, CV_64FC1, D);
				cv::transpose(D_mat, D_mat);

				fmm.compute_discrete_geodesic(D_mat, cv::Point(pfm_end_points[1], pfm_end_points[0]), &geo_path);

				for (int n = geo_path.size() - 1; n >1; n--)
				{
					cum_path.push_back(geo_path[n]);
				}
			}
		}
		if (!cum_path.size())
		{
			continue;
		}

		for (int k = 0; k < cum_path.size(); k++)
			all_vessel_pt->push_back(cum_path[k]);

		newE->push_back(cum_path);
		//lidx = sub2ind([nY, nX], cum_path(:, 1), cum_path(:, 2));
		//all_vessel_pt = [all_vessel_pt; lidx];
	}
	// drawing for junctions labeled as 'dummy'
	

	all_vessel_pt->clear();

	for (int i = 0; i < newE->size(); i++)
	for (int j = 0; j < (*newE)[i].size(); j++)
	{
		all_vessel_pt->push_back((*newE)[i][j]);
	}
	

}

cv::Mat cVCO::postProcGrowVessel(cv::Mat img_tp1, cv::Mat m_frangi_vesselness_tp1, std::vector<cv::Point> all_vessel_pt,
	cVCOParams params, std::vector<std::vector<cv::Point>> *E)
{
	cv::Mat new_bimg;
	std::vector<cv::Point> new_lidx, app_lidx;

	GrowVesselUsingFastMarching(m_frangi_vesselness_tp1, all_vessel_pt, params.thre_ivessel,
		params, &new_bimg, &new_lidx, &app_lidx);


	cv::Mat img_add_seg(new_bimg.rows, new_bimg.cols, CV_8UC1);
	img_add_seg = 0;

	cv::Mat exclude(img_h, img_w, CV_8UC1);
	exclude = 255;
	exclude(cv::Rect(boundaryRange, boundaryRange, img_w - boundaryRange, img_h - boundaryRange)) = 0;

	for (int i = 0; i < app_lidx.size(); i++)
	{
		if (app_lidx[i].x >= 12 && app_lidx[i].y >= 12 && app_lidx[i].x < img_w - 12 && app_lidx[i].y < img_h - 12)
			img_add_seg.at<uchar>(app_lidx[i].y, app_lidx[i].x) = 255;

	}

	img_add_seg.setTo(0,exclude==255);

	cP2pMatching p2p;

	p2p.thin(img_add_seg, img_add_seg);

	cv::Mat CCA;
	int nCCA = cv::connectedComponents(img_add_seg, CCA);

	std::vector<std::vector<cv::Point>> add_pts;

	std::vector<cv::Mat> added_seg_lists;

	std::vector<std::vector<cv::Point>> tmpE;


	for (int i = 1; i < nCCA; i++)
	{
		cv::Mat cur_CCA = CCA == i;
		cv::Mat idx;
		cv::findNonZero(cur_CCA, idx);
		if (idx.rows < 2)
			continue;

		std::vector<std::vector<cv::Point>> added_E;
		std::vector<cv::Point> J,end;
		cv::Mat map, bJ;
		p2p.MakeGraphFromImage(cur_CCA, J, end, bJ, added_E, map);

		
		for (int j = 0; j < added_E.size(); j++)
		{
			cv::Mat seg_img(img_h,img_w,CV_8UC1);
			seg_img = 0;
			for (int k = 0; k < added_E[j].size(); k++)
			{
				seg_img.at<uchar>(added_E[j][k].y, added_E[j][k].x) = 255;
			}

			added_seg_lists.push_back(seg_img);
			tmpE.push_back(added_E[j]);
		}
	}

	add_pts = tmpE;



	for (int i = 0; i < add_pts.size(); i++)
		(*E).push_back(add_pts[i]);
	
	cv::Mat tp1_pp_vscl_mask(img_h, img_w, CV_8UC1);
	tp1_pp_vscl_mask = 0;

	for (int i = 0; i < (*E).size(); i++)
	for (int j = 0; j < (*E)[i].size(); j++)
	{
		tp1_pp_vscl_mask.at<uchar>((*E)[i][j].y, (*E)[i][j].x) = 255;
	}


	char str[100];

	if (bVerbose)
	{
		cv::Mat final_canvas_img;
		sprintf(str, "%s%d-th_frame_final_post_processed_b.png", savePath, fidx_tp1);
		cv::imwrite(str, tp1_pp_vscl_mask);

		cv::cvtColor(img_tp1, final_canvas_img, CV_GRAY2BGR);

		final_canvas_img.setTo(cv::Scalar(255, 0, 0), new_bimg);
		sprintf(str, "%s%d-th_frame_final_post_processed_rgb.png", savePath, fidx_tp1);
		cv::imwrite(str, final_canvas_img);
	}

	return tp1_pp_vscl_mask;

}

void cVCO::GrowVesselUsingFastMarching(cv::Mat ivessel, std::vector<cv::Point> lidx, double thre, cVCOParams p,
	cv::Mat *o_new_bimg, std::vector<cv::Point> *o_new_lidx, std::vector<cv::Point> *o_app_lidx)
{
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
	//% converted by kjNoh(160600)


	cFastMarching ffm;


	bool verbose = false;
	bool IS3D = false;

	int nY = ivessel.rows;
	int nX = ivessel.cols;

	// Convert double image to logical
	cv::Mat Ibin = ivessel >= thre;
	
	cv::Mat CC;
	int numCC = cv::connectedComponents(Ibin, CC);

	cv::Mat bROI = cv::Mat::zeros(numCC + 1, 1, CV_64FC1);
	Ibin = cv::Mat::zeros(nY, nX, CV_8UC1);

	for (int i = 0; i <= numCC; i++)
	{

		cv::Mat curCC = CC == (i + 1);
		std::vector<cv::Point> cc_idx;
		cv::findNonZero(curCC, cc_idx);
		std::vector<int> Lia(cc_idx.size());
		bool isempty = true;
		for (int j = 0; j < cc_idx.size(); j++)
		{
			for (int k = 0; k < lidx.size(); k++)
			{
				if (cc_idx[j] == lidx[k])
				{
					Lia[j] = 1;
					isempty = false;
				}
				else
				{
					Lia[j] = 0;
				}
			}
		}
		if (!isempty)
		{
			bROI.at<double>(i) = 1;
			for (int j = 0; j < cc_idx.size(); j++)
			{
				Ibin.at<uchar>(cc_idx[j]) = 255;
			}
		}
	}

	// Distance to vessel boundary
	cv::Mat BoundaryDistance;
	getBoundaryDistance(Ibin, IS3D, &BoundaryDistance);

	// Get maximum distance value, which is used as starting point of the
	// first skeleton branch
	double maxD;
	cv::Point dummy_pt;
	maxDistancePoint(BoundaryDistance, Ibin, IS3D, &dummy_pt, &maxD);

	//// Make a fastmarching speed image from the distance image
	//SpeedImage = (BoundaryDistance / maxD). ^ 4;
	//SpeedImage(SpeedImage == 0) = 1e-10;

	// Skeleton segments found by fastmarching
	std::vector<std::vector<cv::Point>> SkeletonSegments(1000);

	// Number of skeleton iterations
	int itt = 0;

	std::vector<cv::Point> SourcePoint = lidx;

	cv::Mat trs;
	ivessel.copyTo(trs);
	trs.convertTo(trs, CV_64FC1);
	cv::transpose(trs, trs);
	trs.setTo(1e-10, trs < 1e-10);

	cv::Mat H_mat(ivessel.rows, ivessel.cols, CV_64FC1);
	H_mat = 0;
	cv::Mat S_mat(ivessel.rows, ivessel.cols, CV_64FC1);
	S_mat = 1;
	cv::Mat D_mat(ivessel.rows, ivessel.cols, CV_64FC1);
	D_mat = 1e9;
	cv::Mat Q_mat(ivessel.rows, ivessel.cols, CV_64FC1);
	Q_mat = -1;
	cv::Mat PD_mat(ivessel.rows, ivessel.cols, CV_64FC1);
	PD_mat = -1;

	while (true)
	{

		// Do fast marching using the maximum distance value in the image
		// and the points describing all found branches are sourcepoints.
		double nb_iter_max = std::min(p.pfm_nb_iter_max, 1.2*std::max(ivessel.rows, ivessel.cols)*std::max(ivessel.rows, ivessel.cols));

		double *D, *S;
		S_mat = 1;
		D_mat = 1e9;
		Q_mat = -1;
		PD_mat = -1;


		double* arrSourcePoint = new double[SourcePoint.size() * 2];
		for (int i = 0; i < SourcePoint.size(); i++)
		{
			arrSourcePoint[i * 2 + 0] = SourcePoint[i].y;
			arrSourcePoint[i * 2 + 1] = SourcePoint[i].x;
		}

		cv::Mat tmp = cv::Mat::zeros(nY, nX, CV_8UC1);
		for (int i = 0; i < SourcePoint.size(); i++)
			tmp.at<uchar>(SourcePoint[i]) = 255;


		cv::Mat Y;
		cv::distanceTransform(~tmp, Y, cv::DistanceTypes::DIST_L2, cv::DistanceTransformMasks::DIST_MASK_3, CV_32FC1);

		// Trace a branch back to the used sourcepoints
		cv::Point StartPoint;
		double dummyv;
		maxDistancePoint(Y, Ibin, IS3D, &StartPoint, &dummyv);

		double endpt[2] = { StartPoint.y, StartPoint.x };
		ffm.fast_marching(((double*)trs.data), ivessel.cols, ivessel.rows, arrSourcePoint, SourcePoint.size(), endpt, 1, nb_iter_max,
			(double*)H_mat.data, (double*)S_mat.data, (double*)D_mat.data, (double*)Q_mat.data, (double*)PD_mat.data,
			&D, &S);

		cv::Mat D_mat(nY, nX, CV_64FC1, D);
		cv::transpose(D_mat, D_mat);
		std::vector<cv::Point> ShortestLine;
		ffm.compute_discrete_geodesic(D_mat, StartPoint, &ShortestLine);

		// Calculate the length of the new skeleton segment
		double linelength;
		GetLineLength(ShortestLine, IS3D, &linelength);

		// Stop finding branches, if the lenght of the new branch is smaller
		// then the diameter of the largest vessel
		if (linelength <= maxD * 2)
		{
			break;
		}

		// Store the found branch skeleton

		SkeletonSegments[itt] = ShortestLine;

		itt = itt + 1;


		// Add found branche to the list of fastmarching SourcePoints
		for (int i = 0; i < ShortestLine.size(); i++)
			SourcePoint.push_back(ShortestLine[i]);


	}
	std::vector<std::vector<cv::Point>> tmp_SkeletonSegments;
	for (int i = 0; i < itt; i++)
		tmp_SkeletonSegments.push_back(SkeletonSegments[i]);

	SkeletonSegments.clear();
	SkeletonSegments = tmp_SkeletonSegments;
	tmp_SkeletonSegments.clear();

	std::vector<cv::Point> lidx_app;
	if (SkeletonSegments.size())
	{
		for (int i = 0; i < SkeletonSegments.size(); i++)
		{
			std::vector<cv::Point> L = SkeletonSegments[i];
			for (int j = 0; j < L.size(); j++)
				lidx_app.push_back(L[j]);
		}

	}

	std::vector<cv::Point> app_lidx = lidx_app;
	std::vector<cv::Point> new_lidx = lidx;


	for (int i = 0; i < lidx_app.size(); i++)
	{

		new_lidx.push_back(lidx_app[i]);

	}
	cv::Mat new_bimg = cv::Mat::zeros(nY, nX, CV_8UC1);
	for (int i = 0; i < new_lidx.size(); i++)
		new_bimg.at<uchar>(new_lidx[i]) = 255;

	*o_new_bimg = new_bimg;
	*o_new_lidx = new_lidx;
	*o_app_lidx = app_lidx;

}

void cVCO::GetLineLength(std::vector<cv::Point> L, bool IS3D, double *o_ll)
{
	double ll = 0;
	for (int i = 0; i < L.size() - 1; i++)
	{
		double cur_dist = std::sqrt((L[i + 1].y - L[i].y)*(L[i + 1].y - L[i].y) + (L[i + 1].x - L[i].x)*(L[i + 1].x - L[i].x));
		ll += cur_dist;
	}


	*o_ll = ll;
}

void cVCO::getBoundaryDistance(cv::Mat I, bool IS3D, cv::Mat *o_BoundaryDistance)
{
	//function BoundaryDistance = getBoundaryDistance(I, IS3D)
	// Calculate Distance to vessel boundary

	// Set all boundary pixels as fastmarching source - points(distance = 0)

	cv::Mat S(3, 3, CV_64FC1);
	S = 255;
	cv::Mat B;
	cv::Mat tmp_I;

	I.copyTo(tmp_I);
	cv::dilate(tmp_I, tmp_I, S);
	B = tmp_I^I;

	cv::Mat BoundaryDistance;
	cv::distanceTransform(~B, BoundaryDistance, cv::DistanceTypes::DIST_L2, cv::DistanceTransformMasks::DIST_MASK_3, CV_32FC1);


	BoundaryDistance.setTo(0, ~I);

	*o_BoundaryDistance = BoundaryDistance;
}

void cVCO::maxDistancePoint(cv::Mat BoundaryDistance, cv::Mat I, bool IS3D, cv::Point *o_posD, double *o_maxD)
{
	// Mask the result by the binary input image
	BoundaryDistance.setTo(0, ~I);

	// Find the maximum distance voxel
	double maxv, minv;
	int maxidx[2];
	cv::minMaxIdx(BoundaryDistance, &minv, &maxv, 0, maxidx);

	cv::Point posD(maxidx[1], maxidx[0]);

	*o_maxD = maxv;
	*o_posD = posD;

}

std::vector<cv::Point> cVCO::makeSegvec2Allvec(std::vector<std::vector<cv::Point>> segVec)
{
	int size = segVec.size();

	std::vector<cv::Point> allVec;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < segVec[i].size(); j++)
		{
			allVec.push_back(segVec[i][j]);
		}
	}

	return allVec;

}

std::vector<cv::Point> cVCO::makeSegvec2Allvec(std::vector<std::vector<cv::Point>> segVec, cv::Point tanslation)
{
	int size = segVec.size();

	std::vector<cv::Point> allVec;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < segVec[i].size(); j++)
		{
			allVec.push_back(segVec[i][j] - tanslation);
		}
	}

	return allVec;

}


std::vector<cv::Point> cVCO::makeDisplacementVec(std::vector<cv::Point> pre, std::vector<cv::Point> post)
{
	std::vector<cv::Point> displacementeVec;
	if (pre.size() != post.size())
	{
		printf("previous all vectors is not same size to post all vectors\n");
		return displacementeVec;
	}

	
	for (int i = 0; i < pre.size(); i++)
	{
		if (post[i] == cv::Point(-1, -1))
			displacementeVec.push_back(cv::Point(1000, 1000));
		else
			displacementeVec.push_back(post[i] - pre[i]);
	}

	return displacementeVec;
}

cv::Mat cVCO::drawDisplacementeVec(cv::Mat img,std::vector<cv::Point> pre, std::vector<cv::Point> post, std::vector<cv::Point> dispVec)
{
	cv::Mat palette;
	if (img.channels() == 1)
	{
		cv::cvtColor(img, palette, CV_GRAY2BGR);
	}
	else
	{
		img.copyTo(palette);
	}
	

	int size = dispVec.size();

	for (int i = 0; i < size; i++)
	{
		if (dispVec[i].x < 1000)
		{
			cv::line(palette, pre[i], post[i], CV_RGB(255, 0, 0));
		}
		
	}

	for (int i = 0; i < pre.size(); i++)
	{
		if (dispVec[i].x < 1000)
		{

			palette.at<uchar>(pre[i].y, pre[i].x * 3 + 0) = 255;
			palette.at<uchar>(pre[i].y, pre[i].x * 3 + 1) = 0;
			palette.at<uchar>(pre[i].y, pre[i].x * 3 + 2) = 0;

			palette.at<uchar>(post[i].y, post[i].x * 3 + 0) = 0;
			palette.at<uchar>(post[i].y, post[i].x * 3 + 1) = 255;
			palette.at<uchar>(post[i].y, post[i].x * 3 + 2) = 0;
		}
	}

	return palette;
}


cv::Mat cVCO::get_tp1_vescl_mask_pp()
{
	cv::Mat tp1_vmask_pp(img_h, img_w, CV_8UC1);

	tp1_vmask_pp = 0;

	for (int i = 0; i < m_tp1_vsegm_vpt_2darr_pp.size(); i++)
	for (int j = 0; j < m_tp1_vsegm_vpt_2darr_pp[i].size(); j++)
	{
		int x = m_tp1_vsegm_vpt_2darr_pp[i][j].x;
		int y = m_tp1_vsegm_vpt_2darr_pp[i][j].y;

		tp1_vmask_pp.at<uchar>(y, x) = 255;
	}

	return tp1_vmask_pp;
}

cv::Mat cVCO::get_tp1_vescl_mask()
{
	cv::Mat tp1_vmask(img_h, img_w, CV_8UC1);
	tp1_vmask = 0;

	for (int i = 0; i < m_tp1_vsegm_vpt_2darr.size(); i++)
	for (int j = 0; j < m_tp1_vsegm_vpt_2darr[i].size(); j++)
	{
		int x = m_tp1_vsegm_vpt_2darr[i][j].x;
		int y = m_tp1_vsegm_vpt_2darr[i][j].y;

		tp1_vmask.at<uchar>(y,x) = 255;
	}


	return tp1_vmask;
}

std::vector<ves_feat_pt> cVCO::get_t_VesFeatPts()
{
	return m_t_feat_pts;
}
std::vector<ves_feat_pt> cVCO::get_tp1_VesFeatPts()
{
	return m_tp1_feat_pts;
}

std::vector<ves_feat_pt>cVCO::setVesFeatPts(std::vector<cv::Point> junction, std::vector<cv::Point> end, cv::Point tans)
{ 
	int junctionSize = junction.size();
	int endSize = end.size();

	std::vector<ves_feat_pt> feat_pts;
	

	// stored junction feature points 
	for (int i = 0; i < junctionSize; i++)
	{
		ves_feat_pt cur_feat;
		cur_feat.x = junction[i].x - tans.x;
		cur_feat.y = junction[i].y - tans.y;
		cur_feat.type = 1;
		feat_pts.push_back(cur_feat);
	}

	// stored end feature points 
	for (int i = 0; i < endSize; i++)
	{
		ves_feat_pt cur_feat;
		cur_feat.x = end[i].x - tans.x;
		cur_feat.y = end[i].y - tans.y;
		cur_feat.type = 0;
		feat_pts.push_back(cur_feat);
	}

	return feat_pts;
}
std::vector<std::vector<cv::Point>> cVCO::getVsegVpts2dArr()
{
	return m_tp1_vsegm_vpt_2darr;
}
std::vector<std::vector<cv::Point>> cVCO::getVsegVpts2dArr_pp()
{
	return m_tp1_vsegm_vpt_2darr_pp;
}

std::vector<cv::Point> cVCO::get_t_vpts_arr()
{
	return m_t_vpt_arr;
}
std::vector<cv::Point> cVCO::get_tp1_vpts_arr()
{
	return m_tp1_vpt_arr;
}
std::vector<cv::Point> cVCO::get_disp_vec_arr()
{
	return m_disp_vec_arr;
}

cv::Mat cVCO::get_tp1_FrangiVesselnessMask()
{
	return m_frangi_vesselness_tp1;
}
float* cVCO::get_tp1_p_FrangiVesselnessMask()
{
	return m_p_frangi_vesselness_tp1;
}

unsigned char* cVCO::get_p_tp1_mask_8u()
{
	unsigned char* p_tp1_vmask_8u = new unsigned char[img_h*img_w];

	for (int i = 0; i < img_h*img_w; i++)
	{
		p_tp1_vmask_8u[i] = 0;
	}

	for (int i = 0; i < m_tp1_vsegm_vpt_2darr.size(); i++)
	for (int j = 0; j < m_tp1_vsegm_vpt_2darr[i].size(); j++)
	{
		int x = m_tp1_vsegm_vpt_2darr[i][j].x;
		int y = m_tp1_vsegm_vpt_2darr[i][j].y;

		p_tp1_vmask_8u[y*img_w + x] = 255;
	}
	return p_tp1_vmask_8u;
}

unsigned char* cVCO::get_p_tp1_mask_pp_8u()
{
	unsigned char* p_tp1_vmask_pp_8u = new unsigned char[img_h*img_w];

	for (int i = 0; i < img_h*img_w; i++)
	{
		p_tp1_vmask_pp_8u[i] = 0;
	}

	for (int i = 0; i < m_tp1_vsegm_vpt_2darr_pp.size(); i++)
	for (int j = 0; j < m_tp1_vsegm_vpt_2darr_pp[i].size(); j++)
	{
		int x = m_tp1_vsegm_vpt_2darr_pp[i][j].x;
		int y = m_tp1_vsegm_vpt_2darr_pp[i][j].y;

		p_tp1_vmask_pp_8u[y*img_w + x] = 255;
	}
	

	return p_tp1_vmask_pp_8u;
}

std::vector<ves_feat_pt> cVCO::find_tp1_features(std::vector<ves_feat_pt> t_features, 
	std::vector<std::vector<cv::Point>> t_seg_vec,
	std::vector<std::vector<cv::Point>> tp1_seg_vec)
{
	std::vector<ves_feat_pt> features;
	std::vector<cv::Point> check;
	for (int i = 0; i < t_features.size(); i++)
	{
		for (int j = 0; j < t_seg_vec.size(); j++)
		{
			if (t_seg_vec[j][0].x == t_features[i].x &&
				t_seg_vec[j][0].y == t_features[i].y)
			{
				ves_feat_pt cur_feature;
				cur_feature.x = tp1_seg_vec[j][0].x;
				cur_feature.y = tp1_seg_vec[j][0].y;
				cur_feature.type = t_features[i].type;
				
				features.push_back(cur_feature);
				check.push_back(cv::Point(t_features[i].x, t_features[i].y));
			}
			if (t_seg_vec[j][t_seg_vec[j].size() - 1].x == t_features[i].x &&
				t_seg_vec[j][t_seg_vec[j].size() - 1].y == t_features[i].y)
			{
				ves_feat_pt cur_feature;
				cur_feature.x = tp1_seg_vec[j][tp1_seg_vec[j].size() - 1].x;
				cur_feature.y = tp1_seg_vec[j][tp1_seg_vec[j].size() - 1].y;
				cur_feature.type = t_features[i].type;

				features.push_back(cur_feature);
				check.push_back(cv::Point(t_features[i].x, t_features[i].y));
			}
		}
		
	}
	return features;
}
std::vector<ves_feat_pt> cVCO::find_tp1_features(std::vector<ves_feat_pt> t_features,
	std::vector<cv::Point> t_vseg,
	std::vector<cv::Point> tp1_vseg)
{
	std::vector<ves_feat_pt> features;
	std::vector<cv::Point> check;

	cv::Mat checked_repeat(img_h,img_w,CV_8UC1);
	checked_repeat = 0;

	for (int i = 0; i < t_features.size(); i++)
	{
		for (int j = 0; j < t_vseg.size(); j++)
		{
			if (t_vseg[j].x == t_features[i].x &&
				t_vseg[j].y == t_features[i].y &&
				!checked_repeat.at<uchar>(t_vseg[j]))
			{
				ves_feat_pt cur_feature;
				cur_feature.x = tp1_vseg[j].x;
				cur_feature.y = tp1_vseg[j].y;
				cur_feature.type = t_features[i].type;

				features.push_back(cur_feature);
				check.push_back(cv::Point(t_features[i].x, t_features[i].y));
				checked_repeat.at<uchar>(t_vseg[j]) = 255;
			}
			if (t_vseg[j].x == t_features[i].x &&
				t_vseg[j].y == t_features[i].y &&
				!checked_repeat.at<uchar>(t_vseg[j]))
			{
				ves_feat_pt cur_feature;
				cur_feature.x = tp1_vseg[j].x;
				cur_feature.y = tp1_vseg[j].y;
				cur_feature.type = t_features[i].type;

				features.push_back(cur_feature);
				check.push_back(cv::Point(t_features[i].x, t_features[i].y));
				checked_repeat.at<uchar>(t_vseg[j]) = 255;
			}
		}

	}
	return features;
}

std::vector<std::vector<std::vector<std::vector<int>>>> cVCO::linkedSeg(std::vector<std::vector<cv::Point>> seg)
{
	std::vector<std::vector<std::vector<std::vector<int>>>> linked;

	for (int i = 0; i < seg.size(); i++)
	{
		if (!seg[i].size())
		{
			std::vector<std::vector<std::vector<int>>> dummy;
			dummy.push_back(std::vector<std::vector<int>>());
			dummy.push_back(std::vector<std::vector<int>>());
			linked.push_back(dummy);
			continue;
		}
		cv::Point cur_sp = seg[i][0];
		cv::Point cur_ep = seg[i][seg[i].size()-1];

		std::vector<std::vector<std::vector<int>>> cur_seg_spep_linking;
		std::vector<std::vector<int>> sp_linking;
		std::vector<std::vector<int>> ep_linking;

		for (int s1 = 0; s1 < seg.size(); s1++)
		{
			if (s1 == i)
				continue;

			

			for (int s2 = 0; s2 < seg[s1].size(); s2++)
			{
				


				if (cur_sp == seg[s1][s2])
				{
					std::vector<int> cur_linking;
					if (s2 == 0)
					{
						cur_linking.push_back(s1); // linked other segmentation
						cur_linking.push_back(s2); // linked point index in linking segmentation
						cur_linking.push_back(0); // lined type. 0 = start point, 1 = end point, 2 = mid point
					}
					else if (s2 == seg[s1].size()-1)
					{
						cur_linking.push_back(s1); // linked other segmentation
						cur_linking.push_back(s2); // linked point index in linking segmentation
						cur_linking.push_back(1); // lined type. 0 = start point, 1 = end point, 2 = mid point
					}
					else
					{
						cur_linking.push_back(s1); // linked other segmentation
						cur_linking.push_back(s2); // linked point index in linking segmentation
						cur_linking.push_back(2); // lined type. 0 = start point, 1 = end point, 2 = mid point
					}

					sp_linking.push_back(cur_linking);
				}
				if (cur_ep == seg[s1][s2])
				{
					std::vector<int> cur_linking;
					if (s2 == 0)
					{
						cur_linking.push_back(s1); // linked other segmentation
						cur_linking.push_back(s2); // linked point index in linking segmentation
						cur_linking.push_back(0); // lined type. 0 = start point, 1 = end point, 2 = mid point
					}
					else if (s2 == seg[s1].size() - 1)
					{
						cur_linking.push_back(s1); // linked other segmentation
						cur_linking.push_back(s2); // linked point index in linking segmentation
						cur_linking.push_back(1); // lined type. 0 = start point, 1 = end point, 2 = mid point
					}
					else
					{
						cur_linking.push_back(s1); // linked other segmentation
						cur_linking.push_back(s2); // linked point index in linking segmentation
						cur_linking.push_back(2); // lined type. 0 = start point, 1 = end point, 2 = mid point
					}

					ep_linking.push_back(cur_linking);

				}
			}

			
		}

		cur_seg_spep_linking.push_back(sp_linking);
		cur_seg_spep_linking.push_back(ep_linking);

		linked.push_back(cur_seg_spep_linking);
	}

	return linked;
}
std::vector<std::vector<std::vector<std::vector<int>>>> cVCO::get_tp1_segm_linked_information()
{
	return m_tp1_vsegm_linked_information;
}

cv::Mat cVCO::get_tp1_adjusted_vescl_mask_pp()
{
	cv::Mat tp1_vmask_pp(img_h, img_w, CV_8UC1);

	tp1_vmask_pp = 0;

	for (int i = 0; i < m_tp1_vsegm_vpt_2darr_pp.size(); i++)
	for (int j = 0; j < m_tp1_vsegm_vpt_2darr_pp[i].size(); j++)
	{
		int x = m_tp1_vsegm_vpt_2darr_pp[i][j].x;
		int y = m_tp1_vsegm_vpt_2darr_pp[i][j].y;

		tp1_vmask_pp.at<uchar>(y, x) = 255;
	}

	cP2pMatching p2p;

	cv::Mat kernel(3, 3, CV_8UC1);
	kernel = 255;
	cv::dilate(tp1_vmask_pp, tp1_vmask_pp, kernel);
	p2p.thin(tp1_vmask_pp, tp1_vmask_pp);

	cv::Mat cc;

	int ncc = cv::connectedComponents(tp1_vmask_pp, cc);

	tp1_vmask_pp = 0;

	int max_cc = 0;
	int max_cc_idx; 

	for (int i = 1; i < ncc; i++)
	{
		cv::Mat cur_cc = (cc == i);

		std::vector<cv::Point> idx;

		cv::findNonZero(cur_cc, idx);

		if (idx.size() >max_cc)
		{
			max_cc = idx.size();

			max_cc_idx = i;
		}
	}

	tp1_vmask_pp = (cc == max_cc_idx);

	return tp1_vmask_pp;
}