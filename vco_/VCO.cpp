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

			//         pt1 = all_coors(t_coors_to_all_coors(j), :);
			//         pt2 = all_coors(t_coors_to_all_coors(j + 1), :);
			//         t_dist = sqrt((pt1(1) - pt2(1)) ^ 2 + (pt1(2) - pt2(2)) ^ 2);
			//         t_pCost2 = GetIntervalCost(t_dist, dist_sigma, all_cands(t_coors_to_all_coors(j), :), all_cands(t_coors_to_all_coors(j + 1), :), ...
			//             dummy_pairwise_cost1, dummy_pairwise_cost2);

			//         t_pCost = alpha*t_pCost1 + beta*t_pCost2;

			cv::Mat t_pCost = t_pCost1;

			//for (int y = 0; y < t_pCost1.rows; y++)
			//{
			//	for (int x = 0; x < t_pCost1.cols; x++)
			//	{
			//		if (t_pCost1.at<double>(y, x) > 100000)
			//			printf("i  ");
			//		else
			//		printf("%0.1f  ", t_pCost1.at<double>(y, x));
			//	}
			//	printf("\n");
			//}

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

	//for (int y = 0; y < diff.rows; y++)
	//{
	//	for (int x = 0; x < diff.cols; x++)
	//	{
	//		if (cands1.at<cv::Point>(y, x).x == -1)
	//		{
	//			diff.at<double>(y, x) = DBL_MAX;
	//		}
	//		if (cands2.at<cv::Point>(y, x).x == -1)
	//		{
	//			diff.at<double>(y, x) = DBL_MAX;
	//		}
	//	}
	//}

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

	/* for verification
	cv::Mat tmp_juntion_view;
	img_tp1.copyTo(tmp_juntion_view);
	cv::cvtColor(tmp_juntion_view, tmp_juntion_view, CV_GRAY2BGR);
	for (int i = 0; i < E.size(); i++)
	for (int j = 0; j < E[i].size(); j++) {
	cv::circle(tmp_juntion_view, E[i][j], 2, CV_RGB(255, 0, 0), -1);
	}
	cv::imshow("tmp_juntion_view", tmp_juntion_view);
	cv::waitKey();*/

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
	//[labels, energy] = mrfMinimizeMex_syshin(unaryCost, pairwiseCost, mapMat);

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
	//std::vector<double> sigma = { 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5 };
	//std::vector<double> sigma = { 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5 };
	//std::vector<cv::Mat> result;
	//cv::Mat m_frangi_vesselness_tp1 = frangi.makeFrangiFilter(img_tp1, sigma, result);
	//cv::imshow("m_frangi_vesselness_tp1", m_frangi_vesselness_tp1);
	//cv::waitKey();
	//result.clear();
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
		//if (E[j], J, 'row) {
		//	bJ_all_pts(v_segm_to_all_coors{ j }(1)) = true;
		//}
		int n_j_segm = v_segm_pt_coors[j].size();
		int n_j_c2a = v_segm_to_all_coors[j].size();
		for (int k = 0; k < J.size(); k++)	{
			if (v_segm_pt_coors[j][n_j_segm - 1] == J[k]) {
				bJ_all_pts.at<uchar>(v_segm_to_all_coors[j][n_j_c2a - 1] - 1) = true;
				break;
			}
		}
		//if (ismember(E{ j }(end, :), J, 'rows')) {
		//	bJ_all_pts(v_segm_to_all_coors{ j }(end)) = true;
		//}
	}
	std::vector<cv::Point> bJ_idx;
	cv::findNonZero(bJ_all_pts, bJ_idx);
	//bJ_idx = find(bJ_all_pts);

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
				//find(v_segm_to_all_coors{ k } == bJ_idx(j));
				
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
	std::vector<cv::Point> all_vpts;
	std::vector<cv::Point> all_vessel_pt;
	std::vector<cv::Point> cor_line_pt;

	// make new path in tp1 frame
	mkNewPath(m_frangi_vesselness_tp1,v_segm_to_all_coors,all_cands,all_joining_seg,labels,ves_segm_num,num_all_joining_seg,
		&newE, &all_v, &all_vessel_pt, &all_vpts);


	// stored tp1 feature points
	m_tp1_feat_pts = find_tp1_features(m_t_feat_pts,
		m_t_vpt_arr,
		all_vpts);



	//stored previous segment vector 
	m_tp1_vsegm_vpt_2darr = newE;

	// stored all candidates
	m_tp1_vpt_arr = all_vpts;

	// stroed displacement vectors
	m_disp_vec_arr = makeDisplacementVec(m_t_vpt_arr, m_tp1_vpt_arr);

	if (bVerbose)
	{
		cv::Mat drawDisplacemete = drawDisplacementeVec(img_tp1, m_t_vpt_arr, m_tp1_vpt_arr, m_disp_vec_arr);
		sprintf(str, "%s%d-th_frame_displacemente.png", savePath, fidx_tp1);
		cv::imwrite(str, drawDisplacemete);
	}
		
	

	// ** convert to mkNewPath funtion **//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//cv::Mat trs;
	//m_frangi_vesselness_tp1.copyTo(trs);
	//trs.convertTo(trs, CV_64FC1);
	//cv::transpose(trs, trs);
	//trs.setTo(1e-10, trs < 1e-10);

	//cv::Mat H_mat = cv::Mat::zeros(img_h, img_w, CV_64FC1);
	//cv::Mat S_mat = cv::Mat::ones(img_h, img_w, CV_64FC1);
	//cv::Mat D_mat(img_h, img_w, CV_64FC1);
	//D_mat = 1e9;
	//cv::Mat Q_mat = -cv::Mat::ones(img_h, img_w, CV_64FC1);
	//cv::Mat PD_mat = -cv::Mat::ones(img_h, img_w, CV_64FC1);

	//cFastMarching fmm;
	//for (int j = 0; j < ves_segm_num; j++) {
	//	std::vector<cv::Point> temp_v;
	//	cv::Mat temp = cv::Mat::zeros(nY, nX, CV_64FC1);
	//	std::vector<int> t_seg = v_segm_to_all_coors[j];
	//	int len_t_seg = t_seg.size();
	//	cv::Point st_pt = cv::Point(-1, -1);
	//	cv::Point ed_pt = cv::Point(-1, -1);
	//	std::vector<cv::Point> cum_seg_path;
	//	bool is_first = true;

	//	for (int k = 0; k < len_t_seg; k++) {
	//		int t_idx = t_seg[k] - 1;
	//		double t_label = labels[t_idx] - 1;
	//		if (t_label < params.n_all_cands) {
	//			//int pt1_y = all_coors[t_idx].y; int pt1_x = all_coors[t_idx].x;
	//			//cv::Point pt2 = all_cands.at<cv::Point>(t_idx, t_label);
	//			////[pt2_y, pt2_x] = ind2sub([nY, nX], pt2);
	//			//int pt2_y = pt2.y; int pt2_x = pt2.x;
	//			////[t_path_x, t_path_y] = bresenham(pt1_x, pt1_y, pt2_x, pt2_y);
	//			//std::vector<cv::Point> t_path = p2p.bresenham(cv::Point(pt1_x,pt1_y),pt2);
	//			////cor_line_pt = [cor_line_pt;[t_path_y, t_path_x]];
	//			//for (int a = 0; a < t_path.size(); a++)
	//			//	cor_line_pt.push_back(t_path[a]);

	//			if (st_pt.x == -1) {
	//				st_pt = all_cands.at<cv::Point>(t_idx, (int)t_label);
	//				//temp_v = [temp_v; st_pt];

	//				temp_v.push_back(st_pt);
	//			}
	//			else {
	//				ed_pt = all_cands.at<cv::Point>(t_idx, (int)t_label);
	//				//temp_v = [temp_v; ed_pt];
	//				temp_v.push_back(ed_pt);
	//			}
	//		}
	//		else {
	//			ed_pt = cv::Point(-1, -1);
	//		}
	//		if (st_pt.x != -1 && ed_pt.x != -1) {
	//			//[st_pt_y, st_pt_x] = ind2sub([nY, nX], st_pt);
	//			//[ed_pt_y, ed_pt_x] = ind2sub([nY, nX], ed_pt);
	//			int st_pt_y = st_pt.y; int st_pt_x = st_pt.x;
	//			int ed_pt_y = ed_pt.y; int ed_pt_x = ed_pt.x;
	//			// straight line
	//			//[t_path_x, t_path_y] = bresenham(st_pt_x, st_pt_y, ed_pt_x, ed_pt_y);
	//			// geodesic path
	//			double pfm_end_points[] = { ed_pt_y, ed_pt_x };
	//			double pfm_start_points[] = { st_pt_y, st_pt_x };
	//			//[D, S] = perform_fast_marching(m_frangi_vesselness_tp1, pfm_start_points, pfm_end_points);
	//			double nb_iter_max = std::min(params.pfm_nb_iter_max, 
	//				(1.2*std::max(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols)*
	//				     std::max(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols)) );

	//			double *D, *S, *H;

	//			//double minvv;
	//			//double maxvv;
	//			//cv::minMaxIdx(m_frangi_vesselness_tp1, &minvv, &maxvv);
	//			//cv::Mat frangi_vesselness_tp1_view = (m_frangi_vesselness_tp1 - minvv) / (maxvv-minvv) * 255.f;
	//			//frangi_vesselness_tp1_view.convertTo(frangi_vesselness_tp1_view, CV_8UC1);
	//			//cv::cvtColor(frangi_vesselness_tp1_view, frangi_vesselness_tp1_view, CV_GRAY2BGR);

	//			//cv::circle(frangi_vesselness_tp1_view, st_pt, 2, CV_RGB(0, 0, 255),-1);
	//			//cv::circle(frangi_vesselness_tp1_view, ed_pt,2,CV_RGB(255,0,0),-1);

	//			//cv::imshow("frangi_vesselness_tp1_view", frangi_vesselness_tp1_view);
	//			//cv::waitKey();
	//			S_mat = 1;
	//			D_mat = 1e9;
	//			Q_mat = -1;
	//			PD_mat = -1;
	//			fmm.fast_marching(((double*)trs.data), m_frangi_vesselness_tp1.cols, m_frangi_vesselness_tp1.rows, pfm_start_points, 1, pfm_end_points, 1, nb_iter_max,
	//				(double*)H_mat.data, (double*)S_mat.data, (double*)D_mat.data, (double*)Q_mat.data, (double*)PD_mat.data,
	//				&D, &S);

	//			std::vector<cv::Point> geo_path;
	//			cv::Mat D_mat(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols, CV_64FC1, D);
	//			cv::transpose(D_mat, D_mat);
	//			//double minvv;
	//			//double maxvv;
	//			//int minidx1[2], maxidx1[2];
	//			//cv::minMaxIdx(D_mat,&minvv,&maxvv);
	//			//D_mat = (D_mat - minvv) / (maxvv - minvv)*255.f;
	//			//cv::Mat view_d_mat;
	//			//D_mat.convertTo(view_d_mat, CV_8UC1);
	//			//cv::imshow("view_d_mat", view_d_mat);
	//			//cv::waitKey();
	//			//fmm.compute_geodesic(D, m_frangi_vesselness_tp1.cols, m_frangi_vesselness_tp1.rows, pfm_end_points, &geo_path);
	//			fmm.compute_discrete_geodesic(D_mat, cv::Point(pfm_end_points[1], pfm_end_points[0]), &geo_path);
	//			//geo_path = compute_geodesic(D, [ed_pt_y; ed_pt_x]);
	//			//geo_path = round(geo_path);

	//			//[b, m, n] = unique(geo_path','rows','first');
	//			//geo_path = geo_path(:, sort(m))';
	//			//geo_path = flipud(fliplr(geo_path));
	//			//t_path_x = geo_path(:, 1);
	//			//t_path_y = geo_path(:, 2);

	//			if (is_first) {
	//				cum_seg_path = geo_path;
	//				//cum_seg_path = [cum_seg_path; [t_path_y, t_path_x]];

	//				is_first = false;
	//			}
	//			else {
	//				for (int a = 0; a < geo_path.size(); a++)
	//					cum_seg_path.push_back(geo_path[a]);
	//				//cum_seg_path = [cum_seg_path; [t_path_y(2:end), t_path_x(2:end)]];
	//			}

	//			st_pt = ed_pt;
	//			ed_pt = cv::Point(-1, -1);

	//			//delete[] D;
	//			//delete[] S;
	//		}
	//	}
	//	newE.push_back(cum_seg_path);
	//	if (cum_seg_path.size()) {
	//		for (int k = 0; k < temp_v.size(); k++)
	//			all_v.push_back(temp_v[k]);
	//		//all_v = [all_v; temp_v];
	//		//lidx = sub2ind([nY, nX], cum_seg_path(:, 1), cum_seg_path(:, 2));

	//		for (int k = 0; k < cum_seg_path.size(); k++) {
	//			all_vessel_pt.push_back(cum_seg_path[k]);
	//			temp.at<double>(cum_seg_path[k]) = true;
	//		}
	//		//all_vessel_pt = [all_vessel_pt; lidx];
	//	}
	//}

	//// drawing for junctions labeled as 'dummy'
	//for (int j = 0; j < num_all_joining_seg; j++)
	//{
	//	std::vector<int> joining_seg = all_joining_seg[j];
	//	int n_joining_seg = joining_seg.size();
	//	std::vector<cv::Point> cum_path;
	//	for (int k = 0; k < n_joining_seg / 2; k++)
	//	{
	//		int t_idx = v_segm_to_all_coors[joining_seg[k * 2 + 0]][joining_seg[k * 2 + 1]] - 1;
	//		int t_label = labels[t_idx] - 1;
	//		cv::Point t_coor1 = all_cands.at<cv::Point>(t_idx, t_label);
	//		//[st_pt_y, st_pt_x] = ind2sub([nY, nX], t_coor1);
	//		int st_pt_y = t_coor1.y;
	//		int st_pt_x = t_coor1.x;

	//		for (int m = k; m < n_joining_seg / 2; m++)
	//		{
	//			int t_idx = v_segm_to_all_coors[joining_seg[m * 2 + 0]][joining_seg[m * 2 + 1]] - 1;
	//			int t_label = labels[t_idx] - 1;
	//			cv::Point t_coor2 = all_cands.at<cv::Point>(t_idx, t_label);
	//			//[ed_pt_y, ed_pt_x] = ind2sub([nY, nX], t_coor2);
	//			int ed_pt_y = t_coor2.y;
	//			int ed_pt_x = t_coor2.x;

	//			// straight line
	//			//[t_path_x, t_path_y] = bresenham(st_pt_x, st_pt_y, ed_pt_x, ed_pt_y);
	//			// geodesic path

	//			double pfm_end_points[] = { ed_pt_y, ed_pt_x };
	//			double pfm_start_points[] = { st_pt_y, st_pt_x };


	//			//pfm.end_points = [ed_pt_y; ed_pt_x];

	//			double nb_iter_max = std::min(params.pfm_nb_iter_max, 1.2*std::max(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols)*std::max(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols));

	//			double *D, *S;
	//			S_mat = 1;
	//			D_mat = 1e9;
	//			Q_mat = -1;
	//			PD_mat = -1;
	//			fmm.fast_marching(((double*)trs.data), m_frangi_vesselness_tp1.cols, m_frangi_vesselness_tp1.rows, pfm_start_points, 1, pfm_end_points, 1, nb_iter_max,
	//				(double*)H_mat.data, (double*)S_mat.data, (double*)D_mat.data, (double*)Q_mat.data, (double*)PD_mat.data,
	//				&D, &S);

	//			std::vector<cv::Point> geo_path;
	//			cv::Mat D_mat(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols, CV_64FC1, D);
	//			cv::transpose(D_mat, D_mat);
	//			//fmm.compute_geodesic(D, m_frangi_vesselness_tp1.cols, m_frangi_vesselness_tp1.rows, pfm_end_points, &geo_path);
	//			fmm.compute_discrete_geodesic(D_mat, cv::Point(pfm_end_points[1], pfm_end_points[0]), &geo_path);
	//			//[D, S] = perform_fast_marching(m_frangi_vesselness_tp1, [st_pt_y; st_pt_x], pfm);
	//			//geo_path = compute_geodesic(D, [ed_pt_y; ed_pt_x]);
	//			//geo_path = round(geo_path);
	//			//[b, m, n] = unique(geo_path','rows','first');
	//			//	geo_path = geo_path(:, sort(m))';
	//			//	geo_path = flipud(fliplr(geo_path));
	//			//t_path_x = geo_path(:, 1);
	//			//t_path_y = geo_path(:, 2);
	//			for (int n = 1; n < geo_path.size() - 1; n++)
	//			{
	//				cum_path.push_back(geo_path[n]);
	//			}

	//			//delete[] D;
	//			//delete[] S;

	//			//cum_path = [cum_path;[t_path_y(2:end - 1), t_path_x(2:end - 1)]];
	//		}
	//	}
	//	if (!cum_path.size())
	//	{
	//		continue;
	//	}

	//	for (int k = 0; k < cum_path.size(); k++)
	//		all_vessel_pt.push_back(cum_path[k]);
	//	//lidx = sub2ind([nY, nX], cum_path(:, 1), cum_path(:, 2));
	//	//all_vessel_pt = [all_vessel_pt; lidx];
	//}
	//// drawing for junctions labeled as 'dummy'

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// ** END convert to mkNewPath funtion **//


	
	// **erased repeat point at all_vessel_pt** //
	std::vector<int> tmp_all_vessel_pt(all_vessel_pt.size());
	for (int avp = 0; avp < all_vessel_pt.size(); avp++)
	{
		tmp_all_vessel_pt[avp] = (all_vessel_pt[avp].y * nX + all_vessel_pt[avp].x);
	}
	tmp_all_vessel_pt.erase(std::unique(tmp_all_vessel_pt.begin(), tmp_all_vessel_pt.end()), tmp_all_vessel_pt.end());
	all_vessel_pt.clear();
	all_vessel_pt = std::vector<cv::Point>(tmp_all_vessel_pt.size());
	for (int avp = 0; avp < tmp_all_vessel_pt.size(); avp++)
	{
		all_vessel_pt[avp] = (cv::Point(tmp_all_vessel_pt[avp] % nX, tmp_all_vessel_pt[avp] / nX));
	}
	//all_vessel_pt.erase(std::unique(all_vessel_pt.begin(), all_vessel_pt.end()));
	//all_vessel_pt = unique(all_vessel_pt);

	// **END erased repaet point at all_vessel_pt** //


	cv::Mat draw_bimg_tp1(nY, nX, CV_8UC1);
	draw_bimg_tp1 = 0;
	//draw_bimg_tp1 = false(nY, nX);

	for (int avp = 0; avp < all_vessel_pt.size(); avp++) {
		draw_bimg_tp1.at<uchar>(all_vessel_pt[avp]) = 255;
	}
	//bimg_tp1(all_vessel_pt) = true;

	cv::Mat kernel(3, 3, CV_8UC1);
	kernel = 255;
	cv::dilate(draw_bimg_tp1, draw_bimg_tp1, kernel);
	p2p.thin(draw_bimg_tp1, draw_bimg_tp1);
	//bimg_tp1 = bwmorph(bimg_tp1, 'dilate');
	//bimg_tp1 = bwmorph(bimg_tp1, 'thin', Inf);
	//bimg_tp1 = bwmorph(bimg_tp1, 'fill');
	//bimg_tp1 = bwmorph(bimg_tp1, 'thin', Inf);

	std::vector<cv::Point> draw_idx;
	cv::findNonZero(draw_bimg_tp1, draw_idx);
	//all_vessel_pt = find(draw_bimg_tp1);

	all_vessel_pt = draw_idx;
	if (bVerbose) {
		sprintf(str, "%s%d-th_frame_final_b.png", savePath, fidx_tp1);
		cv::imwrite(str, draw_bimg_tp1);
	}

	if (bVerbose) {
		cv::Mat final_canvas_img;
		cv::cvtColor(img_tp1, final_canvas_img, CV_GRAY2BGR);
		//final_canvas_img = zeros(nY, nX, 3);
		//final_canvas_img(:, : , 1) = img_tp1;
		//final_canvas_img(:, : , 2) = img_tp1;
		//final_canvas_img(:, : , 3) = img_tp1;

		final_canvas_img.setTo(cv::Scalar(255, 0, 0), draw_bimg_tp1);
		sprintf(str, "%s%d-th_frame_final_rgb.png", savePath, fidx_tp1);
		cv::imwrite(str, final_canvas_img);

		for (int avp = 0; avp < all_v.size(); avp++)
		{
			final_canvas_img.at<uchar>(all_v[avp].y, all_v[avp].x * 3 + 0) = 0;
			final_canvas_img.at<uchar>(all_v[avp].y, all_v[avp].x * 3 + 1) = 0;
			final_canvas_img.at<uchar>(all_v[avp].y, all_v[avp].x * 3 + 2) = 255;
		}
		//final_canvas_img(all_v) = true;
		//final_canvas_img(all_v + nY*nX) = false;
		//final_canvas_img(all_v + 2 * nY*nX) = false;
		sprintf(str, "%s%d-th_frame_final_rgb_node.png", savePath, fidx_tp1);
		cv::imwrite(str, final_canvas_img);
	}

	///////////// post - processing///////////////

	cv::Mat tp1_post_processed = postProcGrowVessel(img_tp1, m_frangi_vesselness_tp1, all_vessel_pt, params,&newE);

	m_tp1_vsegm_vpt_2darr_pp = newE;
	// ** convert to postProcGrowVessel funtion ** //
	///////////////////////////////////////////////////////////////////////////////////////////////
	//cv::Mat new_bimg;
	//std::vector<cv::Point> new_lidx, app_lidx;

	//GrowVesselUsingFastMarching(m_frangi_vesselness_tp1, all_vessel_pt, params.thre_ivessel, 
	//	params, &new_bimg, &new_lidx, &app_lidx);


	//p2p.thin(new_bimg, new_bimg);
	//cv::Mat bimg_tp1_post_processed;
	//new_bimg.copyTo(bimg_tp1_post_processed);
	//if (bVerbose)
	//{
	//	cv::Mat final_canvas_img;
	//	sprintf(str, "%s%d-th_frame_final_post_processed_b.png", savePath, fidx_tp1);
	//	cv::imwrite(str, bimg_tp1_post_processed);

	//	cv::cvtColor(img_tp1, final_canvas_img, CV_GRAY2BGR);

	//	final_canvas_img.setTo(cv::Scalar(255, 0, 0), new_bimg);
	//	sprintf(str, "%s%d-th_frame_final_post_processed_rgb.png", savePath, fidx_tp1);
	//	cv::imwrite(str, final_canvas_img);
	//}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// ** END convert to postProcGrowVessel funtion ** //


	// ** convert to double array from Mat **//
	double *arr_tmp_bimg_tp1_nonPostproc, *arr_tmp_bimg_tp1_post_processed;
	
	cvt2Arr(draw_bimg_tp1, tp1_post_processed, &arr_tmp_bimg_tp1_nonPostproc, &arr_tmp_bimg_tp1_post_processed);
	// ** END convert to double array from Mat **//
	
	// ** convert to cvt2Arr funtion ** //
	///////////////////////////////////////////////////////////////////////////////////////////////
	//cv::Mat tmp_bimg_tp1;
	//draw_bimg_tp1.convertTo(tmp_bimg_tp1, CV_64FC1);
	//cv::Mat tmp_bimg_tp1_post_processed;
	//bimg_tp1_post_processed.convertTo(tmp_bimg_tp1_post_processed, CV_64FC1);



	//double* arr_tmp_bimg_tp1 = new double[tmp_bimg_tp1_post_processed.rows*tmp_bimg_tp1_post_processed.cols];
	//double* arr_tmp_bimg_tp1_post_processed = new double[tmp_bimg_tp1_post_processed.rows*tmp_bimg_tp1_post_processed.cols];
	//for (int y = 0; y < tmp_bimg_tp1_post_processed.rows; y++)
	//for (int x = 0; x < tmp_bimg_tp1_post_processed.cols; x++)
	//{
	//	arr_tmp_bimg_tp1[y*tmp_bimg_tp1_post_processed.cols + x] = tmp_bimg_tp1.at<double>(y, x);
	//	arr_tmp_bimg_tp1_post_processed[y*tmp_bimg_tp1_post_processed.cols + x] = tmp_bimg_tp1_post_processed.at<double>(y, x);
	//}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// ** END convert to cvt2Arr funtion ** //

	// OUTPUTS
	//*arr_bimg_tp1 = arr_tmp_bimg_tp1_nonPostproc;
	//*arr_bimg_tp1_post_processed = arr_tmp_bimg_tp1_post_processed;
	//*tp1_nonPostProc = draw_bimg_tp1;
	//*tp1_postProc = tp1_post_processed;
	//*arr_bimg_tp1 = NULL;
	//*arr_bimg_tp1_post_processed = NULL;

	m_p_tp1_vmask = arr_tmp_bimg_tp1_nonPostproc;
	m_p_tp1_vmask_pp = arr_tmp_bimg_tp1_post_processed;

	return;
}

void cVCO::cvt2Arr(cv::Mat draw_bimg_tp1, cv::Mat bimg_tp1_post_processed, double **arr_tmp_bimg_tp1_nonPostproc, double **arr_tmp_bimg_tp1_post_processed)
{
	int nX = draw_bimg_tp1.cols;
	int nY = draw_bimg_tp1.rows;
	

	cv::Mat tmp_bimg_tp1;
	draw_bimg_tp1.convertTo(tmp_bimg_tp1, CV_64FC1);
	cv::Mat tmp_bimg_tp1_post_processed;
	bimg_tp1_post_processed.convertTo(tmp_bimg_tp1_post_processed, CV_64FC1);


	double* arr_tp1nonPostProc = new double[nX*nY];
	double* arr_tp1PostProc = new double[nX*nY];

	//*arr_tmp_bimg_tp1_nonPostproc = new double[nX*nY];
	//*arr_tmp_bimg_tp1_post_processed = new double[nX*nY];
	for (int y = 0; y < nY; y++)
	for (int x = 0; x < nX; x++)
	{
		arr_tp1nonPostProc[y*nX + x] = tmp_bimg_tp1.at<double>(y, x);
		arr_tp1PostProc[y*nX + x] = tmp_bimg_tp1_post_processed.at<double>(y, x);
	}

	*arr_tmp_bimg_tp1_nonPostproc = arr_tp1nonPostProc;
	*arr_tmp_bimg_tp1_post_processed = arr_tp1PostProc;
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
	//cv::Mat gt_bimg_t;
	//int t_x, t_y;
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
		//img_tp1.convertTo(gc_canvas_img,CV_8UC3);
		//gc_canvas_img.convertTo(gc_canvas_img, CV_8UC3);
		//cv::cvtColor(gc_canvas_img, gc_canvas_img, CV_8UC3);
		//gc_canvas_img = cv::Mat(nY, nX, CV_8UC3, img_tp1.data[] , img_tp1.da);

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

void cVCO::mkNewPath(cv::Mat m_frangi_vesselness_tp1, std::vector<std::vector<int>> v_segm_to_all_coors, cv::Mat all_cands, 
	std::vector<std::vector<int>> all_joining_seg, double* labels, int ves_segm_num, int num_all_joining_seg,
	std::vector<std::vector<cv::Point>> *newE, std::vector<cv::Point> *all_v, std::vector<cv::Point> *all_vessel_pt,
	std::vector<cv::Point> *tp1_vpts)
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
				//int pt1_y = all_coors[t_idx].y; int pt1_x = all_coors[t_idx].x;
				//cv::Point pt2 = all_cands.at<cv::Point>(t_idx, t_label);
				////[pt2_y, pt2_x] = ind2sub([nY, nX], pt2);
				//int pt2_y = pt2.y; int pt2_x = pt2.x;
				////[t_path_x, t_path_y] = bresenham(pt1_x, pt1_y, pt2_x, pt2_y);
				//std::vector<cv::Point> t_path = p2p.bresenham(cv::Point(pt1_x,pt1_y),pt2);
				////cor_line_pt = [cor_line_pt;[t_path_y, t_path_x]];
				//for (int a = 0; a < t_path.size(); a++)
				//	cor_line_pt.push_back(t_path[a]);

				if (st_pt.x == -1) {
					st_pt = all_cands.at<cv::Point>(t_idx, (int)t_label);
					//temp_v = [temp_v; st_pt];

					temp_v.push_back(st_pt);
					temp_vpts.push_back(st_pt);
				}
				else {
					ed_pt = all_cands.at<cv::Point>(t_idx, (int)t_label);
					//temp_v = [temp_v; ed_pt];
					temp_v.push_back(ed_pt);
					temp_vpts.push_back(ed_pt);
				}
			}
			else {
				ed_pt = cv::Point(-1, -1);
			}
			if (st_pt.x != -1 && ed_pt.x != -1) {
				//[st_pt_y, st_pt_x] = ind2sub([nY, nX], st_pt);
				//[ed_pt_y, ed_pt_x] = ind2sub([nY, nX], ed_pt);
				int st_pt_y = st_pt.y; int st_pt_x = st_pt.x;
				int ed_pt_y = ed_pt.y; int ed_pt_x = ed_pt.x;
				// straight line
				//[t_path_x, t_path_y] = bresenham(st_pt_x, st_pt_y, ed_pt_x, ed_pt_y);
				// geodesic path
				double pfm_end_points[] = { ed_pt_y, ed_pt_x };
				double pfm_start_points[] = { st_pt_y, st_pt_x };
				//[D, S] = perform_fast_marching(m_frangi_vesselness_tp1, pfm_start_points, pfm_end_points);
				double nb_iter_max = std::min(params.pfm_nb_iter_max,
					(1.2*std::max(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols)*
					std::max(m_frangi_vesselness_tp1.rows, m_frangi_vesselness_tp1.cols)));

				double *D, *S, *H;

				//double minvv;
				//double maxvv;
				//cv::minMaxIdx(m_frangi_vesselness_tp1, &minvv, &maxvv);
				//cv::Mat frangi_vesselness_tp1_view = (m_frangi_vesselness_tp1 - minvv) / (maxvv-minvv) * 255.f;
				//frangi_vesselness_tp1_view.convertTo(frangi_vesselness_tp1_view, CV_8UC1);
				//cv::cvtColor(frangi_vesselness_tp1_view, frangi_vesselness_tp1_view, CV_GRAY2BGR);

				//cv::circle(frangi_vesselness_tp1_view, st_pt, 2, CV_RGB(0, 0, 255),-1);
				//cv::circle(frangi_vesselness_tp1_view, ed_pt,2,CV_RGB(255,0,0),-1);

				//cv::imshow("frangi_vesselness_tp1_view", frangi_vesselness_tp1_view);
				//cv::waitKey();
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
				//double minvv;
				//double maxvv;
				//int minidx1[2], maxidx1[2];
				//cv::minMaxIdx(D_mat,&minvv,&maxvv);
				//D_mat = (D_mat - minvv) / (maxvv - minvv)*255.f;
				//cv::Mat view_d_mat;
				//D_mat.convertTo(view_d_mat, CV_8UC1);
				//cv::imshow("view_d_mat", view_d_mat);
				//cv::waitKey();
				//fmm.compute_geodesic(D, m_frangi_vesselness_tp1.cols, m_frangi_vesselness_tp1.rows, pfm_end_points, &geo_path);
				fmm.compute_discrete_geodesic(D_mat, cv::Point(pfm_end_points[1], pfm_end_points[0]), &geo_path);
				//geo_path = compute_geodesic(D, [ed_pt_y; ed_pt_x]);
				//geo_path = round(geo_path);

				//[b, m, n] = unique(geo_path','rows','first');
				//geo_path = geo_path(:, sort(m))';
				//geo_path = flipud(fliplr(geo_path));
				//t_path_x = geo_path(:, 1);
				//t_path_y = geo_path(:, 2);

				if (is_first) {
					cum_seg_path = geo_path;
					//cum_seg_path = [cum_seg_path; [t_path_y, t_path_x]];

					is_first = false;
				}
				else {
					for (int a = 0; a < geo_path.size(); a++)
						cum_seg_path.push_back(geo_path[a]);
					//cum_seg_path = [cum_seg_path; [t_path_y(2:end), t_path_x(2:end)]];
				}

				st_pt = ed_pt;
				ed_pt = cv::Point(-1, -1);

				//delete[] D;
				//delete[] S;
			}
		}
		newE->push_back(cum_seg_path);
		for (int k = 0; k < temp_vpts.size(); k++)
		{
			tp1_vpts->push_back(temp_vpts[k]);
		}
		if (cum_seg_path.size()) {
			for (int k = 0; k < temp_v.size(); k++)
			{
				all_v->push_back(temp_v[k]);
			}
			
			//all_v = [all_v; temp_v];
			//lidx = sub2ind([nY, nX], cum_seg_path(:, 1), cum_seg_path(:, 2));

			for (int k = 0; k < cum_seg_path.size(); k++) {
				all_vessel_pt->push_back(cum_seg_path[k]);
				temp.at<double>(cum_seg_path[k]) = true;
			}
			//all_vessel_pt = [all_vessel_pt; lidx];
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
			//[st_pt_y, st_pt_x] = ind2sub([nY, nX], t_coor1);
			int st_pt_y = t_coor1.y;
			int st_pt_x = t_coor1.x;

			for (int m = k; m < n_joining_seg / 2; m++)
			{
				int t_idx = v_segm_to_all_coors[joining_seg[m * 2 + 0]][joining_seg[m * 2 + 1]] - 1;
				int t_label = labels[t_idx] - 1;
				cv::Point t_coor2 = all_cands.at<cv::Point>(t_idx, t_label);
				//[ed_pt_y, ed_pt_x] = ind2sub([nY, nX], t_coor2);
				int ed_pt_y = t_coor2.y;
				int ed_pt_x = t_coor2.x;

				// straight line
				//[t_path_x, t_path_y] = bresenham(st_pt_x, st_pt_y, ed_pt_x, ed_pt_y);
				// geodesic path

				double pfm_end_points[] = { ed_pt_y, ed_pt_x };
				double pfm_start_points[] = { st_pt_y, st_pt_x };


				//pfm.end_points = [ed_pt_y; ed_pt_x];

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
				//fmm.compute_geodesic(D, m_frangi_vesselness_tp1.cols, m_frangi_vesselness_tp1.rows, pfm_end_points, &geo_path);
				fmm.compute_discrete_geodesic(D_mat, cv::Point(pfm_end_points[1], pfm_end_points[0]), &geo_path);
				//[D, S] = perform_fast_marching(m_frangi_vesselness_tp1, [st_pt_y; st_pt_x], pfm);
				//geo_path = compute_geodesic(D, [ed_pt_y; ed_pt_x]);
				//geo_path = round(geo_path);
				//[b, m, n] = unique(geo_path','rows','first');
				//	geo_path = geo_path(:, sort(m))';
				//	geo_path = flipud(fliplr(geo_path));
				//t_path_x = geo_path(:, 1);
				//t_path_y = geo_path(:, 2);
				for (int n = 1; n < geo_path.size() - 1; n++)
				{
					cum_path.push_back(geo_path[n]);
				}

				//delete[] D;
				//delete[] S;

				//cum_path = [cum_path;[t_path_y(2:end - 1), t_path_x(2:end - 1)]];
			}
		}
		if (!cum_path.size())
		{
			continue;
		}

		for (int k = 0; k < cum_path.size(); k++)
			all_vessel_pt->push_back(cum_path[k]);
		//lidx = sub2ind([nY, nX], cum_path(:, 1), cum_path(:, 2));
		//all_vessel_pt = [all_vessel_pt; lidx];
	}
	// drawing for junctions labeled as 'dummy'


	*newE = eraseRepeatSegPts(*newE, nX,nY);

}

cv::Mat cVCO::postProcGrowVessel(cv::Mat img_tp1, cv::Mat m_frangi_vesselness_tp1, std::vector<cv::Point> all_vessel_pt,
	cVCOParams params, std::vector<std::vector<cv::Point>> *E)
{
	cv::Mat new_bimg;
	cv::Mat bimg_tp1_post_processed;
	std::vector<cv::Point> new_lidx, app_lidx;

	GrowVesselUsingFastMarching(m_frangi_vesselness_tp1, all_vessel_pt, params.thre_ivessel,
		params, &new_bimg, &new_lidx, &app_lidx);

	cv::Mat img_add_seg(new_bimg.rows, new_bimg.cols, CV_8UC1);
	img_add_seg = 0;

	for (int i = 0; i < app_lidx.size(); i++)
	{
		img_add_seg.at<uchar>(app_lidx[i].y, app_lidx[i].x) = 255;

	}

	cP2pMatching p2p;
	p2p.thin(img_add_seg, img_add_seg);

	cv::Mat CCA;
	int nCCA = cv::connectedComponents(img_add_seg, CCA);

	for (int i = 1; i < nCCA; i++)
	{
		cv::Mat cur_CCA = CCA == i;

		std::vector<cv::Point> cur_idx;
		cv::findNonZero(cur_CCA, cur_idx);

		E->push_back(cur_idx);
	}
	
	
	
	

	p2p.thin(new_bimg, new_bimg);
	new_bimg.copyTo(bimg_tp1_post_processed);

	char str[100];

	if (bVerbose)
	{
		cv::Mat final_canvas_img;
		sprintf(str, "%s%d-th_frame_final_post_processed_b.png", savePath, fidx_tp1);
		cv::imwrite(str, bimg_tp1_post_processed);

		cv::cvtColor(img_tp1, final_canvas_img, CV_GRAY2BGR);

		final_canvas_img.setTo(cv::Scalar(255, 0, 0), new_bimg);
		sprintf(str, "%s%d-th_frame_final_post_processed_rgb.png", savePath, fidx_tp1);
		cv::imwrite(str, final_canvas_img);
	}

	return bimg_tp1_post_processed;

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


	cFastMarching ffm;


	bool verbose = false;
	bool IS3D = false;

	int nY = ivessel.rows;
	int nX = ivessel.cols;
	//[nY, nX] = size(ivessel);

	// Convert double image to logical
	cv::Mat Ibin = ivessel >= thre;
	cv::Mat CC;
	int numCC = cv::connectedComponents(Ibin, CC);
	//CC = bwconncomp(Ibin);
	//numCC = CC.NumObjects;

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
		//Lia = ismember(CC.PixelIdxList{ i }, lidx);


		//if (~isempty(find(Lia)))
		if (!isempty)
		{
			bROI.at<double>(i) = 1;
			for (int j = 0; j < cc_idx.size(); j++)
			{
				Ibin.at<uchar>(cc_idx[j]) = 255;
			}
			//Ibin.at<double>(CC.PixelIdxList{ i }) = true;
		}
	}

	// Distance to vessel boundary
	cv::Mat BoundaryDistance;
	getBoundaryDistance(Ibin, IS3D, &BoundaryDistance);
	if (verbose){
		//disp('Distance Map Constructed');
	}

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

	//[yy, xx] = ind2sub([nY, nX], lidx);
	//SourcePoint = [yy';xx'];

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

		if (verbose)
		{

			//disp(['Find Branches Iterations : ' num2str(itt)]);
		}

		// Do fast marching using the maximum distance value in the image
		// and the points describing all found branches are sourcepoints.
		//[T, Y] = msfm(SpeedImage, SourcePoint, false, false);

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


		//[T, S] = perform_fast_marching(ivessel, SourcePoint);

		cv::Mat tmp = cv::Mat::zeros(nY, nX, CV_8UC1);
		for (int i = 0; i < SourcePoint.size(); i++)
			tmp.at<uchar>(SourcePoint[i]) = 255;


		cv::Mat Y;
		cv::distanceTransform(~tmp, Y, cv::DistanceTypes::DIST_L2, cv::DistanceTransformMasks::DIST_MASK_3, CV_32FC1);

		// Trace a branch back to the used sourcepoints
		cv::Point StartPoint;
		double dummyv;
		maxDistancePoint(Y, Ibin, IS3D, &StartPoint, &dummyv);
		//StartPoint = maxDistancePoint(Y, Ibin, IS3D);

		double endpt[2] = { StartPoint.y, StartPoint.x };
		ffm.fast_marching(((double*)trs.data), ivessel.cols, ivessel.rows, arrSourcePoint, SourcePoint.size(), endpt, 1, nb_iter_max,
			(double*)H_mat.data, (double*)S_mat.data, (double*)D_mat.data, (double*)Q_mat.data, (double*)PD_mat.data,
			&D, &S);

		//ShortestLine = shortestpath(T, StartPoint, SourcePoint, 1, 'rk4');

		cv::Mat D_mat(nY, nX, CV_64FC1, D);
		cv::transpose(D_mat, D_mat);
		std::vector<cv::Point> ShortestLine;
		ffm.compute_discrete_geodesic(D_mat, StartPoint, &ShortestLine);
		//ShortestLine = round(compute_geodesic(T, StartPoint)); %% kjn
		//ShortestLine = ShortestLine';

		ShortestLine.erase(std::unique(ShortestLine.begin(), ShortestLine.end()), ShortestLine.end());
		//ShortestLine = unique(ShortestLine, 'rows', 'first'); %% kjn


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
		//SourcePoint = [SourcePoint ShortestLine'];
		for (int i = 0; i < ShortestLine.size(); i++)
			SourcePoint.push_back(ShortestLine[i]);

		//delete[] D;
		//delete[] S;
	}
	//SkeletonSegments(itt + 1:end) = [];
	std::vector<std::vector<cv::Point>> tmp_SkeletonSegments;
	for (int i = 0; i < itt; i++)
		tmp_SkeletonSegments.push_back(SkeletonSegments[i]);

	SkeletonSegments.clear();
	SkeletonSegments = tmp_SkeletonSegments;
	tmp_SkeletonSegments.clear();

	std::vector<cv::Point> lidx_app;
	if (SkeletonSegments.size())
	{
		//S = OrganizeSkeleton(SkeletonSegments, IS3D);
		if (verbose)
		{
			//disp(['Skeleton Branches Found : ' num2str(length(S))]);
		}


		for (int i = 0; i < SkeletonSegments.size(); i++)
		{
			std::vector<cv::Point> L = SkeletonSegments[i];
			/*[b, m, n] = unique(L','rows','first');
			L = L(:, sort(m));*/
			//SkeletonSegments{ i } = L;
			for (int j = 0; j < L.size(); j++)
				lidx_app.push_back(L[j]);
		}

	}

	//// Display the skeleton
	// figure, imshow(Ibin); hold on;
	// for i = 1:length(S)
	// L = S{ i };
	// plot(L(:, 2), L(:, 1), '-', 'Color', rand(1, 3));
	// end

	std::vector<cv::Point> app_lidx = lidx_app;
	std::vector<cv::Point> new_lidx = lidx;


	for (int i = 0; i < lidx_app.size(); i++)
	{

		new_lidx.push_back(lidx_app[i]);

	}
	cv::Mat new_bimg = cv::Mat::zeros(nY, nX, CV_8UC1);
	for (int i = 0; i < new_lidx.size(); i++)
		new_bimg.at<uchar>(new_lidx[i]) = 255;
	//new_bimg.setTo(255,new_lidx);

	*o_new_bimg = new_bimg;
	*o_new_lidx = new_lidx;
	*o_app_lidx = app_lidx;

}

void cVCO::GetLineLength(std::vector<cv::Point> L, bool IS3D, double *o_ll)
{
	//function ll = GetLineLength(L, IS3D)

	//std::vector<double> dist;
	double ll = 0;
	for (int i = 0; i < L.size() - 1; i++)
	{
		double cur_dist = std::sqrt((L[i + 1].y - L[i].y)*(L[i + 1].y - L[i].y) + (L[i + 1].x - L[i].x)*(L[i + 1].x - L[i].x));
		//dist.push_back(cur_dist);
		ll += cur_dist;
	}

	//dist = sqrt((L(2:end, 1) - L(1:end - 1, 1)). ^ 2 + ...
	//	(L(2:end, 2) - L(1:end - 1, 2)). ^ 2);

	/*ll = sum(dist);*/

	*o_ll = ll;

}

//void OrganizeSkeleton(SkeletonSegments, IS3D, *o_S)
//{
//	//function S = OrganizeSkeleton(SkeletonSegments, IS3D)
//	int n = length(SkeletonSegments);
//	if (IS3D)
//		Endpoints = zeros(n * 2, 3);
//	else
//		Endpoints = zeros(n * 2, 2);
//
//	l = 1;
//	for (w = 1 : n)
//	{
//		ss = SkeletonSegments{ w };
//		l = max(l, length(ss));
//		Endpoints(w * 2 - 1, :) = ss(1, :);
//		Endpoints(w * 2, :) = ss(end, :);
//	}
//	CutSkel = spalloc(size(Endpoints, 1), l, 10000);
//	ConnectDistance = 2 ^ 2;
//
//	for (w = 1 : n)
//	{
//		ss = SkeletonSegments{ w };
//		ex = repmat(Endpoints(:, 1), 1, size(ss, 1));
//		sx = repmat(ss(:, 1)',size(Endpoints,1),1);
//			ey = repmat(Endpoints(:, 2), 1, size(ss, 1));
//		sy = repmat(ss(:, 2)',size(Endpoints,1),1);
//		if (IS3D)
//		{
//			ez = repmat(Endpoints(:, 3), 1, size(ss, 1));
//			sz = repmat(ss(:, 3)',size(Endpoints,1),1);
//		}
//		if (IS3D)
//			D = (ex - sx). ^ 2 + (ey - sy). ^ 2 + (ez - sz). ^ 2;
//		else
//			D = (ex - sx). ^ 2 + (ey - sy). ^ 2;
//
//		check = min(D, [], 2)<ConnectDistance;
//		check(w * 2 - 1) = false; check(w * 2) = false;
//		if (any(check))
//		{
//
//			j = find(check);
//			for (i = 1 : length(j))
//			{
//				line = D(j(i), :);
//				[foo, k] = min(line);
//				if ((k>2) && (k < (length(line) - 2)))
//				{
//					CutSkel(w, k) = 1;
//				}
//			}
//		}
//	}
//
//	pp = 0;
//	for (w = 1 : n)
//	{
//		ss = SkeletonSegments{ w };
//		r = [1 find(CutSkel(w, :)) length(ss)];
//		for (i = 1 : length(r) - 1)
//		{
//
//			pp = pp + 1;
//			S{ pp } = ss(r(i) :r(i + 1), : );
//		}
//	}
//
//	*o_S = S;
//}

void cVCO::getBoundaryDistance(cv::Mat I, bool IS3D, cv::Mat *o_BoundaryDistance)
{
	//function BoundaryDistance = getBoundaryDistance(I, IS3D)
	// Calculate Distance to vessel boundary

	// Set all boundary pixels as fastmarching source - points(distance = 0)

	cv::Mat S(3, 3, CV_64FC1);
	S = 255;
	//if (IS3D)
	//{ 
	//	S = ones(3, 3, 3); 
	//}
	//else 
	//{ 
	//	S = ones(3, 3);
	//}
	cv::Mat B;
	cv::Mat tmp_I;

	I.copyTo(tmp_I);
	cv::dilate(tmp_I, tmp_I, S);
	B = tmp_I^I;
	//B = xor(I, imdilate(I, S));

	cv::Mat BoundaryDistance;
	//B.convertTo(B, CV_32FC1);
	cv::distanceTransform(~B, BoundaryDistance, cv::DistanceTypes::DIST_L2, cv::DistanceTransformMasks::DIST_MASK_3, CV_32FC1);

	//std::vector<cv::Point> SourcePoint;
	//cv::findNonZero(B, SourcePoint);
	//ind = find(B(:));

	//if (IS3D)
	//{
	//	[x, y, z] = ind2sub(size(B), ind);
	//	SourcePoint = [x(:) y(:) z(:)]';
	//}
	//else
	//{
	//	[x, y] = ind2sub(size(B), ind);
	//	SourcePoint = [x(:) y(:)]';
	//}

	// Calculate Distance to boundarypixels for every voxel in the volume
	/*cv::Mat SpeedImage(I.rows, I.cols, CV_64FC1);
	SpeedImage = 1;*/
	//SpeedImage = ones(size(I));


	//BoundaryDistance = msfm(SpeedImage, SourcePoint, false, true);

	// Mask the result by the binary input image
	BoundaryDistance.setTo(0, ~I);

	*o_BoundaryDistance = BoundaryDistance;
}

void cVCO::maxDistancePoint(cv::Mat BoundaryDistance, cv::Mat I, bool IS3D, cv::Point *o_posD, double *o_maxD)
{
	//function[posD, maxD] = maxDistancePoint(BoundaryDistance, I, IS3D)
	// Mask the result by the binary input image
	BoundaryDistance.setTo(0, ~I);

	// Find the maximum distance voxel
	//[maxD, ind] = max(BoundaryDistance(:));
	double maxv, minv;
	int maxidx[2];
	cv::minMaxIdx(BoundaryDistance, &minv, &maxv, 0, maxidx);

	//if (~isfinite(maxD))
	//{
	//	error('Skeleton:Maximum', 'Maximum from MSFM is infinite !');
	//}


	//[x, y] = ind2sub(size(I), ind); posD = [x; y];

	cv::Point posD(maxidx[1], maxidx[0]);

	*o_maxD = maxv;
	*o_posD = posD;

}
//VCO_EXPORTS void maxDistancePoint(cv::Mat BoundaryDistance, cv::Mat I, bool IS3D, double *o_maxD)
//{
//	//function[posD, maxD] = maxDistancePoint(BoundaryDistance, I, IS3D)
//	// Mask the result by the binary input image
//	BoundaryDistance.setTo(0, ~I);
//
//	// Find the maximum distance voxel
//	//[maxD, ind] = max(BoundaryDistance(:));
//	double maxv, minv;
//	int maxidx[2];
//	cv::minMaxIdx(BoundaryDistance, &minv, &maxv, 0, maxidx);
//
//	//if (~isfinite(maxD))
//	//{
//	//	error('Skeleton:Maximum', 'Maximum from MSFM is infinite !');
//	//}
//
//
//	//[x, y] = ind2sub(size(I), ind); posD = [x; y];
//
//	cv::Point posD(maxidx[1], maxidx[0]);
//
//	*o_maxD = maxv;
//
//
//}


//void cVCO::GetIntervalCost()
////function cost_mat = GetIntervalCost(pre_dist, dist_sigma, cands1, cands2, dummy_pairwise_cost1, dummy_pairwise_cost2)
//{
//
//	nY = 512; nX = 512;
//	cand_num = length(cands1);
//	nLabel = cand_num + 1;
//	[cands1_yy, cands1_xx] = ind2sub([nY, nX], cands1);
//	[cands2_yy, cands2_xx] = ind2sub([nY, nX], cands2);
//	diff_y = repmat(cands2_yy, cand_num, 1) - repmat(cands1_yy',1,cand_num);
//		diff_x = repmat(cands2_xx, cand_num, 1) - repmat(cands1_xx',1,cand_num);
//		diff = sqrt(diff_x. ^ 2 + diff_y. ^ 2);
//
//	cost_mat = dummy_pairwise_cost1*ones(nLabel, nLabel);
//	fun = @(x)1 - exp(-(x - pre_dist). ^ 2 * 0.5*dist_sigma^-2);
//	diff = fun(diff)*dummy_pairwise_cost2;
//
//	diff(cands1 == 0, :) = inf; % needless
//		diff(:, cands2 == 0) = inf; % needless
//		cost_mat(1:cand_num, 1 : cand_num) = diff;
//
//	
//}

//void cVCO::unique(cv::Mat inputMat, std::vector<cv::Point> *o_uniqueSotrtPts, std::vector<int> *o_uniqueSotrtIdx)
//{
//	cv::Mat copyMat;
//	inputMat.copyTo(copyMat);
//	std::vector<double> uniqueValue;
//	std::vector<int> uniqueIdx;
//	
//	for (int i = 0; i < copyMat.cols; i++)
//	{
//		double curValue = copyMat.at<cv::Point>(0, i).y * 512 + copyMat.at<cv::Point>(0, i).x;
//		std::vector<cv::Point> idx;
//		cv::Mat repeatValueMat = copyMat == curValue;
//		cv::findNonZero(repeatValueMat, idx);
//		if (idx.size() == 1)
//		{
//			uniqueValue.push_back(curValue);
//			uniqueIdx.push_back(i);
//			break;
//		}
//
//		else
//		{
//			uniqueValue.push_back(idx[0]);
//			uniqueIdx.push_back(i);
//		}
//	}
//	std::vector<double> sorting;
//	std::vector<int> sortingIdx;
//	sorting = uniqueIdx;
//	sortingIdx = uniqueIdx;
//
//	for (int i = 0; i < sorting.size()-1; i++)
//	{
//		for (int j = 0; j < sorting.size() - 1; j++)
//		{
//			if (sorting[j] < sorting[j+1])
//			{
//				double tmp = sorting[j];
//				sorting[j] = sorting[j + 1];
//				sorting[j + 1] = tmp;
//
//				int tmpIdx = sortingIdx[j];
//				sortingIdx[j] = sortingIdx[j + 1];
//				sortingIdx[j + 1] = tmpIdx;
//			}
//		}
//	}
//
//	std::vector<cv::Point> uniqueSotingPts(sorting.size());
//	for (int i = 0; i < sorting.size(); i++)
//	{
//		uniqueSotingPts[i] = cv::Point(((int)sorting[i] / 512), ((int)sorting[i] % 512));
//	}
//
//	o_uniqueSotrtPts = &uniqueSotingPts;
//	o_uniqueSotrtIdx = &sortingIdx;
//}



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

std::vector<cv::Point> cVCO::makeSegvec2Allvec(cv::Mat allMat)
{
	//int size = allMat.rows;

	//std::vector<cv::Point> allVec;

	//for (int i = 0; i < size; i++)
	//{
	//	for (int j = 0; j < segVec[i].size(); j++)
	//	{
	//		allVec.push_back(segVec[i][j]);
	//	}
	//}

	//return allVec;

	return std::vector<cv::Point>();
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
		//palette.at<uchar>(dispVec[i].y, dispVec[i].y * 3 + 0) = 0;
		//palette.at<uchar>(dispVec[i].y, dispVec[i].y * 3 + 1) = 0;
		//palette.at<uchar>(dispVec[i].y, dispVec[i].y * 3 + 2) = 255;
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

std::vector<std::vector<cv::Point>> cVCO::eraseRepeatSegPts(std::vector<std::vector<cv::Point>> segVec, int nX, int nY)
{
	cP2pMatching p2p;

	cv::Mat kernel(3, 3, CV_8UC1);
	kernel = 255;

	std::vector<std::vector<cv::Point>> smoothLine_seg;

	for (int i = 0; i < segVec.size(); i++)
	{
		cv::Mat cur_seg_draw(nY,nX,CV_8UC1);
		cur_seg_draw = 0;
		for (int j = 0; j < segVec[i].size(); j++)
		{
			cur_seg_draw.at<uchar>(segVec[i][j].y, segVec[i][j].x) = 255;
		}

		
		cv::dilate(cur_seg_draw, cur_seg_draw, kernel);
		p2p.thin(cur_seg_draw, cur_seg_draw);

		std::vector<cv::Point> cur_lineSmooth_seg;
		cv::findNonZero(cur_seg_draw, cur_lineSmooth_seg);

		
		smoothLine_seg.push_back(cur_lineSmooth_seg);
	}

	return smoothLine_seg;
}

cv::Mat cVCO::get_tp1_Mask_pp()
{
	cv::Mat tp1_vmask_pp(512, 512, CV_64FC1, m_p_tp1_vmask_pp);

	tp1_vmask_pp.convertTo(tp1_vmask_pp,CV_8UC1);
	
	return tp1_vmask_pp;
}

cv::Mat cVCO::get_tp1_Mask()
{
	cv::Mat tp1_vmask(512, 512, CV_64FC1, m_p_tp1_vmask);

	tp1_vmask.convertTo(tp1_vmask, CV_8UC1);

	return tp1_vmask;
}


double* cVCO::get_p_tp1_mask_pp()
{
	return m_p_tp1_vmask_pp;
}

double* cVCO::get_p_tp1_mask()
{
	return m_p_tp1_vmask;
}

std::vector<cVCO::ves_feat_info> cVCO::get_t_VesFeatPts()
{
	return m_t_feat_pts;
}
std::vector<cVCO::ves_feat_info> cVCO::get_tp1_VesFeatPts()
{
	return m_tp1_feat_pts;
}

std::vector<cVCO::ves_feat_info>cVCO::setVesFeatPts(std::vector<cv::Point> junction, std::vector<cv::Point> end, cv::Point tans)
{ 
	int junctionSize = junction.size();
	int endSize = end.size();

	std::vector<cVCO::ves_feat_info> feat_pts;
	

	// stored junction feature points 
	for (int i = 0; i < junctionSize; i++)
	{
		cVCO::ves_feat_info cur_feat;
		cur_feat.x = junction[i].x - tans.x;
		cur_feat.y = junction[i].y - tans.y;
		cur_feat.type = 1;
		feat_pts.push_back(cur_feat);
	}

	// stored end feature points 
	for (int i = 0; i < endSize; i++)
	{
		cVCO::ves_feat_info cur_feat;
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
		p_tp1_vmask_8u[i] = m_p_tp1_vmask[i];
	}
	return p_tp1_vmask_8u;
}

unsigned char* cVCO::get_p_tp1_mask_pp_8u()
{
	unsigned char* p_tp1_vmask_pp_8u = new unsigned char[img_h*img_w];

	for (int i = 0; i < img_h*img_w; i++)
	{
		p_tp1_vmask_pp_8u[i] = (unsigned char)m_p_tp1_vmask_pp[i];
	}

	return p_tp1_vmask_pp_8u;
}

std::vector<cVCO::ves_feat_info> cVCO::find_tp1_features(std::vector<cVCO::ves_feat_info> t_features, 
	std::vector<std::vector<cv::Point>> t_seg_vec,
	std::vector<std::vector<cv::Point>> tp1_seg_vec)
{
	std::vector<cVCO::ves_feat_info> features;
	std::vector<cv::Point> check;
	for (int i = 0; i < t_features.size(); i++)
	{
		for (int j = 0; j < t_seg_vec.size(); j++)
		{
			if (t_seg_vec[j][0].x == t_features[i].x &&
				t_seg_vec[j][0].y == t_features[i].y)
			{
				cVCO::ves_feat_info cur_feature;
				cur_feature.x = tp1_seg_vec[j][0].x;
				cur_feature.y = tp1_seg_vec[j][0].y;
				cur_feature.type = t_features[i].type;
				
				features.push_back(cur_feature);
				check.push_back(cv::Point(t_features[i].x, t_features[i].y));
			}
			if (t_seg_vec[j][t_seg_vec[j].size() - 1].x == t_features[i].x &&
				t_seg_vec[j][t_seg_vec[j].size() - 1].y == t_features[i].y)
			{
				cVCO::ves_feat_info cur_feature;
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
std::vector<cVCO::ves_feat_info> cVCO::find_tp1_features(std::vector<cVCO::ves_feat_info> t_features,
	std::vector<cv::Point> t_vseg,
	std::vector<cv::Point> tp1_vseg)
{
	std::vector<cVCO::ves_feat_info> features;
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
				cVCO::ves_feat_info cur_feature;
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
				cVCO::ves_feat_info cur_feature;
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