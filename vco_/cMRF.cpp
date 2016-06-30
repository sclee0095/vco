#include "cMRF.h"

typedef int mwSize;
typedef int mwIndex;

cMRF::cMRF()
{
}


cMRF::~cMRF()
{
}

void cMRF::computeCost(std::vector<std::vector<cv::Point>> cell_coors, std::vector<cv::Mat> cell_cands, std::vector<cv::Mat> cell_cands_dists, cParam p,
	cv::Mat *o_unaryCost, std::vector<cv::Mat> * o_pairwiseCost, cv::Mat* o_mapMat,
	std::vector<cv::Point> * o_all_coors, cv::Mat *o_all_cands, std::vector<std::vector<int>>  *o_cell_coors_to_all_coors)
{
	//function[unaryCost, pairwiseCost, mapMat, all_coors, all_cands, cell_coors_to_all_coors] = ComputeCosts2(cell_coors, cell_cands, cell_cands_dists, p)
	//	% input
	//	%
	//	% cell_coors : cell for paths of each segment, nSegm * 1 cell, nSegm is the number of segments
	//	%           each segment has(nPT*d) values
	//	% cell_cands : this contains candidate points per each(sampled) point
	//	%           in each segment, nSegm * 1 cell, nSegm is the number of segments
	//	% cell_cands_dists : corresponding(unary) costs, nSegm * 1 cell,
	//	%                   nSegm is the number of segments
	//	% p : parameters
	//	%
	//	% output
	//	%
	//	% unaryCost : unary costs, nLabel*nNode
	//	% pairwiseCost : pairwise costs, nEdge * 1, each element is size of
	//	% (nLabel*nLabel)
	//	% mapMat : mapping indices for 'pairwiseCost', 'mapMat(i.j) = k' means that
	//	%       there is an edge between node i & j and its costs are in pairwiseCost{ k }
	//% all_coors :
	//	% all_cands :
	//	% cell_coors_to_all_coors :
	//	%
	//	% coded by syshin(160130)

	// params
	int unary_thre = 800;
	int unary_trun_thre = 750;
	int dummy_unary_cost = unary_thre;
	int dummy_pairwise_cost1 = 75;
	int dummy_pairwise_cost2 = 4 * dummy_pairwise_cost1;
	double dist_sigma = p.sampling_period / (double)3;
	double alpha = 0.5;
	double beta = 0.5;


	// add nodes
	int nSegm = cell_coors.size();
	int nCand = cell_cands[0].cols;
	int nLabel = nCand + 1;
	std::vector<cv::Point> all_coors;
	//std::vector<cv::Mat> all_cands;
	cv::Mat all_cands;
	cv::Mat all_cands_dists;
	for (int i = 0; i < cell_coors[0].size(); i++)
	{
		all_coors.push_back(cell_coors[0][i]);
	}
	//all_cands.push_back(cell_cands[0]);
	cell_cands[0].copyTo(all_cands);
	//all_cands_dists.push_back(cell_cands_dists[0]);
	cell_cands_dists[0].copyTo(all_cands_dists);
	all_cands_dists.convertTo(all_cands_dists, CV_64FC1);

	std::vector<std::vector<int>> cell_coors_to_all_coors(nSegm);

	//cell_coors_to_all_coors[0] = [1:size(cell_coors{ 1 }, 1)]';
	for (int i = 0; i < cell_coors[0].size(); i++)
	{
		cell_coors_to_all_coors[0].push_back(i+1);
	}

	for (int i = 1; i < nSegm; i++) /// for each segment
	{

		std::vector<cv::Point> temp = cell_coors[i];

		std::vector<int> Lia(temp.size());
		for (int j = 0; j < temp.size(); j++)
		{
			Lia[j] = 0;
			for (int k = 0; k < all_coors.size(); k++)
			{
				if (temp[j] == all_coors[k])
				{
					Lia[j] = 1;
					break;
				}
			}

		}
		//[Lia, Locb] = ismember(temp, all_coors, 'rows');

		int cnt_zero = 0;
		for (int j = 0; j < temp.size(); j++)
		{
			if (!Lia[j])
			{
				all_coors.push_back(temp[j]);
				cnt_zero++;
			}
		}
		//all_coors = [all_coors; temp(~Lia, :)];

		cv::Mat new_line = cv::Mat::zeros(cnt_zero, all_cands.cols, CV_32SC2);
		new_line = -1;
		//new_line = zeros(nnz(~Lia), size(all_cands, 2));

		int cnt = 0;
		for (int j = 0; j < temp.size(); j++)
		{
			if (!Lia[j])
			{
				cell_cands[i].row(j).copyTo(new_line.row(cnt));
				cnt++;
			}
		}
		//new_line.at<double>(:, 1 : nCand) = cell_cands{ i }(~Lia, :);

		for (int j = 0; j < new_line.rows; j++)
			all_cands.push_back(new_line.row(j));
		//all_cands = [all_cands; new_line];

		new_line = cv::Mat::zeros(cnt_zero, all_cands.cols, CV_64FC1);
		new_line = INFINITY;
		//new_line = inf(nnz(~Lia), size(all_cands_dists, 2));

		cnt = 0;
		for (int j = 0; j < temp.size(); j++)
		{
			if (!Lia[j])
			{
				cell_cands_dists[i].row(j).copyTo(new_line.row(cnt));
				cnt++;
			}
		}
		//new_line(:, 1 : nCand) = cell_cands_dists{ i }(~Lia, :);

		for (int j = 0; j < new_line.rows; j++)
		{

			all_cands_dists.push_back(new_line.row(j));
			//all_cands_dists.push_back((double*)new_line.row(j).data);
			//all_cands_dists.push_back(new_line.col(j));
		}
		//all_cands_dists = [all_cands_dists; new_line];

		std::vector<int> Locb(temp.size());
		for (int j = 0; j < temp.size(); j++)
		{
			Locb[j] = 0;
			for (int k = 0; k < all_coors.size(); k++)
			{
				if (temp[j] == all_coors[k])
				{
					Locb[j] = k+1;
					break;
				}
			}

		}
		//[Lia, Locb] = ismember(temp, all_coors, 'rows');

		cell_coors_to_all_coors[i] = (Locb);
		//cell_coors_to_all_coors{ i } = Locb;
	}

	//??? i don't know that what is processing for
	//	temp = all_cands_dists(:, nCand + 1 : size(all_cands_dists, 2));
	//temp(temp == 0) = inf;
	//all_cands_dists(:, nCand + 1 : size(all_cands_dists, 2)) = temp;


	//for (int y = 0; y < all_cands_dists.rows; y++)
	//{
	//	for (int x = 0; x < all_cands_dists.cols; x++)
	//	{
	//		if (all_cands_dists.at<double>(y, x) >= 100000)
	//		{
	//			printf("%s  ", "inf");
	//		}
	//		else
	//			printf("%.1f  ", all_cands_dists.at<double>(y, x));
	//	}
	//	printf("\n");
	//}

	//printf("\n\n");

	int nNode = all_coors.size();
	// unary cost
	// add a dummy label for nodes having no candidate

	for (int y = 0; y < all_cands_dists.rows; y++)
	for (int x = 0; x < all_cands_dists.cols; x++)
	{
		if (all_cands_dists.at<double>(y, x) > unary_thre)
		{
			all_cands_dists.at<double>(y, x) = INFINITY;
		}
	}
	//all_cands_dists(all_cands_dists > unary_thre) = inf;

	for (int y = 0; y < all_cands_dists.rows; y++)
	for (int x = 0; x < all_cands_dists.cols; x++)
	{
		if (all_cands_dists.at<double>(y, x) <= unary_thre & all_cands_dists.at<double>(y, x) > unary_trun_thre)
		{
			all_cands_dists.at<double>(y, x) = unary_trun_thre;
		}
	}
	//all_cands_dists(all_cands_dists <= unary_thre&all_cands_dists > unary_trun_thre) = unary_trun_thre;


	
	for (int y = 0; y < all_cands.rows; y++)
	for (int x = 0; x < all_cands.cols; x++)
	{
		if (all_cands_dists.at<double>(y, x) == INFINITY)
		{
			all_cands.at<cv::Point>(y, x) = cv::Point(-1, -1);

		}
	}
	//all_cands(all_cands_dists == inf) = 0;



	
	// added for a redundancy check
	cv::Mat temp_all_cands;
	all_cands.copyTo(temp_all_cands);
	cv::Mat temp_all_cands_dists;
	all_cands_dists.copyTo(temp_all_cands_dists);
	all_cands = cv::Mat::zeros(nNode, nCand, CV_32SC2);
	all_cands = -1;
	all_cands_dists = cv::Mat::zeros(nNode, nCand, CV_64FC1);
	all_cands_dists = INFINITY;

	for (int i = 0; i < nNode; i++)
	{
		std::vector<cv::Point> C;
		std::vector<int> v_idxC;
		std::vector<int> ia;

		cv::Mat cur_row_cand = temp_all_cands.row(i);

		for (int j = 0; j < cur_row_cand.cols; j++)
		{
			int cur_x = cur_row_cand.at<cv::Point>(j).x;
			int cur_y = cur_row_cand.at<cv::Point>(j).y;
			v_idxC.push_back(cur_y * 512 + cur_x);
		}
		std::sort(v_idxC.begin(), v_idxC.end());
		v_idxC.erase(std::unique(v_idxC.begin(), v_idxC.end()) , v_idxC.end());

		for (int j = 0; j < v_idxC.size(); j++)
		{
			C.push_back(cv::Point(v_idxC[j] % 512, v_idxC[j] / 512));

			for (int k = 0; k < cur_row_cand.cols; k++)
			{
				if (C[j] == cur_row_cand.at<cv::Point>(k))
				{
					ia.push_back(k);
					break;
				}
			}

		}




		//unique(temp_all_cands.row(i), &C, &ia);
		//[C, ia, ic] = unique(temp_all_cands(i, :));

		cv::Mat unique_cands;
		cv::Mat unique_cands_dists;
		if (C[0].x != -1)
		{

			unique_cands = cv::Mat(1, ia.size(), CV_32SC2);
			unique_cands_dists = cv::Mat(1, ia.size(), CV_64FC1);
			for (int j = 0; j < ia.size(); j++)
			{

				unique_cands.at<cv::Point>(0, j) = temp_all_cands.at<cv::Point>(i, ia[j]);
				unique_cands_dists.at<double>(0, j) = temp_all_cands_dists.at<double>(i, ia[j]);
			}

		}
		else
		{
			//unique_cands = temp_all_cands(i, ia(2:end));


			unique_cands = cv::Mat(1, ia.size() - 1, CV_32SC2);
			unique_cands_dists = cv::Mat(1, ia.size() - 1, CV_64FC1);

			for (int j = 1; j < ia.size(); j++)
			{

				unique_cands.at<cv::Point>(0, j - 1) = temp_all_cands.at<cv::Point>(i, ia[j]);
				unique_cands_dists.at<double>(0, j - 1) = temp_all_cands_dists.at<double>(i, ia[j]);

			}


			//unique_cands_dists = temp_all_cands_dists(i, ia(2:end));

		}
		for (int j = 0; j < unique_cands.cols; j++)
		{
			all_cands.at<cv::Point>(i, j) = unique_cands.at<cv::Point>(0, j);

		}
		for (int j = 0; j < unique_cands_dists.cols; j++)
		{
			all_cands_dists.at<double>(i, j) = unique_cands_dists.at<double>(0, j);

		}
		//all_cands(i, 1:length(unique_cands)) = unique_cands;
		//all_cands_dists(i, 1:length(unique_cands_dists)) = unique_cands_dists;

	}
	//added for a redundancy check

	cv::Mat unaryCost;
	all_cands_dists.copyTo(unaryCost);
	cv::transpose(unaryCost, unaryCost);
	cv::Mat dummy(1, unaryCost.cols, CV_64FC1);
	dummy = 1 * dummy_unary_cost;
	unaryCost.push_back(dummy);

	//unaryCost = [all_cands_dists';dummy_unary_cost*ones(1,nNode)];

	//for (int y = 0; y < temp_all_cands_dists.rows; y++)
	//{
	//	for (int x = 0; x < temp_all_cands_dists.cols; x++)
	//	{
	//		if (temp_all_cands_dists.at<double>(y, x) >= 1000000)
	//		{
	//			printf("%s  ", "inf");
	//		}
	//		else
	//			printf("%.1f  ", temp_all_cands_dists.at<double>(y, x));
	//	}
	//	printf("\n");
	//}

	//printf("\n\n");

	// add edges
	int nEdge = 0;
	int nIntraEdge = 0;
	std::vector<cv::Mat> pairwiseCost;
	cv::Mat mapMat = cv::Mat::zeros(nNode, nNode,CV_64FC1);
	for (int i = 0; i < nSegm; i++) /// for each segment
	{
		std::vector<int> t_coors_to_all_coors = cell_coors_to_all_coors[ i ];
		for (int j = 0; j < t_coors_to_all_coors.size() - 1; j++)
		{
			cv::Mat t_pCost1;
			GetTruncatedPairwiseCost(all_coors[t_coors_to_all_coors[j]-1], all_coors[t_coors_to_all_coors[j+1]-1],
				all_cands.row(t_coors_to_all_coors[j]-1), all_cands.row(t_coors_to_all_coors[j+1]-1), dummy_pairwise_cost1, dummy_pairwise_cost2, &t_pCost1);

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
			mapMat.at<double>(t_coors_to_all_coors[j]-1, t_coors_to_all_coors[j + 1]-1) = nEdge;

			
			
		}
		
	}

	*o_unaryCost = unaryCost;
	*o_pairwiseCost = pairwiseCost;
	*o_mapMat = mapMat;
	*o_all_coors = all_coors;
	*o_all_cands = all_cands;
	*o_cell_coors_to_all_coors = cell_coors_to_all_coors;

}

void cMRF::GetTruncatedPairwiseCost(cv::Point coor1, cv::Point coor2, cv::Mat cands1, cv::Mat cands2, int dummy_pairwise_cost1, int dummy_pairwise_cost2,cv::Mat *o_mapMat)
//function cost_mat = GetTruncatedPairwiseCost(coor1, coor2, cands1, cands2, dummy_pairwise_cost1, dummy_pairwise_cost2)
{

	int nY = 512; int nX = 512;
	int nCand = cands1.cols;
	int nLabel = nCand + 1;
	std::vector<cv::Point> cands1_xxyy, cands2_xxyy;
	for (int i = 0; i < cands1.cols; i++)
	{
		cands1_xxyy.push_back(cands1.at<cv::Point>(0,i));
	}
	for (int i = 0; i < cands2.cols; i++)
	{
		cands2_xxyy.push_back(cands2.at<cv::Point>(0, i));
	}
	//[cands1_yy, cands1_xx] = ind2sub([nY, nX], cands1);
	//[cands2_yy, cands2_xx] = ind2sub([nY, nX], cands2);

	std::vector<double> diff1_y(cands1_xxyy.size()), diff1_x(cands1_xxyy.size()), diff2_y(cands1_xxyy.size()), diff2_x(cands1_xxyy.size());
	for (int i = 0; i < cands1_xxyy.size(); i++)
	{
		diff1_y[i] = cands1_xxyy[i].y - coor1.y;
		diff1_x[i] = cands1_xxyy[i].x - coor1.x;
		diff2_y[i] = cands2_xxyy[i].y - coor2.y;
		diff2_x[i] = cands2_xxyy[i].x - coor2.x;
	}
	//diff1_y = cands1_yy - coor1(1);
	//diff1_x = cands1_xx - coor1(2);
	//diff2_y = cands2_yy - coor2(1);
	//diff2_x = cands2_xx - coor2(2);

	cv::Mat repmat1(1, nCand, CV_64FC1); cv::Mat repmat2(1, nCand, CV_64FC1);
	for (int i = 0; i < repmat1.cols; i++)
	{
		repmat1.at<double>(i) = diff2_y[i];
		repmat2.at<double>(i) = diff1_y[i];
	}

	
	//for (int x = 0; x < repmat1.cols; x++)
	//{
	//	printf("%0.1f ", repmat1.at<double>(0, x));
	//}
	//	
	//
	//printf("\n\n");

	for (int i = 0; i < nCand-1; i++)
	{
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
	//diff_y = repmat(diff2_y, nCand, 1) - repmat(diff1_y',1,nCand);

	repmat1 = cv::Mat(1, nCand, CV_64FC1);
	repmat2 = cv::Mat(1, nCand, CV_64FC1);
	for (int i = 0; i < repmat1.cols; i++)
	{
		repmat1.at<double>(i) = diff2_x[i];
		repmat2.at<double>(i) = diff1_x[i];
	}

	for (int i = 0; i < nCand - 1; i++)
	{
		repmat1.push_back(repmat1.row(0));
		repmat2.push_back(repmat2.row(0));
	}

	//repmat1.copyTo(repmat2);

	cv::transpose(repmat2, repmat2);

	cv::Mat diff_x = repmat1 - repmat2;
	//diff_x = repmat(diff2_x, nCand, 1) - repmat(diff1_x',1,nCand);

	cv::Mat diff(nCand, nCand, CV_64FC1);
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
	for (int x = 0; x < diff.cols; x++)
	{
		if (diff.at<double>(y, x) > dummy_pairwise_cost2)
		{
			diff.at<double>(y, x) = dummy_pairwise_cost2;
		}
	}
	//diff(diff > dummy_pairwise_cost2) = dummy_pairwise_cost2;

	for (int i = 0; i < cands1.cols; i++)
	{
		if (cands1.at<cv::Point>(i).x == -1)
		{
			for (int j = 0; j < diff.rows; j++)
			{
				diff.at<double>(i,j) = INFINITY;
			}
		}
	}

	for (int i = 0; i < cands2.cols; i++)
	{
		if (cands2.at<cv::Point>(i).x == -1)
		{
			for (int j = 0; j < diff.rows; j++)
			{
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

	for (int y = 0; y < nCand; y++)
	for (int x = 0; x < nCand; x++)
	{
		
		cost_mat.at<double>(y, x) = diff.at<double>(y, x);
	}
	//diff(cands1 == 0, :) = inf; % needless
	//diff(:, cands2 == 0) = inf; % needless
	//cost_mat(1:nCand, 1 : nCand) = diff;

	*o_mapMat = cost_mat;
	
}
//void cMRF::GetIntervalCost()
////function cost_mat = GetIntervalCost(pre_dist, dist_sigma, cands1, cands2, dummy_pairwise_cost1, dummy_pairwise_cost2)
//{
//
//	nY = 512; nX = 512;
//	nCand = length(cands1);
//	nLabel = nCand + 1;
//	[cands1_yy, cands1_xx] = ind2sub([nY, nX], cands1);
//	[cands2_yy, cands2_xx] = ind2sub([nY, nX], cands2);
//	diff_y = repmat(cands2_yy, nCand, 1) - repmat(cands1_yy',1,nCand);
//		diff_x = repmat(cands2_xx, nCand, 1) - repmat(cands1_xx',1,nCand);
//		diff = sqrt(diff_x. ^ 2 + diff_y. ^ 2);
//
//	cost_mat = dummy_pairwise_cost1*ones(nLabel, nLabel);
//	fun = @(x)1 - exp(-(x - pre_dist). ^ 2 * 0.5*dist_sigma^-2);
//	diff = fun(diff)*dummy_pairwise_cost2;
//
//	diff(cands1 == 0, :) = inf; % needless
//		diff(:, cands2 == 0) = inf; % needless
//		cost_mat(1:nCand, 1 : nCand) = diff;
//
//	
//}

//void cMRF::unique(cv::Mat inputMat, std::vector<cv::Point> *o_uniqueSotrtPts, std::vector<int> *o_uniqueSotrtIdx)
//{
//	cv::Mat copyMat;
//	inputMat.copyTo(copyMat);
//	std::vector<double> uniqueValue;
//	std::vector<int> uniqueIdx;
//
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

void cMRF::mrf_trw_s(double *u, int uw, int uh, double **p, double* m, int nm, int mw, int mh, /*int in_Method, int in_iter_max, int in_min_iter,*/
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

	//get pairwise potentials

	//mwIndex colNum = nm-1;
	//const mwIndex* ir = mxGetIr(mInPtr);
	//const mwIndex* jc = mxGetJc(mInPtr);
	//double*        pr = mxGetPr(mInPtr);

	//mwIndex* ir = new mwIndex[nm];
	//mwIndex* jc = new mwIndex[nm];
	//double*  pr = new double[nm];
	//for (int i = 0; i < nm; i++)
	//{
	//	ir[i] = (int)m[i * 3 + 0];
	//	jc[i] = (int)m[i * 3 + 1];
	//	pr[i] = m[i * 3 + 2];
	//}

	////check pairwise terms
	//mwSize numEdges = 0;
	//for (mwIndex c = 0; c < colNum; ++c) {
	//	mwIndex rowStart = jc[c]; 
	//	mwIndex rowEnd   = jc[c+1]; 
	//	for (mwIndex ri = rowStart; ri < rowEnd; ++ri)  {
	//		mwIndex r = ir[ri];

	//		double dw = pr[ri];
	//		if( r < c) numEdges++;
	//	}
	//}


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
	for (int i = 0; i < numNodes; ++i){
		nodes[i] = mrf->AddNode(TypeGeneral::LocalSize(numLabels), TypeGeneral::NodeData(termW + i * numLabels));
	}

	//add pairwise terms
	//for (mwIndex c = 0; c < colNum; ++c) {
	//	mwIndex rowStart = jc[c];
	//	mwIndex rowEnd = jc[c + 1];
	//	for (mwIndex ri = rowStart; ri < rowEnd; ++ri)  {
	//		mwIndex r = ir[ri];
	//		int edge_idx = pr[ri];

	//		//int *pInPtr = mxGetCell(prhs[1], edge_idx - 1);
	//		//double* pCost = (double*)mxGetData(pInPtr);

	//		double* pCost = p[edge_idx - 1];

	//		//add matrix that is specified by user
	//		for (int i = 0; i < numLabels; ++i)
	//		for (int j = 0; j < numLabels; ++j)
	//			P[j + numLabels * i] = pCost[j + numLabels * i];

	//		mrf->AddEdge(nodes[r], nodes[c], TypeGeneral::EdgeData(TypeGeneral::GENERAL, P));
	//	}
	//}

	for (mwIndex c = 0; c < nm; ++c) {
		int y = m[c * 3 + 1];
		int x = m[c * 3 + 0];
		int edge_idx = m[c*3+2];

		double* pCost = p[edge_idx -1];
		
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
	//if (eOutPtr != NULL)	{
	//	*eOutPtr = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
	//	*(double*)mxGetData(*eOutPtr) = (double)energy;
	//}

	//*(double*)mxGetData(*eOutPtr) = (double)energy;
	*e = (double)energy;
	//output the best solution
	//if (sOutPtr != NULL)	{
	//	//*sOutPtr = mxCreateNumericMatrix(numNodes, 1, mxDOUBLE_CLASS, mxREAL);
	//	//double* segment = (double*)mxGetData(*sOutPtr);
	//	//for(int i = 0; i < numNodes; ++i)
	//	//	segment[i] = (double)(mrf -> GetSolution(nodes[i])) + 1;
	//}
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