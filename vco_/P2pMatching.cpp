#include "P2pMatching.h"

cP2pMatching::cP2pMatching(int size)
{

	patchSize = size;
	
	
	halfPatchSize = patchSize / 2;
}


cP2pMatching::~cP2pMatching()
{
}

void cP2pMatching::run(cv::Mat img_t, cv::Mat img_tp1, cv::Mat gt_bimg_t, 
	cVCOParams p, 
	int t_x, int t_y, 
	cv::Mat ivessel_tp1, 
	int fidx_tp1, char* savePath,
	bool bVerbose,
	std::vector<std::vector<cv::Point>> *o_E, 
	std::vector<cv::Mat>* o_cell_cands, 
	std::vector<cv::Mat>* o_cell_cands_dists, 
	std::vector<cv::Point> *o_J,
	std::vector<cv::Point> *o_end)
{
	int nY = img_tp1.rows;
	int nX = img_tp1.cols;

	std::vector<cv::Point> J,end;
	std::vector<std::vector<cv::Point>> E;
	cv::Mat bJ, mapMat;
	
	MakeGraphFromImage(gt_bimg_t, J, end, bJ, E, mapMat);
	float *d_tp1 = 0;
	int numKeyPts;

	cv::Mat crop;
	numKeyPts = (nY - 2 * halfPatchSize)*(nX - 2 * halfPatchSize)/4;

	cv::Mat d_tp1_img;
	d_tp1_img = cv::Mat(patchSize*patchSize,numKeyPts, CV_8UC1);
	
	//f_tp1 = new float[numKeyPts * 3];
	//d_tp1 = new float[numKeyPts*patchSize*patchSize];
	
	//std::vector<cv::Point> v_tp1(numKeyPts);

	cv::Mat idx_img_tp1(nY, nX, CV_32SC1);
	idx_img_tp1 = -1;

	int cnt = 0;
	for (int y = halfPatchSize; y < nY - halfPatchSize; y+=2)
	for (int x = halfPatchSize; x < nX - halfPatchSize; x+=2)
	{

		cv::Rect rc(x - halfPatchSize, y - halfPatchSize, patchSize, patchSize);
		crop = img_tp1(rc);

		/*f_tp1[(y - halfPatchSize)*(nX - 2 * halfPatchSize) + (x - halfPatchSize) * 3 + 0] = x;
		f_tp1[(y - halfPatchSize)*(nX - 2 * halfPatchSize) + (x - halfPatchSize) * 3 + 1] = y;
		f_tp1[(y - halfPatchSize)*(nX - 2 * halfPatchSize) + (x - halfPatchSize) * 3 + 2] = 0;*/

		//f_tp1[cnt * 3 + 0] = (x);
		//f_tp1[cnt * 3 + 1] = (y);
		//f_tp1[cnt * 3 + 2] = (0);

		idx_img_tp1.at<int>(y, x) = cnt;


		//for (int i = 0; i < patchSize*patchSize; i++)
		//{
		//	//d_tp1[(y - halfPatchSize)*(nX - 2 * halfPatchSize) + x - halfPatchSize + i] = crop.at<uchar>(i);
		//	//tmp_d_tp1.push_back(crop.at<uchar>(i));
		//	d_tp1[patchSize*patchSize*cnt + i] = crop.at<uchar>(i);
		//}

		for (int i = 0; i < patchSize*patchSize; i++)
		{
			d_tp1_img.at<uchar>(i, cnt) = crop.at<uchar>(i);
		}

		cnt++;
	}
	d_tp1_img.convertTo(d_tp1_img, CV_32FC1);
	d_tp1 = ((float*)d_tp1_img.data);



	

	// junction matching
	int nJ = J.size();
	cv::Mat j_cands = cv::Mat::zeros(nJ, p.n_junction_cands, CV_16UC1);
	cv::Mat j_cands_dists = cv::Mat::zeros(nJ, p.n_junction_cands, CV_32FC1);
	cv::Mat num_j_flows = cv::Mat::zeros(nJ, 1, CV_16UC1);
	std::vector<cv::Point> j_flows(nJ);


	
	//#pragma omp parallel for private(j) 
	for (int j = 0; j < nJ; j++) ////// for each junction
	{
		cv::Point cpt_xxyy(J[j]);
		cv::Point cpt = cpt_xxyy;
		cv::Point old_cpt_xxyy(cpt_xxyy - cv::Point(t_x, t_y));
		cv::Point old_cpt = old_cpt_xxyy;



		float fPts[4] = { old_cpt_xxyy.x, old_cpt_xxyy.y, p.psift_scale, 0 };
		float *d_t = 0;


		cv::Mat d_t_img;

		cv::Rect rc(fPts[0] - halfPatchSize, fPts[1] - halfPatchSize, patchSize, patchSize);
		cv::Mat crop = img_t(rc);

		d_t_img = cv::Mat(patchSize*patchSize, 1, CV_32FC1);
		for (int y = 0; y < patchSize; y++)
		for (int x = 0; x < patchSize; x++)
		{
			d_t_img.at<float>(y*patchSize + x) = crop.at<uchar>(y, x);
		}
		



		cv::Mat cand_dist = cv::Mat::zeros(nY, nX, CV_8UC1);
		for (int y1 = std::max(0, cpt_xxyy.y - p.thre_dist_step1); y1 < std::min(nY, cpt_xxyy.y + p.thre_dist_step1); y1++)
		for (int x1 = std::max(0, cpt_xxyy.x - p.thre_dist_step1); x1 < std::min(nX, cpt_xxyy.x + p.thre_dist_step1); x1++)
			cand_dist.at<uchar>(y1, x1) = 255;

		cv::Mat cand_ivessel = ivessel_tp1 >= p.thre_ivessel;
		cv::Mat cand = cand_ivessel&cand_dist;
		cv::Mat cand_idx;
		cv::findNonZero(cand, cand_idx);

		cv::Mat dist_img(nY, nX, CV_32FC1);
		dist_img = INFINITY;

		for (int k = 0; k < cand_idx.rows; k++)
		{

			if (idx_img_tp1.at<int>(cand_idx.at<cv::Point>(k)) != -1)
			{
				//cv::norm()

				//float t = cv::norm(d_t_img.col(0) - d_tp1_img.col(idx_img_tp1.at<unsigned short>(cand_idx.at<cv::Point>(k))));

				dist_img.at<float>(cand_idx.at<cv::Point>(k, 0)) =
					cv::norm(d_t_img.col(0) - d_tp1_img.col(idx_img_tp1.at<int>(cand_idx.at<cv::Point>(k))), cv::NormTypes::NORM_L2);

			}
		}

	
		cv::Mat idx_img(nY,nX,CV_32SC2);
		
		cv::Mat tmp_dist_img;
		dist_img.copyTo(tmp_dist_img);
		cand_idx = cv::Mat(p.n_junction_cands, 1, CV_32SC2);
		std::vector<cv::Point> qq;
		for (int k = 0; k < p.n_junction_cands; k++)
		{
			int minIdx[2];
			double minV;
			cv::minMaxIdx(tmp_dist_img, &minV, 0, minIdx);

			tmp_dist_img.at<float>(cv::Point(minIdx[1], minIdx[0])) = INFINITY;
			cand_idx.at<cv::Point>(k) = cv::Point(minIdx[1],minIdx[0]);
			//qq.push_back(cv::Point(minIdx[1], minIdx[0]));
		}

		j_cands.row(j) = cand_idx;
		//j_cands_dists.row(j) = dist_img(cand_idx);
		for (int k = 0; k < cand_idx.rows; k++)
			j_cands_dists.at<float>(j, k) = dist_img.at<float>(cand_idx.at<cv::Point>(k));

		////// get displacement vectors for junctions using connected component
		////// analysis & non - maximum supreesion
		cv::Mat tempBW = cv::Mat::zeros(nY, nX,CV_8UC1);
		for (int k = 0; k < cand_idx.rows; k++)
			tempBW.at<uchar>(cand_idx.at<cv::Point>(k)) = 255;
		//CC = bwconncomp(tempBW);
		cv::Mat CC;
		int nCC = cv::connectedComponents(tempBW,CC);
		nCC -= 1;
		cv::Mat CCsize = cv::Mat::zeros(nCC, 1,CV_16UC1);
		std::vector<cv::Point> Flows;
		for (int k = 0; k < nCC; k++)
		{

			//tCC = CC.PixelIdxList{ k };

			cv::Mat findIdx_img = (CC == k+1);
			cv::Mat tCC;
			cv::findNonZero(findIdx_img, tCC);

			CCsize.at<unsigned short>(k) = tCC.rows;

			//[~, idx] = min(dist_img(tCC));
			int minIdx=0;
			float minV = INFINITY;
			for (int l = 0; l < tCC.rows; l++)
			{
				if (minV > dist_img.at<float>(tCC.at<cv::Point>(l)))
				{
					minV = dist_img.at<float>(tCC.at<cv::Point>(l));
					minIdx = l;
				}
			}
			int idx = minIdx;
			//[endY, endX] = ind2sub([nY, nX], tCC(idx));
			cv::Mat endXY_img(1, 1, CV_32SC2 );
			endXY_img.at<cv::Point>(0) = tCC.at<cv::Point>(idx);
			cv::Point tFlow;
			tFlow = ((endXY_img.at<cv::Point>(0) - (cv::Point(cpt_xxyy.x, cpt_xxyy.y))));

			Flows.push_back(tFlow);
		}
		//[~, idx] = sort(CCsize, 'descend');
		cv::Mat idx;
		cv::sortIdx(CCsize, idx, cv::SORT_ASCENDING);
		int n_sel = cv::min(nCC, p.n_junction_cc_cands);
		//sel_idx = idx(1:n_sel);
		cv::Mat sel_idx(n_sel,1,CV_32FC1);
		for (int k = 0; k < n_sel;k++)
			sel_idx.at<float>(k) = idx.at<int>(k);

		num_j_flows.at<unsigned short>(j) = n_sel;
		for (int k = 0; k < n_sel; k++)
			j_flows.push_back( Flows[sel_idx.at<float>(k)]);

		
		if (bVerbose)
		{


			/// draw image
			cv::Mat canvas;
			cv::cvtColor(img_tp1, canvas, CV_GRAY2BGR);

			for (int y = 0; y < 512; y++)
			for (int x = 0; x < 512; x++)
			{
				if (gt_bimg_t.at<uchar>(y, x))
				{
					canvas.at<uchar>(y, (x)* 3 + 0) = 0;
					canvas.at<uchar>(y, (x)* 3 + 1) = 0;
					canvas.at<uchar>(y, (x)* 3 + 2) = 255;
				}
			}

			for (int pp = -1; pp <= 1; pp++)
			for (int qq = -1; qq <= 1; qq++)
			{
				canvas.at<uchar>(cpt_xxyy.y + pp, (cpt_xxyy.x + qq) * 3 + 2) = 0;
				canvas.at<uchar>(cpt_xxyy.y + pp, (cpt_xxyy.x + qq) * 3 + 1) = 0;
				canvas.at<uchar>(cpt_xxyy.y + pp, (cpt_xxyy.x + qq) * 3 + 0) = 255;
				//canvas.at<uchar>(old_cpt_xxyy.y + pp, (old_cpt_xxyy.x + qq) * 3 + 2) = 0;
				//canvas.at<uchar>(old_cpt_xxyy.y + pp, (old_cpt_xxyy.x + qq) * 3 + 1) = 0;
				//canvas.at<uchar>(old_cpt_xxyy.y + pp, (old_cpt_xxyy.x + qq) * 3 + 0) = 255;

			}

			for (int k = 0; k < cand_idx.rows; k++)
			{
				canvas.at<uchar>(cand_idx.at<cv::Point>(k).y, cand_idx.at<cv::Point>(k).x * 3 + 2) = 0;
				canvas.at<uchar>(cand_idx.at<cv::Point>(k).y, cand_idx.at<cv::Point>(k).x * 3 + 1) = 255;
				canvas.at<uchar>(cand_idx.at<cv::Point>(k).y, cand_idx.at<cv::Point>(k).x * 3 + 0) = 0;
			}


			char str[200];
			sprintf(str, "%s%d-th_frame_%d-th_feature.png", savePath, fidx_tp1, j);
			cv::imwrite(str, canvas);

		}
		
	}
	

	

	// local point matching
	int nE = E.size();
	std::vector<cv::Mat> v_segm_pt_cands(nE);
	std::vector<cv::Mat> v_segm_pt_cands_d(nE);

	//printf("\n");
	//printf("\n");
	//for (int y = 0; y < mapMat.rows; y++)
	//{

	//	for (int x = 0; x < mapMat.cols; x++)
	//	{
	//		printf("%d\t", mapMat.at<uchar>(y, x));
	//	}
	//	printf("\n");
	//}
	//printf("\n\n");

	cv::Mat cand_ivessel = ivessel_tp1 >= p.thre_ivessel;

	
	//int j;
	//#pragma omp parallel for private(j) 


	for (int j = 0; j < nE; j++) ////// for each segment
	{
		

		printf("each segment %d\n", j);
		if (j == 10)
			int b = 0;
			

		int iTrial = 1;
		////// original
		cv::Mat arr_cands, arr_cands_dists;

		//d_tp1_img(descSize, numKeyPts, CV_32FC1, d_tp1);

		GetCandidates(img_t, img_tp1, E[j], cv::Point(t_x, t_y), d_tp1, numKeyPts, idx_img_tp1, ivessel_tp1, p, cand_ivessel,
			arr_cands, arr_cands_dists, d_tp1_img);

		int npt = arr_cands.rows;
		cv::Mat all_arr_cands = cv::Mat::zeros(npt, p.n_all_cands, CV_32SC2);
		all_arr_cands = 0;
		cv::Mat all_arr_cands_dists(npt, p.n_all_cands,CV_32FC1);
		all_arr_cands_dists = INFINITY;
		
		cv::Rect rc(p.n_cands*(iTrial - 1), 0, p.n_cands*iTrial - p.n_cands*(iTrial - 1), npt);
		cv::Mat roi = all_arr_cands(rc);
		//roi = arr_cands;
		arr_cands.copyTo(roi);
		//all_arr_cands.at<cv::Point>(k, p.n_cands*(iTrial - 1) + 1 : p.n_cands*iTrial) = arr_cands;
		roi = all_arr_cands_dists(rc);
		//roi = arr_cands_dists;
		arr_cands_dists.copyTo(roi);
		//all_arr_cands_dists(:, p.n_cands*(iTrial - 1) + 1 : p.n_cands*iTrial) = arr_cands_dists;
		iTrial++;
		////// translated
		//[stV, edV] = find(mapMat == j);
		cv::Mat stedV;
		cv::findNonZero(mapMat == j+1, stedV);
		if (stedV.rows == 0)
		{
			cv::Point stpt = E[j].front();
			cv::Point edpt = E[j].back();

			int cnt = E.size()-1;
			//stedV = cv::Mat(1, 1, CV_32FC2);
			while (true)
			{
				if (stpt.x == E[cnt].front().x && stpt.y == E[cnt].front().y &&
					edpt.x == E[cnt].back().x && edpt.y == E[cnt].back().y)
				{
					cv::findNonZero(mapMat == cnt + 1, stedV);
					break;
				}
				cnt--;
			}
				
		}
		
		int is_stV_J = bJ.at<uchar>(stedV.at<cv::Point>(0).y);
		int is_edV_J = bJ.at<uchar>(stedV.at<cv::Point>(0).x);

		if (is_stV_J)
		{
			int nFlows = num_j_flows.at<unsigned short>(stedV.at<cv::Point>(0).y);
			for (int k = 0; k < nFlows; k++)
			{
				std::vector<cv::Point> tFlow;
				
				tFlow.push_back(j_flows[stedV.at<cv::Point>(0).y]);
				std::vector<cv::Point> translated_E;
				for (int l = 0; l < E[j].size(); l++)
					translated_E.push_back(E[ j ][l]+tFlow[0]);
				
				bool bState = true;;
				for (int l = 0; l < translated_E.size(); l++)
				{
					if (translated_E[l].y<0 | translated_E[l].x<0 | translated_E[l].y>nY - 1 | translated_E[l].x>nX - 1)
					{
						bState = false;
						break;
					}
						
				}
				if (!bState)
					continue;
				//if (nnz(translated_E(:, 1)<1 | translated_E(:, 2)<1 | translated_E(:, 1)>nY | translated_E(:, 2)>nX))
				//	continue;

				GetCandidates(img_t, img_tp1, translated_E, cv::Point(t_x + tFlow[0].x, t_y + tFlow[0].y), d_tp1, numKeyPts, idx_img_tp1, ivessel_tp1, p, cand_ivessel,
					arr_cands, arr_cands_dists, d_tp1_img);

				cv::Rect rc(p.n_cands*(iTrial - 1), 0, 
					//p.n_cands*iTrial - p.n_cands*(iTrial - 1), 
					p.n_cands,
					npt);
				cv::Mat roi = all_arr_cands(rc);
				//roi = arr_cands;
				arr_cands.copyTo(roi);
				roi = all_arr_cands_dists(rc);
				//roi = arr_cands_dists;
				arr_cands_dists.copyTo(roi);

				//all_arr_cands(:, p.n_cands*(iTrial - 1) + 1 : p.n_cands*iTrial) = arr_cands;
				//all_arr_cands_dists(:, p.n_cands*(iTrial - 1) + 1 : p.n_cands*iTrial) = arr_cands_dists;
				iTrial++;
			}

		}
		if (is_edV_J)
		{

			//nFlows = num_j_flows(edV);
			int nFlows = num_j_flows.at<unsigned short>(stedV.at<cv::Point>(0).x);
			for (int k = 0; k < nFlows; k++)
			{
				std::vector<cv::Point> tFlow;
				tFlow.push_back( j_flows[stedV.at<cv::Point>(0).x]);
				std::vector<cv::Point> translated_E;
				//= E{ j }+repmat(tFlow, size(E{ j }, 1), 1);
				
				for (int l = 0; l < E[j].size(); l++)
					translated_E.push_back(E[j][l] + tFlow[0]);

				//if (nnz(translated_E(:, 1)<1 | translated_E(:, 2)<1 | translated_E(:, 1)>nY | translated_E(:, 2)>nX))
				//	continue;

				bool bState = true;;
				for (int l = 0; l < translated_E.size(); l++)
				{
					if (translated_E[l].y<0 | translated_E[l].x<0 | translated_E[l].y>nY - 1 | translated_E[l].x>nX - 1)
					{
						bState = false;
						break;
					}

				}
				if (!bState)
					continue;

				GetCandidates(img_t, img_tp1, translated_E, cv::Point(t_x + tFlow[0].x, t_y + tFlow[0].y), d_tp1, numKeyPts, idx_img_tp1, ivessel_tp1, p, cand_ivessel,
					arr_cands, arr_cands_dists, d_tp1_img);

				//all_arr_cands(:, p.n_cands*(iTrial - 1) + 1 : p.n_cands*iTrial) = arr_cands;
				//all_arr_cands_dists(:, p.n_cands*(iTrial - 1) + 1 : p.n_cands*iTrial) = arr_cands_dists;

				cv::Rect rc(p.n_cands*(iTrial - 1), 0, 
					//p.n_cands*iTrial - p.n_cands*(iTrial - 1), 
					p.n_cands,
					npt);
				cv::Mat roi = all_arr_cands(rc);
				arr_cands.copyTo(roi);
				roi = all_arr_cands_dists(rc);
				//roi = arr_cands_dists;
				arr_cands_dists.copyTo(roi);

				iTrial++;
			}
			
		}
		v_segm_pt_cands[j] = ( all_arr_cands);
		v_segm_pt_cands_d[j] = (all_arr_cands_dists);

		//cv::Mat ind;
		//all_arr_cands.copyTo(ind);

		//cv::Mat tmpNonZero;
		////cv::findNonZero(ind, tmpNonZero);
		//for (int y = 0; y < ind.rows; y++)
		//for (int x = 0; x < ind.cols; x++)
		//{

		//}
		/*std::vector<cv::Point> tmp_ind;
		for (int k = 0; k < ind.rows; k++)
		{
			if (ind.at<cv::Point>(k) != cv::Point(0, 0))
				tmp_ind.push_back(ind.at<cv::Point>(k));
		}*/
		//ind(ind == 0) = [];
	}


	for (int j = 0; j < nE; j++) ////// for each segment
	{
		int len = E[ j ].size();
		//cv::Mat samp_idx = [1:p.sampling_period : len];
		std::vector<int> samp_idx(round(len / (float)p.sampling_period));
		int sumPeriod = 0;
		//samp_idx[0] = 0;
		for (int i = 0; i < samp_idx.size(); i++)
		{
			samp_idx[i] = sumPeriod;
			sumPeriod += p.sampling_period;
		}

		if (!samp_idx.size())
		{
			samp_idx.push_back(0);
		}

		if (samp_idx.back() != len-1)
		{
			samp_idx.push_back( len-1);
		}
		std::vector<cv::Point> tmpE;
		for (int k = 0; k < samp_idx.size(); k++)
			tmpE.push_back(E[j][samp_idx[k]]);

		E[j] = tmpE;
	}

	printf("end of p2p matching\n");

	// assign outputs
	*o_E = E;
	*o_cell_cands = v_segm_pt_cands;
	*o_cell_cands_dists = v_segm_pt_cands_d;
	*o_J = J;
	*o_end = end;
	//delete[] d_tp1;
	//delete[] f_tp1;
}
void cP2pMatching::MakeGraphFromImage(cv::Mat bimg, std::vector<cv::Point> &J, std::vector<cv::Point> &o_end, cv::Mat &bJ,
	std::vector<std::vector<cv::Point>> &E,cv::Mat &mapMat)
{
	// parameter
	int patch_half_size = 5;

	int nY = bimg.rows;
	int nX = bimg.cols;

	//junction & end point detection

	cv::Mat copy;
	bimg.copyTo(copy);
	cv::Mat C;
	branch(copy, C);

	cv::Mat B;
	backcount4(copy, B);
	cv::Mat matE = (B == 1);

	
	cv::Mat FC = C.mul(~matE);
	cv::Mat D;
	cv::Mat kernel(3, 3, CV_8UC1);
	cv::Mat Vp = ((B == 2) & ~matE);
	cv::Mat Vq = ((B > 2) & ~matE);

	kernel = 255;
	cv::dilate(Vq, D, kernel);
	
	cv::Mat M = D&(FC & Vp);
	cv::Mat branch_img = FC & ~M;

	bool verbe = false;
	int waitTime = 1;

	

	cv::Mat end_img;
	endp(bimg, end_img);
	std::vector<cv::Point> ends;
	cv::findNonZero(end_img, ends);

	// added
	cv::Mat dilated_branch_img;
	cv::dilate(branch_img, dilated_branch_img,kernel);
	cv::Mat CC;
	//std::vector<cv::Point> CC;
	int numCC = cv::connectedComponents(dilated_branch_img,CC,4);

	//cv::Mat CC_8u;
	//CC.convertTo(CC_8u, CV_8UC1);
	//cv::imshow("CC", CC_8u*40);
	//cv::waitKey();

	numCC -= 1;

	std::vector<std::vector<cv::Point>> PixelIdxList(numCC);
	for (int j = 1; j < numCC+1; j++)
	{
		cv::Mat cur_label_img = (CC == j);
		
		std::vector<cv::Point> cur_label;

		cv::findNonZero(cur_label_img,cur_label);

		PixelIdxList[j-1] = (cur_label);
	}
	std::vector<cv::Point> branch(numCC);
	

	cv::Mat branch_idx_img;
	CC.copyTo(branch_idx_img);

	for (int j = 0; j < numCC; j++)
	{
		std::vector<cv::Point> tCC = PixelIdxList[j];
		int sumX=0;
		int sumY=0;
		cv::Mat xx(tCC.size(), 1, CV_16UC1), yy(tCC.size(), 1, CV_16UC1);
		for (int k = 0; k < tCC.size(); k++)
		{
			sumX += tCC[k].x;
			sumY += tCC[k].y;

			xx.at<unsigned short>(k, 0) = tCC[k].x;
			yy.at<unsigned short>(k, 0) = tCC[k].y;
			
		}

		int cy = round(sumY / (float)tCC.size());
		int cx = round(sumX / (float)tCC.size());
		cv::Mat patch = ExtractPatchWithZeroPadding(bimg, cv::Point(cx, cy), patch_half_size * 2 + 1);


		cv::Mat DT;
		cv::Mat CPT;

		patch.convertTo(patch, CV_8UC1);
		/*[DT, CPT] = */cv::distanceTransform(~patch, DT, CPT, CV_DIST_L1, 3, cv::DIST_LABEL_PIXEL);

		std::vector<cv::Point> cptSeed;
		for (int y = 0; y < DT.rows; y++)
		for (int x = 0; x < DT.cols; x++)
		{
			if (DT.at<float>(y, x) == 0)
				cptSeed.push_back(cv::Point(x,y));

		}

		cv::Point xxyy = cptSeed[CPT.at<int>(patch_half_size + 1, patch_half_size + 1)-1];

		cv::Point cc = xxyy + cv::Point(cx - patch_half_size - 1, cy - patch_half_size - 1);

		branch[j] = cc;
	}

	//idx = find(ismember([end_y, end_x], [branch_y, branch_x], 'rows'));
	std::vector<cv::Point> tmpEnds;
	bool bExist = false;
	for (int i = 0; i < ends.size(); i++)
	{

		for (int j = 0; j < ends.size(); j++)
		{
			if (i == j)
				continue;

			if (ends[i] == ends[j])
				bExist = true;
		}
		if (!bExist)
			tmpEnds.push_back(ends[i]);
	}
	ends = tmpEnds;
	
	J = branch;
	o_end = ends;
	std::vector<cv::Point> V;
	for (int i = 0; i < branch.size(); i++)
		V.push_back(branch[i]);
	for (int i = 0; i < ends.size(); i++)
		V.push_back(ends[i]);


	int numV = V.size();
	bJ = cv::Mat::zeros(numV, 1,CV_8UC1); 
	cv::Rect rc(0,0,1,J.size()); bJ(rc) = true;

	cv::Mat conn = cv::Mat::zeros(numV, numV,CV_8UC1);
	mapMat = cv::Mat::zeros(numV, numV,CV_8UC1);

	
	cv::Mat simg;
	bimg.copyTo(simg);
	std::vector<cv::Point> all_CC_pts ; // junction CC
	for (int j = 0; j < numCC; j++)
	{
		cv::Mat temp = cv::Mat::zeros(nY, nX, CV_8UC1);
		for (int k = 0; k < PixelIdxList[j].size(); k++)
		{

			simg.at<uchar>(PixelIdxList[j][k]) = false;
			temp.at<uchar>(PixelIdxList[j][k]) = 255;
		}
		
		
		std::vector<cv::Point> idx;
		cv::findNonZero((bimg==255)&(temp==255), idx);
		for (int k = 0; k < idx.size(); k++)
			all_CC_pts.push_back(idx[k]);
	}

	numCC = cv::connectedComponents(simg, CC);
	numCC -= 1;
	int numE = numCC;
	E = std::vector<std::vector<cv::Point>>::vector(numE);

	//PixelIdxList.clear();
	//PixelIdxList.assign(numCC,std::vector<cv::Point>);
	PixelIdxList = std::vector<std::vector<cv::Point>>::vector(numE);
	for (int j = 1; j < numCC + 1; j++)
	{
		cv::Mat cur_label_img = (CC == j);

		std::vector<cv::Point> cur_label;

		cv::findNonZero(cur_label_img, cur_label);

		PixelIdxList[j - 1] = (cur_label);
	}

	for (int j = 0; j < numCC; j++)
	{

		if (j == 46)
		{
			int asdf = 0;
			waitTime = 0;
		}
		
		
		std::vector<cv::Point> tCC = PixelIdxList[ j ];
		cv::Mat timg;
		bimg.copyTo(timg);
		timg /= 255;
		for (int k = 0; k < all_CC_pts.size(); k++)
			timg.at<uchar>(all_CC_pts[k]) = 2;
		std::vector<cv::Point> tE;
		std::vector<int> idx;
		//std::vector<int> stV,edV;
		int stV = -1, edV = -1;
		for (int k = 0; k < tCC.size(); k++)
		{
			if (end_img.at<uchar>(tCC[k]))
				idx.push_back(k);
		}

		
		//int stV = 0; int edV = 0;
		if (idx.size() == 2) // case: end - end
		{

			cv::Point stIDX = tCC[0];
			timg.at<uchar>(stIDX) = 0;
			
			tE.push_back(stIDX);
			bool bForwardFirst = true;
			bool bBackwardFirst = true;
			cv::Point curPt;
			// forward path
			while (true)
			{
				
				if (bForwardFirst)
				{
					curPt = stIDX;
					bForwardFirst = false;
				}


				cv::Mat temp = ExtractPatchWithZeroPadding(timg, curPt, 3);
				std::vector<cv::Point> incXY;
				cv::findNonZero(temp, incXY);

				std::vector<cv::Point> nextXY;
				for (int k = 0; k < incXY.size(); k++)
				{
					cv::Point tmp = cv::Point(curPt.x + incXY[k].x - 1, curPt.y + incXY[k].y - 1);
					nextXY.push_back(tmp);
				}
				
				if (!nextXY.size())
					break;

			
				curPt = cv::Point(nextXY[0]);
				timg.at<uchar>(curPt) = 0;

				tE.push_back(curPt);
			}
			// backward path
			while (true)
			{
				if (bBackwardFirst)
				{
					curPt = stIDX;
					bBackwardFirst = false;
				}
				cv::Mat temp = ExtractPatchWithZeroPadding(timg, curPt, 3);
				std::vector<cv::Point> incXY;
				cv::findNonZero(temp, incXY);

				std::vector<cv::Point> nextXY;
				for (int k = 0; k < incXY.size(); k++)
				{
					cv::Point tmp = cv::Point(curPt.x + incXY[k].x - 1, curPt.y + incXY[k].y - 1);
					nextXY.push_back(tmp);
				}
				
				if (!nextXY.size())
					break;
				curPt = cv::Point(nextXY[0]);
				timg.at<uchar>(curPt) = 0;

				std::vector<cv::Point>::iterator it;

				it = tE.begin();

				tE.insert(it, curPt);
			}

			
			for (int k = 0; k < V.size(); k++)
			{
				if (V[k] == tE[0])
				{
					stV = k;
				}
				if (V[k] == tE[tE.size()-1])
				{
					edV = k;
				}
			}

		}
		else if (idx.size() == 1) // case: junction - end
		{

			cv::Point curXY = tCC[idx[0]];
			timg.at<uchar>(curXY) = 0;

			for (int k = 0; k < V.size(); k++)
			{
				if (V[k] == curXY)
				{
					edV=k;
				}
			}
			tE.push_back(curXY);

			
			cv::Mat check_img2(512, 512, CV_8UC3);
			
			if (verbe)
			{

				for (int y = 0; y < 512; y++)
				{

					for (int x = 0; x < 512; x++)
					{

						if (timg.at<uchar>(y, x) == 1)
						{
							check_img2.at<uchar>(y, x * 3 + 0) = 255;
							check_img2.at<uchar>(y, x * 3 + 1) = 255;
							check_img2.at<uchar>(y, x * 3 + 2) = 255;
						}
						else if (timg.at<uchar>(y, x) == 2)
						{
							check_img2.at<uchar>(y, x * 3 + 0) = 0;
							check_img2.at<uchar>(y, x * 3 + 1) = 0;
							check_img2.at<uchar>(y, x * 3 + 2) = 255;
						}
						else
						{
							check_img2.at<uchar>(y, x * 3 + 0) = 0;
							check_img2.at<uchar>(y, x * 3 + 1) = 0;
							check_img2.at<uchar>(y, x * 3 + 2) = 0;
						}


					}

				}
			}

			while (true)
			{
				cv::Mat temp = ExtractPatchWithZeroPadding(timg, curXY, 3);

				std::vector<cv::Point> incXY;
				cv::findNonZero(temp, incXY);



				if (verbe)
				{
					check_img2.at<uchar>(curXY.y, curXY.x * 3 + 0) = 0;
					check_img2.at<uchar>(curXY.y, curXY.x * 3 + 1) = 255;
					check_img2.at<uchar>(curXY.y, curXY.x * 3 + 2) = 0;

					cv::Mat view;
					check_img2.copyTo(view);
					cv::resize(view, view, cv::Size(800, 800));
					cv::imshow("view", view);
					cv::waitKey(waitTime);
				}

				std::vector<cv::Point> nextXY;
				for (int k = 0; k < incXY.size(); k++)
				{
					cv::Point tmp = cv::Point(curXY.x + incXY[k].x - 1, curXY.y + incXY[k].y - 1);
					nextXY.push_back(tmp);
				}
				std::vector<int> ii;
				for (int k = 0; k < nextXY.size(); k++)
				{
					if (timg.at<uchar>(nextXY[k]) == 2)
						ii.push_back(k);
				}
				if (ii.size())
				{
					if (ii.size() > 1)
					{
						cv::Mat nextXY_img(ii.size(),2,CV_16UC1);
						cv::Mat curXY_img(ii.size(), 2, CV_16UC1);
						std::vector<int> dists;
						int minDist = INT_MAX;
						int min_idx = 0;
						for (int k = 0; k < ii.size(); k++)
						{
							nextXY_img.at<unsigned short>(k, 0) = nextXY[ii[k]].x;
							nextXY_img.at<unsigned short>(k, 1) = nextXY[ii[k]].y;

							curXY_img.at<unsigned short>(k, 0) = curXY.x;
							curXY_img.at<unsigned short>(k, 1) = curXY.y;

							int cur_dist = std::abs(nextXY[ii[k]].x - curXY.x) + std::abs(nextXY[ii[k]].y - curXY.y);
							if (minDist > cur_dist)
							{
								minDist = cur_dist;
								min_idx = k;
							}
						}
						//dists = sum(abs(nextXY_img - curXY_img), 2);
						//[~, min_idx] = min(dists);
						int tmp_ii = ii[min_idx];
						
						ii.clear();
						ii.push_back(tmp_ii);

					}
					//cv::Point curXY;
					curXY = cv::Point(nextXY[ii[0]]);
					
					int branch_idx = branch_idx_img.at<int>(curXY)-1;
					std::vector<cv::Point> t_path;
					t_path = bresenham(curXY, branch[branch_idx]);
					//stV.clear();
					//stV.push_back(branch_idx);
					stV = branch_idx;
					for (int k = 0; k < t_path.size(); k++)
					{
						timg.at<uchar>(t_path[k]) = 0;


						std::vector<cv::Point>::iterator it;
						it = tE.begin();

						tE.insert(it, t_path[k]);
					}
					
					
					//tE = [flipud([t_path_y, t_path_x]); tE];
					break;
				}
				//curY = nextY; curX = nextX;
				curXY = nextXY[0];
				
				//for (int k = 0; k < curXY_vec.size(); k++)
				timg.at<uchar>(curXY) = 0;

				std::vector<cv::Point>::iterator it;
				it = tE.begin();

				tE.insert(it, curXY);

				
			}
		}
		else // case: junction - junction
		{
			cv::Point stIDX = tCC[0];
			timg.at<uchar>(stIDX) = 0;
			
			tE.push_back(stIDX);

			bool bForwardFirst = true;
			bool bBackwardFirst = true;
			cv::Point curXY;

			cv::Mat check_img2(512, 512, CV_8UC3);

			if (verbe)
			{

				for (int y = 0; y < 512; y++)
				{

					for (int x = 0; x < 512; x++)
					{

						if (timg.at<uchar>(y, x) == 1)
						{
							check_img2.at<uchar>(y, x * 3 + 0) = 255;
							check_img2.at<uchar>(y, x * 3 + 1) = 255;
							check_img2.at<uchar>(y, x * 3 + 2) = 255;
						}
						else if (timg.at<uchar>(y, x) == 2)
						{
							check_img2.at<uchar>(y, x * 3 + 0) = 0;
							check_img2.at<uchar>(y, x * 3 + 1) = 0;
							check_img2.at<uchar>(y, x * 3 + 2) = 255;
						}
						else
						{
							check_img2.at<uchar>(y, x * 3 + 0) = 0;
							check_img2.at<uchar>(y, x * 3 + 1) = 0;
							check_img2.at<uchar>(y, x * 3 + 2) = 0;
						}


					}

				}
			}
			
			// forward path
			while (true)
			{
				
				if (bForwardFirst)
				{
					//curY = stY; curX = stX;
					curXY = stIDX;
					bForwardFirst = false;
				}
				if (verbe)
				{
					check_img2.at<uchar>(curXY.y, curXY.x * 3 + 0) = 0;
					check_img2.at<uchar>(curXY.y, curXY.x * 3 + 1) = 255;
					check_img2.at<uchar>(curXY.y, curXY.x * 3 + 2) = 0;

					cv::Mat view;
					check_img2.copyTo(view);
					cv::resize(view, view, cv::Size(800, 800));
					cv::imshow("view", view);
					cv::waitKey(waitTime);

				}

				cv::Mat temp = ExtractPatchWithZeroPadding(timg, curXY, 3);
				std::vector<cv::Point> incXY;
				std::vector<cv::Point> nextXY;
				cv::findNonZero(temp, incXY);
	
				
				
				for (int k = 0; k < incXY.size(); k++)
				{
					cv::Point tmp = cv::Point(curXY.x + incXY[k].x - 1, curXY.y + incXY[k].y - 1);
					nextXY.push_back(tmp);
				}
				//nextY = curY + incY - (size(temp, 1) - 1); nextX = curX + incX - (size(temp, 2) - 1);
				std::vector<int> ii;
				for (int k = 0; k < nextXY.size(); k++)
				{
					if (timg.at<uchar>(nextXY[k]) == 2)
						ii.push_back(k);
				}
				if (ii.size())
				{
					if (ii.size() > 1)
					{
						cv::Mat nextXY_img(ii.size(), 2, CV_16UC1);
						cv::Mat curXY_img(ii.size(), 2, CV_16UC1);
						std::vector<int> dists;
						int minDist = INT_MAX;
						int min_idx = 0;
						for (int k = 0; k < ii.size(); k++)
						{
							nextXY_img.at<unsigned short>(k, 0) = nextXY[ii[k]].x;
							nextXY_img.at<unsigned short>(k, 1) = nextXY[ii[k]].y;

							curXY_img.at<unsigned short>(k, 0) = curXY.x;
							curXY_img.at<unsigned short>(k, 1) = curXY.y;

							int cur_dist = std::abs(nextXY[ii[k]].x - curXY.x) + std::abs(nextXY[ii[k]].y - curXY.y);
							if (minDist > cur_dist)
							{
								minDist = cur_dist;
								min_idx = k;
							}
						}
						//dists = sum(abs(nextXY_img - curXY_img), 2);
						//[~, min_idx] = min(dists);
						int tmp_ii = ii[min_idx];

						ii.clear();
						ii.push_back(tmp_ii);
					}
					//curY = nextY(ii); curX = nextX(ii);
					curXY = nextXY[ii[0]];
					int branch_idx = branch_idx_img.at<int>(curXY)-1;
					std::vector<cv::Point> t_path = bresenham(curXY, branch[branch_idx]);
					

					//tE = [tE; [t_path_y, t_path_x]];	
					for (int k = 0; k < t_path.size(); k++)
					{
						timg.at<uchar>(t_path[k]) = 0;

						tE.push_back(t_path[k]);

					}

					

					break;
				}
				//curY = nextY(1); curX = nextX(1);
				curXY = nextXY[0];

				timg.at<uchar>(curXY) = 0;

				tE.push_back(curXY);
				
			}
			// backward path
			while (true)
			{
				if (bBackwardFirst)
				{
					//curY = stY; curX = stX;
					curXY = stIDX;
					bBackwardFirst = false;
				}
				if (verbe)
				{
					check_img2.at<uchar>(curXY.y, curXY.x * 3 + 0) = 0;
					check_img2.at<uchar>(curXY.y, curXY.x * 3 + 1) = 255;
					check_img2.at<uchar>(curXY.y, curXY.x * 3 + 2) = 0;

					cv::Mat view;
					check_img2.copyTo(view);
					cv::resize(view, view, cv::Size(800, 800));
					cv::imshow("view", view);
					cv::waitKey(waitTime);
				}

				cv::Mat temp = ExtractPatchWithZeroPadding(timg, curXY, 3);
				std::vector<cv::Point> incXY;
				std::vector<cv::Point> nextXY;
				cv::findNonZero(temp, incXY);



				for (int k = 0; k < incXY.size(); k++)
				{
					cv::Point tmp = cv::Point(curXY.x + incXY[k].x - 1, curXY.y + incXY[k].y - 1);
					nextXY.push_back(tmp);
				}
				std::vector<int> ii;
				for (int k = 0; k < nextXY.size(); k++)
				{
					if (timg.at<uchar>(nextXY[k]) == 2)
						ii.push_back(k);
				}
				if (ii.size())
				{
					if (ii.size() > 1)
					{
						cv::Mat nextXY_img(ii.size(), 2, CV_16UC1);
						cv::Mat curXY_img(ii.size(), 2, CV_16UC1);
						std::vector<int> dists;
						int minDist = INT_MAX;
						int min_idx = 0;
						for (int k = 0; k < ii.size(); k++)
						{
							nextXY_img.at<unsigned short>(k, 0) = nextXY[ii[k]].x;
							nextXY_img.at<unsigned short>(k, 1) = nextXY[ii[k]].y;

							curXY_img.at<unsigned short>(k, 0) = curXY.x;
							curXY_img.at<unsigned short>(k, 1) = curXY.y;

							int cur_dist = std::abs(nextXY[ii[k]].x - curXY.x) + std::abs(nextXY[ii[k]].y - curXY.y);
							if (minDist > cur_dist)
							{
								minDist = cur_dist;
								min_idx = k;
							}
						}
						int tmp_ii = ii[min_idx];

						ii.clear();
						ii.push_back(tmp_ii);
					}
					/*curY = nextY(ii); curX = nextX(ii);*/
					curXY = nextXY[ii[0]];
					int branch_idx = branch_idx_img.at<int>(curXY)-1;
					std::vector<cv::Point> t_path;
					t_path = bresenham(curXY, branch[branch_idx]);
					for (int k = 0; k < t_path.size(); k++)
					{
						timg.at<uchar>(t_path[k]) = 0;

						std::vector<cv::Point>::iterator it;
						it = tE.begin();


						tE.insert(it,t_path[k]);
					}
					//tE = [flipud([t_path_y, t_path_x]); tE];
					break;
				}
				//curY = nextY; curX = nextX;
				curXY = nextXY[0];
				timg.at<uchar>(curXY) = 0;

				std::vector<cv::Point>::iterator it;

				it = tE.begin();

				tE.insert(it,curXY);



			}
			//stV = find(ismember(V, tE(1, :), 'rows'));
			//edV = find(ismember(V, tE(end, :), 'rows'));
			for (int k = 0; k < V.size(); k++)
			{
				if (V[k] == tE[0])
				{
					stV=k;
				}
				if (V[k] == tE[tE.size()-1])
				{
					edV=k;
				}
			}
		}
		E [j]  = tE;


			conn.at<uchar>(stV, edV) = true; conn.at<uchar>(edV, stV) = true;
			mapMat.at<uchar>(stV, edV) = j+1;
		
			if (j == 44)
				int a = 0;
			tE.clear();
		
	}

	PixelIdxList.clear();
	//cv::Mat branch_img = bwmorph(bimg, 'branchpoints');
	//[branch_y, branch_x] = find(branch_img);
	//end_img = bwmorph(bimg, 'endpoints');
	//[end_y, end_x] = find(end_img);

	////added
	//dilated_branch_img = bwmorph(branch_img, 'dilate');
	//CC = bwconncomp(dilated_branch_img, 4);
	//numCC = CC.NumObjects;
	//branch_y = zeros(numCC, 1);
	//branch_x = zeros(numCC, 1);
	//branch_idx_img = zeros(nY, nX);
}


//-----------------------------------------------------------------------------------------------------
// LUT for skeletonization
//-----------------------------------------------------------------------------------------------------
void cP2pMatching::GetLutSkel(cv::Mat& Lut)
{
	Lut = cv::Mat(8, 512, CV_16UC1);
	static int lut1[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	static int lut2[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	static int lut3[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	static int lut4[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	static int lut5[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1 };
	static int lut6[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	static int lut7[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	static int lut8[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	for (int i = 0; i<512; i++)
	{
		Lut.at<unsigned short>(0, i) = lut1[i];
		Lut.at<unsigned short>(1, i) = lut2[i];
		Lut.at<unsigned short>(2, i) = lut3[i];
		Lut.at<unsigned short>(3, i) = lut4[i];
		Lut.at<unsigned short>(4, i) = lut5[i];
		Lut.at<unsigned short>(5, i) = lut6[i];
		Lut.at<unsigned short>(6, i) = lut7[i];
		Lut.at<unsigned short>(7, i) = lut8[i];
	}
}

//-----------------------------------------------------------------------------------------------------
// http://matlab.exponenta.ru/imageprocess/book3/13/applylut.php
//-----------------------------------------------------------------------------------------------------
void cP2pMatching::applylut_1(cv::Mat &src, cv::Mat &dst)
{
	static int lut_endpoints[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0 };

	cv::Mat k(3, 3, CV_16UC1);

	k.at<unsigned short>(0, 0) = 256;
	k.at<unsigned short>(1, 0) = 128;
	k.at<unsigned short>(2, 0) = 64;
	k.at<unsigned short>(0, 1) = 32;
	k.at<unsigned short>(1, 1) = 16;
	k.at<unsigned short>(2, 1) = 8;
	k.at<unsigned short>(0, 2) = 4;
	k.at<unsigned short>(1, 2) = 2;
	k.at<unsigned short>(2, 2) = 1;

	dst = src.clone();

	filter2D(dst, dst, CV_16UC1, k);
	for (int i = 0; i<dst.rows; i++)
	{
		for (int j = 0; j<dst.cols; j++)
		{
			dst.at<unsigned short>(i, j) = lut_endpoints[dst.at<unsigned short>(i, j)];
		}
	}

	dst.convertTo(dst, CV_8UC1);
}

//-----------------------------------------------------------------------------------------------------
// http://matlab.exponenta.ru/imageprocess/book3/13/applylut.php
//-----------------------------------------------------------------------------------------------------
void cP2pMatching::applylut_8(cv::Mat &src, cv::Mat &dst, cv::Mat& lut)
{
	cv::Mat k(3, 3, CV_16UC1);

	k.at<unsigned short>(0, 0) = 256;
	k.at<unsigned short>(1, 0) = 128;
	k.at<unsigned short>(2, 0) = 64;
	k.at<unsigned short>(0, 1) = 32;
	k.at<unsigned short>(1, 1) = 16;
	k.at<unsigned short>(2, 1) = 8;
	k.at<unsigned short>(0, 2) = 4;
	k.at<unsigned short>(1, 2) = 2;
	k.at<unsigned short>(2, 2) = 1;

	dst = src.clone();

	for (int I = 7; I >= 0; I--)
	{
		filter2D(dst, dst, CV_16UC1, k);
		for (int i = 0; i<dst.rows; i++)
		{
			for (int j = 0; j<dst.cols; j++)
			{
				int a = dst.at<unsigned short>(i, j);
				int b = lut.at<unsigned short>(I, dst.at<unsigned short>(i, j));
				dst.at<unsigned short>(i, j) = lut.at<unsigned short>(I, dst.at<unsigned short>(i, j));
			}
		}
	}
	dst.convertTo(dst, CV_8UC1);
}
//-----------------------------------------------------------------------------------------------------
// LUT Skeletonizer
//-----------------------------------------------------------------------------------------------------
void cP2pMatching::skel(cv::Mat &src, cv::Mat &dst)
{
	cv::Mat lut;
	GetLutSkel(lut);
	dst = src.clone();
	// ¬±¬â¬Ö¬à¬Ò¬â¬Ñ¬Ù¬å¬Ö¬Þ ¬Ó ¬á¬à¬ã¬Ý¬Ö¬Õ¬à¬Ó¬Ñ¬ä¬Ö¬Ý¬î¬ß¬à¬Þ¬ä¬î 0 ¬Ú 1.
	cv::threshold(dst, dst, 0, 1, cv::THRESH_BINARY);

	int last_pc = INT_MAX;
	for (int pc = cv::countNonZero(dst); pc<last_pc; pc = cv::countNonZero(dst))
	{
		last_pc = pc;
		applylut_8(dst, dst, lut);
	}

	// ¬¹¬ä¬à¬Ò¬í ¬Ò¬í¬Ý¬à ¬Ó¬Ú¬Õ¬ß¬à ¬ß¬Ñ ¬ï¬Ü¬â¬Ñ¬ß¬Ö
	dst = dst * 255;
}

//-----------------------------------------------------------------------------------------------------
// LUT endpoints
//-----------------------------------------------------------------------------------------------------
void cP2pMatching::endp(cv::Mat &src, cv::Mat &dst)
{
	dst = src.clone();
	// ¬±¬â¬Ö¬à¬Ò¬â¬Ñ¬Ù¬å¬Ö¬Þ ¬Ó ¬á¬à¬ã¬Ý¬Ö¬Õ¬à¬Ó¬Ñ¬ä¬Ö¬Ý¬î¬ß¬à¬Þ¬ä¬î 0 ¬Ú 1.
	cv::threshold(dst, dst, 0, 1, cv::THRESH_BINARY);

	applylut_1(dst, dst);

	// ¬¹¬ä¬à¬Ò¬í ¬Ò¬í¬Ý¬à ¬Ó¬Ú¬Õ¬ß¬à ¬ß¬Ñ ¬ï¬Ü¬â¬Ñ¬ß¬Ö
	dst = dst * 255;
}

//-----------------------------------------------------------------------------------------------------
// LUT branchpoins
//-----------------------------------------------------------------------------------------------------
void cP2pMatching::branch(cv::Mat &src, cv::Mat &dst)
{
	dst = src.clone();
	// ¬±¬â¬Ö¬à¬Ò¬â¬Ñ¬Ù¬å¬Ö¬Þ ¬Ó ¬á¬à¬ã¬Ý¬Ö¬Õ¬à¬Ó¬Ñ¬ä¬Ö¬Ý¬î¬ß¬à¬Þ¬ä¬î 0 ¬Ú 1.
	cv::threshold(dst, dst, 0, 255, cv::THRESH_BINARY);
	//cv::imshow("dst", dst);
	/*cv::waitKey();*/
	cv::threshold(dst, dst, 0, 1, cv::THRESH_BINARY);
	applylut_branch(dst, dst);

	// ¬¹¬ä¬à¬Ò¬í ¬Ò¬í¬Ý¬à ¬Ó¬Ú¬Õ¬ß¬à ¬ß¬Ñ ¬ï¬Ü¬â¬Ñ¬ß¬Ö
	cv::threshold(dst, dst, 0, 255, cv::THRESH_BINARY);
	//dst = dst * 255;
}
void cP2pMatching::backcount4(cv::Mat &src, cv::Mat &dst)
{
	dst = src.clone();
	// ¬±¬â¬Ö¬à¬Ò¬â¬Ñ¬Ù¬å¬Ö¬Þ ¬Ó ¬á¬à¬ã¬Ý¬Ö¬Õ¬à¬Ó¬Ñ¬ä¬Ö¬Ý¬î¬ß¬à¬Þ¬ä¬î 0 ¬Ú 1.
	cv::threshold(dst, dst, 0, 1, cv::THRESH_BINARY);

	applylut_backcount4(dst, dst);

	// ¬¹¬ä¬à¬Ò¬í ¬Ò¬í¬Ý¬à ¬Ó¬Ú¬Õ¬ß¬à ¬ß¬Ñ ¬ï¬Ü¬â¬Ñ¬ß¬Ö
	//dst = dst * 255;
}

void cP2pMatching::applylut_branch(cv::Mat &src, cv::Mat &dst)
{
	static int lut_branchpoints[] = { 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	1,	1,	0,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	1,	1,	0,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	1,	1,	0,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	1,	1,	0,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1 };

	cv::Mat k(3, 3, CV_16UC1);

	k.at<unsigned short>(0, 0) = 256;
	k.at<unsigned short>(1, 0) = 128;
	k.at<unsigned short>(2, 0) = 64;
	k.at<unsigned short>(0, 1) = 32;
	k.at<unsigned short>(1, 1) = 16;
	k.at<unsigned short>(2, 1) = 8;
	k.at<unsigned short>(0, 2) = 4;
	k.at<unsigned short>(1, 2) = 2;
	k.at<unsigned short>(2, 2) = 1;

	dst = src.clone();

	filter2D(dst, dst, CV_16UC1, k);
	for (int i = 0; i<dst.rows; i++)
	{
		for (int j = 0; j<dst.cols; j++)
		{
			dst.at<unsigned short>(i, j) = lut_branchpoints[dst.at<unsigned short>(i, j)];
		}
	}

	dst.convertTo(dst, CV_8UC1);
}

void cP2pMatching::applylut_backcount4(cv::Mat &src, cv::Mat &dst)
{
	static int lut_backcount4[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 2, 1, 1, 2, 2, 3, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 3, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 3, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 3, 2, 3, 2, 2, 3, 3, 4, 3, 3, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 3, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 3, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 2, 1, 1, 2, 2, 3, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 3, 3, 4, 3, 3, 2, 2, 3, 2, 3, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 3, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 3, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 2, 1, 1, 2, 2, 3, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0 };

	cv::Mat k(3, 3, CV_16UC1);

	k.at<unsigned short>(0, 0) = 256;
	k.at<unsigned short>(1, 0) = 128;
	k.at<unsigned short>(2, 0) = 64;
	k.at<unsigned short>(0, 1) = 32;
	k.at<unsigned short>(1, 1) = 16;
	k.at<unsigned short>(2, 1) = 8;
	k.at<unsigned short>(0, 2) = 4;
	k.at<unsigned short>(1, 2) = 2;
	k.at<unsigned short>(2, 2) = 1;

	dst = src.clone();

	filter2D(dst, dst, CV_16UC1, k);
	for (int i = 0; i<dst.rows; i++)
	{
		for (int j = 0; j<dst.cols; j++)
		{
			dst.at<unsigned short>(i, j) = lut_backcount4[dst.at<unsigned short>(i, j)];
		}
	}

	dst.convertTo(dst, CV_8UC1);
}

void cP2pMatching::thin(cv::Mat &src, cv::Mat &dst)
{
	dst = src.clone();
	// ¬±¬â¬Ö¬à¬Ò¬â¬Ñ¬Ù¬å¬Ö¬Þ ¬Ó ¬á¬à¬ã¬Ý¬Ö¬Õ¬à¬Ó¬Ñ¬ä¬Ö¬Ý¬î¬ß¬à¬Þ¬ä¬î 0 ¬Ú 1.
	cv::threshold(dst, dst, 0, 1, cv::THRESH_BINARY);
	cv::Mat old;
	dst.copyTo(old);
	
	while (true)
	//for (int i = 0; i < 20; i++)
	{
		applylut_thin1(dst, dst);
		applylut_thin2(dst, dst);
		std::vector<cv::Point> idx;
		//cv::Scalar a = cv::sum(~(old & dst));
		
		cv::findNonZero( (old != dst), idx);
		if (!idx.size())
			break;
		
		//cv::imshow("(old & dst)==1", (old != dst)*255);
		//cv::imshow("dst", dst*255);
		//cv::imshow("old", old*255);
		//cv::waitKey();

		dst.copyTo(old);

	}

	// ¬¹¬ä¬à¬Ò¬í ¬Ò¬í¬Ý¬à ¬Ó¬Ú¬Õ¬ß¬à ¬ß¬Ñ ¬ï¬Ü¬â¬Ñ¬ß¬Ö
	dst = dst * 255;
}

void cP2pMatching::applylut_thin1(cv::Mat &src, cv::Mat &dst)
{
	static int lut_thin1[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	cv::Mat k(3, 3, CV_16UC1);

	k.at<unsigned short>(0, 0) = 256;
	k.at<unsigned short>(1, 0) = 128;
	k.at<unsigned short>(2, 0) = 64;
	k.at<unsigned short>(0, 1) = 32;
	k.at<unsigned short>(1, 1) = 16;
	k.at<unsigned short>(2, 1) = 8;
	k.at<unsigned short>(0, 2) = 4;
	k.at<unsigned short>(1, 2) = 2;
	k.at<unsigned short>(2, 2) = 1;

	dst = src.clone();

	filter2D(dst, dst, CV_16UC1, k);
	for (int i = 0; i<dst.rows; i++)
	{
		for (int j = 0; j<dst.cols; j++)
		{
			dst.at<unsigned short>(i, j) = lut_thin1[dst.at<unsigned short>(i, j)];
		}
	}
	//delete[] lut_thin1;
	dst.convertTo(dst, CV_8UC1);
}

void cP2pMatching::applylut_thin2(cv::Mat &src, cv::Mat &dst)
{
	static int lut_thin2[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1 };

	cv::Mat k(3, 3, CV_16UC1);

	k.at<unsigned short>(0, 0) = 256;
	k.at<unsigned short>(1, 0) = 128;
	k.at<unsigned short>(2, 0) = 64;
	k.at<unsigned short>(0, 1) = 32;
	k.at<unsigned short>(1, 1) = 16;
	k.at<unsigned short>(2, 1) = 8;
	k.at<unsigned short>(0, 2) = 4;
	k.at<unsigned short>(1, 2) = 2;
	k.at<unsigned short>(2, 2) = 1;

	dst = src.clone();

	filter2D(dst, dst, CV_16UC1, k);
	for (int i = 0; i<dst.rows; i++)
	{
		for (int j = 0; j<dst.cols; j++)
		{
			dst.at<unsigned short>(i, j) = lut_thin2[dst.at<unsigned short>(i, j)];
		}
	}

	dst.convertTo(dst, CV_8UC1);
}

cv::Mat cP2pMatching::ExtractPatchWithZeroPadding(cv::Mat img, cv::Point patch_center,int patch_size)
{
	int nY = img.rows;
	int nX = img.cols;

	int half_patch_size = floor(patch_size / 2);
	cv::Mat padded_img(nY + 2 * half_patch_size, nX + 2 * half_patch_size,CV_8UC1);
	padded_img = 0;
	cv::Rect crop(half_patch_size/*:end - half_patch_size*/, half_patch_size /*: end - half_patch_size*/, img.cols, img.rows );
	cv::Mat ROI = padded_img(crop);
	img.copyTo(ROI);


	cv::Point ex_center = patch_center + cv::Point(half_patch_size, half_patch_size);
	crop = cv::Rect(ex_center.x - half_patch_size , ex_center.y - half_patch_size ,
		ex_center.x + half_patch_size - (ex_center.x - half_patch_size)+1, ex_center.y + half_patch_size - (ex_center.y - half_patch_size)+1);
	cv::Mat patch = padded_img(crop);
	
	cv::Mat copy;
	patch.copyTo(copy);
	
	return copy;
}

std::vector<cv::Point> cP2pMatching::bresenham(cv::Point xy1, cv::Point xy2)
{
	xy1 = cv::Point(round(xy1.x),round(xy1.y));
	xy2 = cv::Point(round(xy2.x), round(xy2.y));

	
	int dx = abs(xy2.x - xy1.x);
	int dy = abs(xy2.y - xy1.y);
	bool steep = abs(dy) > abs(dx);
	if (steep)
	{
		int t = dx; dx = dy; dy = t;
	}

	//The main algorithm goes here.
	std::vector<int> q;
	if (dy == 0)
	{
		for (int k = 0; k < dx + 1; k++)
			q.push_back(0);
	}
	else
	{
		//q = [0; diff(mod([floor(dx / 2):-dy : -dy*dx + floor(dx / 2)]',dx))>=0];

		std::vector<int> vList;
		//std::vector<int> vDiff;
		//std::vector<int> q;

		q.push_back(0);
		
		for (int i = floor(dx / 2); i >= -dy*dx + floor(dx / 2); i -= dy)
		{
			if (i % dx < 0)
				vList.push_back((i % dx) + dx);
			else
				vList.push_back(i % dx);

			
			if (vList.size() != 1)
			{
				//vDiff.push_back(vList.end()-vList[vList.size()-2]);

				if (vList.back() - vList[vList.size() - 2] >= 0)
					q.push_back(1);
				else
					q.push_back(0);
			}
			
		}

		
	}
		


	//and ends here.

	std::vector<cv::Point> xy;

	if (steep)
	{
		if (xy1.y <= xy2.y)
		{ 
			/*xy.y = [y1:y2]';*/
			for (int i = xy1.y; i <= xy2.y; i++)
				xy.push_back(cv::Point(0, i));
		}
		else
		{
			/*y = [y1:-1 : y2]';*/ 
			for (int i = xy1.y; i >= xy2.y; i--)
				xy.push_back(cv::Point(0, i));
		}

		if (xy1.x <= xy2.x)
		{ 
			//xy.x = yx1.x + cumsum(q);
			int cumsum = 0;
			for (int i = 0; i < q.size(); i++)
			{
				cumsum += q[i];
				int tmp_x = xy1.x + cumsum;
				xy[i].x = tmp_x;
			}
		}
		else 
		{
			//x = x1 - cumsum(q); 
			int cumsum = 0;
			for (int i = 0; i < q.size(); i++)
			{
				cumsum += q[i];
				int tmp_x = xy1.x - cumsum;
				xy[i].x = tmp_x;
			}
		}
	}
	else
	{
		if (xy1.x <= xy2.x)
		{
			//x = [x1:x2]';
			for (int i = xy1.x; i <= xy2.x; i++)
				xy.push_back(cv::Point(i,0));
		}
		else
		{
			//x = [x1:-1 : x2]'; 
			for (int i = xy1.x; i >= xy2.x; i--)
				xy.push_back(cv::Point(i,0));
		}
		if (xy1.y <= xy2.y)
		{
			//y = y1 + cumsum(q);
			int cumsum = 0;
			for (int i = 0; i < q.size(); i++)
			{
				cumsum += q[i];
				int tmp_y = xy1.y + cumsum;
				xy[i].y = tmp_y;
			}
		}
		else
		{
			//y = y1 - cumsum(q);
			int cumsum = 0;
			for (int i = 0; i < q.size(); i++)
			{
				cumsum += q[i];
				int tmp_y = xy1.y - cumsum;
				xy[i].y = tmp_y;
			}
		}
	
	}

	return xy;
}	

//function[arr_cands, arr_cands_dists] = GetCandidates(img_t, img_tp1, E, tran_vec, d_tp1, idx_img_tp1, ivessel_tp1, p, psift)
void cP2pMatching::GetCandidates(cv::Mat img_t, cv::Mat img_tp1, std::vector<cv::Point> E,
	cv::Point tran_vec, float* d_tp1, int d_tp1_numkeys, cv::Mat idx_img_tp1, cv::Mat ivessel_tp1, cVCOParams p, cv::Mat cand_ivessel,
	cv::Mat &arr_cands, cv::Mat &arr_cands_dists, cv::Mat d_tp1_img)
{
	//[nY, nX] = size(img_t);
	int nY = img_t.rows;
	int nX = img_t.cols;

	

	cv::Mat tCC(E.size(), 1, CV_32SC2);
	for (int i = 0; i < E.size(); i++)
		tCC.at<cv::Point>(i, 0) = E[i];

	// sample points on the current vessel segment to reduce
	// computational burdens
	int len = tCC.rows;
	std::vector<int> samp_idx(round(len/(float)p.sampling_period));
	int sumPeriod = 0;
	for (int i = 0; i < samp_idx.size(); i++)
	{
		
		samp_idx[i] = sumPeriod;
		sumPeriod += p.sampling_period;
	}
	
	if (!samp_idx.size())
	{
		samp_idx.push_back(0);
	}
	
	
	if (samp_idx.back() != len-1)
		samp_idx.push_back(len-1);

	
	cv::Mat tmp_tCC(samp_idx.size(), 1, CV_32SC2);
	for (int i = 0; i < samp_idx.size(); i++)
		tmp_tCC.at<cv::Point>(i) = tCC.at<cv::Point>(samp_idx[i]);

	tCC = tmp_tCC;

	int npt = tCC.rows;
	arr_cands = cv::Mat::zeros(npt, p.n_cands, CV_32SC2);
	arr_cands_dists = cv::Mat(npt, p.n_cands, CV_32FC1);
	arr_cands_dists = INFINITY;


	int numKeyPts = d_tp1_numkeys;

	

	//float sigma = p.psift_scale*p.psift_magnif;
	float size = 0;
	//vl_imsmooth_f(img_vec_smooth, img_tp1.cols, img_vec.data(), img_tp1.cols, img_tp1.rows, img_tp1.cols, sigma, sigma);

	for (int idx_pt = 0; idx_pt < npt; idx_pt++) // for each point in this segment
	{
		if (idx_pt == 10)
			int afasd = 0;
		cv::Point cpt = tCC.at<cv::Point>(idx_pt);
		//[cpt_yy, cpt_xx] = ind2sub([nY, nX], cpt);
		cv::Point cpt_xxyy = cpt;
		//old_cpt_yy = cpt_yy - tran_vec(1);
		//old_cpt_xx = cpt_xx - tran_vec(2);
		cv::Point old_cpt_xxyy = cpt_xxyy - tran_vec;
		if (old_cpt_xxyy.y > nY-1 | old_cpt_xxyy.y < 0 | old_cpt_xxyy.x > nX-1 | old_cpt_xxyy.x < 0)
			continue;

		cv::Point old_cpt = old_cpt_xxyy;
		
		//double fc[] = { old_cpt_xxyy.x, old_cpt_xxyy.y, p.psift_scale, 0 };
				
		
				

		//std::vector<float> b();
		//float *a = b.data();

		//a[0] = 255;
		//const float* test = new float[512 * 512];
		//for (int k = 0; k < 512 * 512; k++)
			
		
		float fPts[4] = { old_cpt_xxyy.x, old_cpt_xxyy.y, p.psift_scale, 0 };
		float *d_t = 0;
		cv::Mat d_t_img;


		cv::Rect rc(fPts[0] - halfPatchSize, fPts[1] - halfPatchSize, patchSize, patchSize);
		cv::Mat crop = img_t(rc);


		d_t_img = cv::Mat(patchSize*patchSize, 1, CV_8UC1);
		for (int y1 = 0; y1 < crop.rows; y1++)
		for (int x1 = 0; x1 < crop.cols; x1++)
		{
			d_t_img.at<uchar>(y1 * patchSize + x1, 0) = crop.at<uchar>(y1, x1);
		}
		d_t_img.convertTo(d_t_img, CV_32FC1);

		cv::Mat cand_dist(nY, nX, CV_8UC1);
		cand_dist = 0;
		rc = cv::Rect(std::max(1, cpt_xxyy.x - p.thre_dist_step2), std::max(1, cpt_xxyy.y - p.thre_dist_step2),
			std::min(nX, cpt_xxyy.x + p.thre_dist_step2) - std::max(1, cpt_xxyy.x - p.thre_dist_step2), std::min(nY, cpt_xxyy.y + p.thre_dist_step2) - std::max(1, cpt_xxyy.y - p.thre_dist_step2));
		cv::Mat roi = cand_dist(rc);
		roi = 255;
		
		/*for (int y = 0; y < roi.rows; y += 2)
		for (int x = 0; x < roi.cols; x += 2)
		{
			roi.at<uchar>(y, x) = 255;
		}*/
		//cv::imshow("check", cand_dist);
		//cv::waitKey();
		//cv::Mat cand_ivessel = ivessel_tp1 >= p.thre_ivessel;
		cv::Mat cand = cand_ivessel&cand_dist;
		//cand_ivessel.release();
		//cand_idx = find(cand);
		cv::Mat cand_idx;
		cv::findNonZero(cand, cand_idx);

		//cv::Mat check(512, 512,CV_8UC1);
		//check = 0;
		//for (int cc = 0; cc < cand_idx.rows; cc++)
		//{
		//	if (idx_img_tp1.at<int>(cand_idx.at<cv::Point>(cc)) != -1)
		//		check.at<uchar>(cand_idx.at<cv::Point>(cc).y, cand_idx.at<cv::Point>(cc).x) = 255;
		//}
		//cv::imshow("check",check);
		//cv::waitKey();
		//cv::Mat dist_img(nY, nX, CV_32FC1, INFINITY);

		cv::Mat dist_img(nY, nX, CV_32FC1);
		crop = dist_img(rc);
		crop = INFINITY;
		//dist_img = INFINITY;
		int k;

		//cv::Mat a;
		//cv::subtract(d_tp1_img, d_t_img, a,cv::noArray());
		//
		////a = d_tp1_img - d_t_img;
		//cv::Mat b = d_tp1_img.col(10) - d_t_img;
		//cv::Mat c = a.col(10);
		
		//double r1 = cv::norm(b,cv::NormTypes::NORM_L2);
		//double r2 = cv::norm(c, cv::NormTypes::NORM_L2);

		//cv::Mat tmp_cand_idx = cv::Mat(p.n_cands, 1, CV_32SC2);
		//tmp_cand_idx = INT_MAX;
		//float fMinValLists[5];
		//cv::Point minValPtLists[5];

		//for (k = 0; k < 5; k++)
		//{
		//	fMinValLists[k] = INFINITY;
		//}
		
		for (k = 0; k < cand_idx.rows; k++)
		{
			
			
			//cv::Mat ta = d_tp1_img.col(idx_img_tp1.at<unsigned short>(cand_idx.at<cv::Point>(k)));
			if (idx_img_tp1.at<int>(cand_idx.at<cv::Point>(k)) != -1)
			{
				dist_img.at<float>(cand_idx.at<cv::Point>(k)) = cv::norm(d_t_img - d_tp1_img.col(idx_img_tp1.at<int>(cand_idx.at<cv::Point>(k))), cv::NormTypes::NORM_L2);


				//double sumv=0;
				//for (int m = 0; m < d_t_img.rows; m++)
				//{
				//	float d = (d_t_img.at<float>(m, 0) - d_tp1_img.col(idx_img_tp1.at<int>(cand_idx.at<cv::Point>(k))).at<float>(m, 0)) ;
				//	sumv += d*d;
				//}
				//float ssdv = sqrt(sumv);
				//dist_img.at<float>(cand_idx.at<cv::Point>(k)) = sqrt(sumv);

				//for (int q = 0; q < 5; q++)
				//{
				//	if (fMinValLists[q] > dist_img.at<float>(cand_idx.at<cv::Point>(k)))
				//	{
				//		for (int p = 5-1; p >q; p--)
				//		{
				//			fMinValLists[p] = fMinValLists[p-1];
				//			minValPtLists[p] = minValPtLists[p - 1];
				//		}
				//		fMinValLists[q] = dist_img.at<float>(cand_idx.at<cv::Point>(k));
				//		tmp_cand_idx.at<cv::Point>(q) = cand_idx.at<cv::Point>(k);
				//		break;
				//	}
				//}
				
			}
				
			
			
		}


		//tmp_cand_idx.copyTo(cand_idx);

		cand_idx = cv::Mat(1, p.n_cands, CV_32SC2);
		cv::Mat tmp_dist_img;
		dist_img.copyTo(tmp_dist_img);
		cand_idx = INT_MAX;
		
		roi = tmp_dist_img(rc);

		for (int k = 0; k < p.n_cands; k++)
		{
			

			int minIdx[2];
			double minV;
			cv::minMaxIdx(roi, &minV, 0, minIdx);

			cand_idx.at<cv::Point>(k) = cv::Point(minIdx[1] + rc.x, minIdx[0] + rc.y);

			//double minvv = DBL_MAX;
			//cv::Point minpt;
			//for (int tt1 = 0; tt1 < roi.rows; tt1++)
			//{

			//	for (int tt2 = 0; tt2 < roi.cols; tt2++)
			//	{
			//		if (roi.at<float>(tt1, tt2) >= INFINITY)
			//			printf("i     ");
			//		else
			//			printf("%0.1f  ", roi.at<float>(tt1, tt2));
			//		if (minvv >= roi.at<float>(tt1, tt2))
			//		{
			//			minvv = roi.at<float>(tt1, tt2);
			//			minpt = cv::Point(tt2, tt1);
			//		}
			//	}
			//	printf("\n");
			//}
			//printf("\n\n");

			tmp_dist_img.at<float>(cv::Point(minIdx[1] + rc.x, minIdx[0] + rc.y)) = INFINITY;


			
			int adf = 0;

		}
		//tmp_dist_img.release();

		for (int k = 0; k < cand_idx.cols; k++)
		{
			//printf("%d, %d\n", cand_idx.at<cv::Point>(k).x, cand_idx.at<cv::Point>(k).y);
			float a = dist_img.at<float>(cand_idx.at<cv::Point>(k));
			arr_cands_dists.at<float>(idx_pt, k) = dist_img.at<float>(cand_idx.at<cv::Point>(k));
			
		}
		

		cand_idx.copyTo(arr_cands.row(idx_pt));
		//cand_idx.release();
		
		delete[] d_t;

	}

	//_CrtDumpMemoryLeaks();

}