#include "make_dll.h"

//TESTDLL_NKJ void VesselCorrespondenceOptimization(cv::Mat img_t, cv::Mat img_tp1, cv::Mat bimg_t,
//	cParam p, std::string ave_path, int nextNum,
//	cv::Mat* bimg_tp1, cv::Mat* bimg_tp1_post_processed, int fidx_tp1, char* savePath)
TESTDLL_NKJ void VesselCorrespondenceOptimization(double* arr_img_t, double* arr_img_tp1, double* arr_bimg_t, int img_w, int img_h,
	cParam p, std::string ave_path, int nextNum,
	double** arr_bimg_tp1, double** arr_bimg_tp1_post_processed, int fidx_tp1, char* savePath, bool bVerbose)
{
	//std::vector<double> sigma = { 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5 };
	//std::vector<double> sigma = { 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5 };

	cv::Mat img_t(img_h, img_w, CV_64FC1, arr_img_t);
	cv::Mat img_tp1(img_h, img_w, CV_64FC1, arr_img_tp1);
	cv::Mat bimg_t(img_h, img_w, CV_64FC1, arr_bimg_t);
	img_t.convertTo(img_t, CV_8UC1);
	img_tp1.convertTo(img_tp1, CV_8UC1);
	bimg_t.convertTo(bimg_t, CV_8UC1);
	cv::Mat bimg_tp1_post_processed;

	std::vector<cv::Mat> result;
	int nY = img_t.rows;
	int nX = img_t.cols;

	cFrangiFilter frangi;
	cChamferMatching chamf;
	cP2pMatching p2p(18);
	cFastMarching fmm;


	//cv::Mat ivessel_tp1 = frangi.makeFrangiFilter(img_tp1, sigma, result);
	cv::Mat ivessel_tp1;
	frangi.frangi(img_tp1, &ivessel_tp1);
	//cv::imshow("ivessel_tp1", ivessel_tp1);
	//cv::waitKey();

	result.clear();
	// bimg of 't' frame
	p2p.thin(bimg_t, bimg_t);
	//thinning(bimg_t);

	// bimg of 't+1' frame
	cv::Mat bw;
	cv::Mat ivessel_tp1_32f;
	ivessel_tp1.convertTo(ivessel_tp1_32f, CV_32FC1);
	cv::threshold(ivessel_tp1_32f, bw, p.thre_ivessel, 255, 0);
	//ivessel_tp1_32f.deallocate();
	ivessel_tp1_32f.release();
	bw.convertTo(bw, CV_8UC1);

	p2p.thin(bw, bw);
	cv::Mat bimg_tp1;
	bw.copyTo(bimg_tp1);

	cv::Mat gc_canvas_img(img_tp1.rows, img_tp1.cols, CV_8UC3);
	//img_tp1.convertTo(gc_canvas_img,CV_8UC3);
	//gc_canvas_img.convertTo(gc_canvas_img, CV_8UC3);
	//cv::cvtColor(gc_canvas_img, gc_canvas_img, CV_8UC3);
	cv::Mat gt_bimg_t;

	//// global chamfer matching


	int t_x, t_y;
	if (p.use_gc)
	{
		gt_bimg_t = chamf.computeChamferMatch(bimg_t, bimg_tp1, p, t_x, t_y);
	}
	else
	{
		gt_bimg_t = bimg_t;
		t_x = 0; t_y = 0;
	}


	//gc_canvas_img = cv::Mat(nY, nX, CV_8UC3, img_tp1.data[] , img_tp1.da);

	if (img_tp1.channels() == 1)
		cv::cvtColor(img_tp1, gc_canvas_img, CV_GRAY2BGR);
	else if (img_tp1.channels() == 3)
		img_tp1.copyTo(gc_canvas_img);



	///// draw
	char str[200];
	if (bVerbose)
	{

		for (int y = 0; y < gc_canvas_img.rows; y++)
		for (int x = 0; x < gc_canvas_img.cols; x++)
		{
			if (gt_bimg_t.at<uchar>(y, x))
			{
				gc_canvas_img.at<uchar>(y, x * 3 + 0) = 0;
				gc_canvas_img.at<uchar>(y, x * 3 + 1) = 0;
				gc_canvas_img.at<uchar>(y, x * 3 + 2) = 255;
			}
			else if (bimg_t.at<uchar>(y, x))
			{
				gc_canvas_img.at<uchar>(y, x * 3 + 0) = 0;
				gc_canvas_img.at<uchar>(y, x * 3 + 1) = 255;
				gc_canvas_img.at<uchar>(y, x * 3 + 2) = 0;
			}

		}

		sprintf(str, "%s%d-th_frame_gc_rgb.png", savePath, fidx_tp1);
		cv::imwrite(str, gc_canvas_img);
	}


	std::vector<std::vector<cv::Point>> E;
	std::vector<cv::Mat> cell_cands;
	std::vector<cv::Mat> cell_cands_dists;

	std::vector<cv::Point> J;
	//std::vector<std::vector<cv::Point>> E;
	cv::Mat bJ;
	//thinning(gt_bimg_t);
	p2p.thin(gt_bimg_t, gt_bimg_t);

	p2p.run(img_t, img_tp1, gt_bimg_t, p, t_x, t_y, ivessel_tp1, fidx_tp1, savePath, bVerbose, &E, &cell_cands, &cell_cands_dists, &J);



	// MRF regularization
	cMRF mrf;
	cv::Mat unaryCost;
	std::vector<cv::Mat>  pairwiseCost;
	cv::Mat mapMat;
	std::vector<cv::Point>  all_coors;
	cv::Mat all_cands;
	std::vector<std::vector<int>>  cell_coors_to_all_coors;
	mrf.computeCost(E, cell_cands, cell_cands_dists, p, &unaryCost, &pairwiseCost, &mapMat, &all_coors, &all_cands, &cell_coors_to_all_coors);
	//[unaryCost, pairwiseCost, mapMat, all_coors, all_cands, cell_coors_to_all_coors] = ComputeCosts2(E, cell_cands, cell_cands_dists, p);

	//cv::Mat tmp_juntion_view;
	//img_tp1.copyTo(tmp_juntion_view);
	//cv::cvtColor(tmp_juntion_view, tmp_juntion_view, CV_GRAY2BGR);

	//for (int i = 0; i < E.size(); i++)
	//for (int j = 0; j < E[i].size(); j++)
	//{
	//	cv::circle(tmp_juntion_view, E[i][j], 2, CV_RGB(255, 0, 0), -1);
	//}
	//cv::imshow("tmp_juntion_view", tmp_juntion_view);
	//cv::waitKey();
	int nm = 0;

	for (int y = 0; y < mapMat.rows; y++)
	{
		for (int x = 0; x < mapMat.cols; x++)
		{
			if (mapMat.at<double>(y, x))
			{
				nm++;
			}
		}

	}

	double* sparse_mapMat = new double[nm * 3];
	int nonZeroCnt = 0;
	for (int y = 0; y < mapMat.rows; y++)
	for (int x = 0; x < mapMat.cols; x++)
	{
		if (mapMat.at<double>(y, x) && !nonZeroCnt)
		{
			sparse_mapMat[nonZeroCnt * 3 + 0] = x;
			sparse_mapMat[nonZeroCnt * 3 + 1] = y;
			sparse_mapMat[nonZeroCnt * 3 + 2] = mapMat.at<double>(y, x);
			nonZeroCnt++;
		}
		else if (mapMat.at<double>(y, x))
		{
			sparse_mapMat[nonZeroCnt * 3 + 0] = x;
			sparse_mapMat[nonZeroCnt * 3 + 1] = y;
			sparse_mapMat[nonZeroCnt * 3 + 2] = mapMat.at<double>(y, x);
			nonZeroCnt++;
		}
	}

	//mapMat = sparse(mapMat);

	double energy;
	double *labels;

	double **arrayPairwiseCost = new double*[pairwiseCost.size()];

	for (int i = 0; i < pairwiseCost.size(); i++)
	{
		arrayPairwiseCost[i] = new double[pairwiseCost[i].rows*pairwiseCost[i].cols];

		for (int x = 0; x < pairwiseCost[i].cols; x++)
		for (int y = 0; y < pairwiseCost[i].rows; y++)
		{
			arrayPairwiseCost[i][x* pairwiseCost[i].cols + y] = pairwiseCost[i].at<double>(y, x);
		}

	}



	cv::transpose(unaryCost, unaryCost);


	mrf.mrf_trw_s(((double*)unaryCost.data), unaryCost.cols, unaryCost.rows, arrayPairwiseCost, sparse_mapMat, nm, mapMat.cols, mapMat.rows, &energy, &labels);
	//[labels, energy] = mrfMinimizeMex_syshin(unaryCost, pairwiseCost, mapMat);


	for (int i = 0; i < pairwiseCost.size(); i++)
	{
		delete[] arrayPairwiseCost[i];
	}
	delete[] sparse_mapMat;


	//// for visualization(draw nodes & connect edges)


	// detect junctions labeled as 'dummy'
	cv::Mat bJ_all_pts = cv::Mat::zeros(all_coors.size(), 1, CV_8UC1);
	for (int j = 0; j < E.size(); j++)
	{

		for (int k = 0; k < J.size(); k++)
		{
			if (E[j][0] == J[k])
			{
				bJ_all_pts.at<uchar>(cell_coors_to_all_coors[j][0] - 1) = true;
				break;
			}
		}
		//if (E[j], J, 'row)
		//{
		//	bJ_all_pts(cell_coors_to_all_coors{ j }(1)) = true;
		//}

		for (int k = 0; k < J.size(); k++)
		{
			if (E[j][E[j].size() - 1] == J[k])
			{
				bJ_all_pts.at<uchar>(cell_coors_to_all_coors[j][cell_coors_to_all_coors[j].size() - 1] - 1) = true;
				break;
			}
		}
		//if (ismember(E{ j }(end, :), J, 'rows'))
		//{
		//	bJ_all_pts(cell_coors_to_all_coors{ j }(end)) = true;
		//}
	}
	std::vector<cv::Point> bJ_idx;
	cv::findNonZero(bJ_all_pts, bJ_idx);
	//bJ_idx = find(bJ_all_pts);

	std::vector<std::vector<int>> all_joining_seg;
	int num_all_joining_seg = 0;
	for (int j = 0; j < bJ_idx.size(); j++)
	{
		if (labels[bJ_idx[j].y * bJ_all_pts.cols + bJ_idx[j].x] == p.n_all_cands + 1)
		{
			/// find segments joining at this junction
			std::vector<int> joining_seg;
			for (int k = 0; k < E.size(); k++)
			{
				int idx = -1;
				for (int a = 0; a < cell_coors_to_all_coors[k].size(); a++)
				{
					if (cell_coors_to_all_coors[k][a] - 1 == (bJ_idx[j].y * bJ_all_pts.cols + bJ_idx[j].x))
					{
						idx = a;
					}
				}
				//find(cell_coors_to_all_coors{ k } == bJ_idx(j));

				if (idx != -1)
				{
					int meet_pt = -1;
					if (idx == 1)
					{

						for (int a = 0; a < cell_coors_to_all_coors[k].size(); a++)
						{
							if (labels[cell_coors_to_all_coors[k][a] - 1] != p.n_all_cands + 1)
							{
								meet_pt = a;
								break;
							}
						}
						//meet_pt = find(labels(cell_coors_to_all_coors{ k })~= p.n_all_cands + 1, 1, 'first');
					}
					else
					{
						for (int a = cell_coors_to_all_coors[k].size() - 1; a >= 0; a--)
						{
							if (labels[cell_coors_to_all_coors[k][a] - 1] != p.n_all_cands + 1)
							{
								meet_pt = a;
								break;
							}

						}
						//meet_pt = find(labels(cell_coors_to_all_coors{ k })~= p.n_all_cands + 1, 1, 'last');
					}

					if (meet_pt == -1)
					{
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
	//detect junctions labeled as 'dummy'

	std::vector<std::vector<cv::Point>> newE;
	std::vector<cv::Point> all_v;
	std::vector<cv::Point> all_vessel_pt;
	std::vector<cv::Point> cor_line_pt;

	cv::Mat trs;
	ivessel_tp1.copyTo(trs);
	trs.convertTo(trs, CV_64FC1);
	cv::transpose(trs, trs);
	trs.setTo(1e-10, trs < 1e-10);

	cv::Mat H_mat(ivessel_tp1.rows, ivessel_tp1.cols, CV_64FC1);
	H_mat = 0;
	cv::Mat S_mat(ivessel_tp1.rows, ivessel_tp1.cols, CV_64FC1);
	S_mat = 1;
	cv::Mat D_mat(ivessel_tp1.rows, ivessel_tp1.cols, CV_64FC1);
	D_mat = 1e9;
	cv::Mat Q_mat(ivessel_tp1.rows, ivessel_tp1.cols, CV_64FC1);
	Q_mat = -1;
	cv::Mat PD_mat(ivessel_tp1.rows, ivessel_tp1.cols, CV_64FC1);
	PD_mat = -1;

	for (int j = 0; j < E.size(); j++)
	{
		std::vector<cv::Point> temp_v;
		cv::Mat temp = cv::Mat::zeros(nY, nX, CV_64FC1);
		std::vector<int> t_seg = cell_coors_to_all_coors[j];
		int len_t_seg = t_seg.size();
		cv::Point st_pt = cv::Point(-1, -1);
		cv::Point ed_pt = cv::Point(-1, -1);
		std::vector<cv::Point> cum_seg_path;
		bool is_first = true;


		for (int k = 0; k < len_t_seg; k++)
		{
			int t_idx = t_seg[k] - 1;
			double t_label = labels[t_idx] - 1;
			if (t_label < p.n_all_cands)
			{
				//int pt1_y = all_coors[t_idx].y; int pt1_x = all_coors[t_idx].x;
				//cv::Point pt2 = all_cands.at<cv::Point>(t_idx, t_label);
				////[pt2_y, pt2_x] = ind2sub([nY, nX], pt2);
				//int pt2_y = pt2.y; int pt2_x = pt2.x;
				////[t_path_x, t_path_y] = bresenham(pt1_x, pt1_y, pt2_x, pt2_y);
				//std::vector<cv::Point> t_path = p2p.bresenham(cv::Point(pt1_x,pt1_y),pt2);
				////cor_line_pt = [cor_line_pt;[t_path_y, t_path_x]];
				//for (int a = 0; a < t_path.size(); a++)
				//	cor_line_pt.push_back(t_path[a]);

				if (st_pt.x == -1)
				{
					st_pt = all_cands.at<cv::Point>(t_idx, (int)t_label);
					//temp_v = [temp_v; st_pt];

					temp_v.push_back(st_pt);
				}
				else
				{
					ed_pt = all_cands.at<cv::Point>(t_idx, (int)t_label);
					//temp_v = [temp_v; ed_pt];
					temp_v.push_back(ed_pt);
				}
			}
			else
			{
				ed_pt = cv::Point(-1, -1);
			}
			if (st_pt.x != -1 && ed_pt.x != -1)
			{
				//[st_pt_y, st_pt_x] = ind2sub([nY, nX], st_pt);
				//[ed_pt_y, ed_pt_x] = ind2sub([nY, nX], ed_pt);
				int st_pt_y = st_pt.y; int st_pt_x = st_pt.x;
				int ed_pt_y = ed_pt.y; int ed_pt_x = ed_pt.x;
				// straight line
				//[t_path_x, t_path_y] = bresenham(st_pt_x, st_pt_y, ed_pt_x, ed_pt_y);
				// geodesic path
				double pfm_end_points[] = { ed_pt_y, ed_pt_x };
				double pfm_start_points[] = { st_pt_y, st_pt_x };
				//[D, S] = perform_fast_marching(ivessel_tp1, pfm_start_points, pfm_end_points);
				double nb_iter_max = std::min(p.pfm_nb_iter_max, 1.2*std::max(ivessel_tp1.rows, ivessel_tp1.cols)*std::max(ivessel_tp1.rows, ivessel_tp1.cols));

				double *D, *S, *H;

				//double minvv;
				//double maxvv;
				//cv::minMaxIdx(ivessel_tp1, &minvv, &maxvv);
				//cv::Mat ivessel_tp1_view = (ivessel_tp1 - minvv) / (maxvv-minvv) * 255.f;
				//ivessel_tp1_view.convertTo(ivessel_tp1_view, CV_8UC1);
				//cv::cvtColor(ivessel_tp1_view, ivessel_tp1_view, CV_GRAY2BGR);

				//cv::circle(ivessel_tp1_view, st_pt, 2, CV_RGB(0, 0, 255),-1);
				//cv::circle(ivessel_tp1_view, ed_pt,2,CV_RGB(255,0,0),-1);

				//cv::imshow("ivessel_tp1_view", ivessel_tp1_view);
				//cv::waitKey();
				S_mat = 1;
				D_mat = 1e9;
				Q_mat = -1;
				PD_mat = -1;
				fmm.fast_marching(((double*)trs.data), ivessel_tp1.cols, ivessel_tp1.rows, pfm_start_points, 1, pfm_end_points, 1, nb_iter_max,
					(double*)H_mat.data, (double*)S_mat.data, (double*)D_mat.data, (double*)Q_mat.data, (double*)PD_mat.data,
					&D, &S);

				std::vector<cv::Point> geo_path;
				cv::Mat D_mat(ivessel_tp1.rows, ivessel_tp1.cols, CV_64FC1, D);
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
				//fmm.compute_geodesic(D, ivessel_tp1.cols, ivessel_tp1.rows, pfm_end_points, &geo_path);
				fmm.compute_discrete_geodesic(D_mat, cv::Point(pfm_end_points[1], pfm_end_points[0]), &geo_path);
				//geo_path = compute_geodesic(D, [ed_pt_y; ed_pt_x]);
				//geo_path = round(geo_path);


				//[b, m, n] = unique(geo_path','rows','first');
				//geo_path = geo_path(:, sort(m))';
				//geo_path = flipud(fliplr(geo_path));
				//t_path_x = geo_path(:, 1);
				//t_path_y = geo_path(:, 2);

				if (is_first)
				{
					cum_seg_path = geo_path;
					//cum_seg_path = [cum_seg_path; [t_path_y, t_path_x]];

					is_first = false;
				}
				else
				{
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
		newE.push_back(cum_seg_path);
		if (cum_seg_path.size())
		{
			for (int k = 0; k < temp_v.size(); k++)
				all_v.push_back(temp_v[k]);
			//all_v = [all_v; temp_v];
			//lidx = sub2ind([nY, nX], cum_seg_path(:, 1), cum_seg_path(:, 2));

			for (int k = 0; k < cum_seg_path.size(); k++)
			{

				all_vessel_pt.push_back(cum_seg_path[k]);

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
			int t_idx = cell_coors_to_all_coors[joining_seg[k * 2 + 0]][joining_seg[k * 2 + 1]] - 1;
			int t_label = labels[t_idx] - 1;
			cv::Point t_coor1 = all_cands.at<cv::Point>(t_idx, t_label);
			//[st_pt_y, st_pt_x] = ind2sub([nY, nX], t_coor1);
			int st_pt_y = t_coor1.y;
			int st_pt_x = t_coor1.x;

			for (int m = k; m < n_joining_seg / 2; m++)
			{
				int t_idx = cell_coors_to_all_coors[joining_seg[m * 2 + 0]][joining_seg[m * 2 + 1]] - 1;
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

				double nb_iter_max = std::min(p.pfm_nb_iter_max, 1.2*std::max(ivessel_tp1.rows, ivessel_tp1.cols)*std::max(ivessel_tp1.rows, ivessel_tp1.cols));

				double *D, *S;
				S_mat = 1;
				D_mat = 1e9;
				Q_mat = -1;
				PD_mat = -1;
				fmm.fast_marching(((double*)trs.data), ivessel_tp1.cols, ivessel_tp1.rows, pfm_start_points, 1, pfm_end_points, 1, nb_iter_max,
					(double*)H_mat.data, (double*)S_mat.data, (double*)D_mat.data, (double*)Q_mat.data, (double*)PD_mat.data,
					&D, &S);

				std::vector<cv::Point> geo_path;
				cv::Mat D_mat(ivessel_tp1.rows, ivessel_tp1.cols, CV_64FC1, D);
				cv::transpose(D_mat, D_mat);
				//fmm.compute_geodesic(D, ivessel_tp1.cols, ivessel_tp1.rows, pfm_end_points, &geo_path);
				fmm.compute_discrete_geodesic(D_mat, cv::Point(pfm_end_points[1], pfm_end_points[0]), &geo_path);
				//[D, S] = perform_fast_marching(ivessel_tp1, [st_pt_y; st_pt_x], pfm);
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
			all_vessel_pt.push_back(cum_path[k]);
		//lidx = sub2ind([nY, nX], cum_path(:, 1), cum_path(:, 2));
		//all_vessel_pt = [all_vessel_pt; lidx];
	}
	// drawing for junctions labeled as 'dummy'

	std::vector<int> tmp_all_vessel_pt(all_vessel_pt.size());
	for (int avp = 0; avp < all_vessel_pt.size(); avp++)
	{
		tmp_all_vessel_pt[avp] = (all_vessel_pt[avp].y * 512 + all_vessel_pt[avp].x);
	}
	tmp_all_vessel_pt.erase(std::unique(tmp_all_vessel_pt.begin(), tmp_all_vessel_pt.end()), tmp_all_vessel_pt.end());
	all_vessel_pt.clear();
	all_vessel_pt = std::vector<cv::Point>(tmp_all_vessel_pt.size());
	for (int avp = 0; avp < tmp_all_vessel_pt.size(); avp++)
	{
		all_vessel_pt[avp] = (cv::Point(tmp_all_vessel_pt[avp] % 512, tmp_all_vessel_pt[avp] / 512));
	}
	//all_vessel_pt.erase(std::unique(all_vessel_pt.begin(), all_vessel_pt.end()));
	//all_vessel_pt = unique(all_vessel_pt);

	cv::Mat draw_bimg_tp1(nY, nX, CV_8UC1);
	draw_bimg_tp1 = 0;
	//draw_bimg_tp1 = false(nY, nX);

	for (int avp = 0; avp < all_vessel_pt.size(); avp++)
	{
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
	if (bVerbose)
	{

		sprintf(str, "%s%d-th_frame_final_b.png", savePath, fidx_tp1);
		cv::imwrite(str, draw_bimg_tp1);
	}

	cv::Mat final_canvas_img;
	cv::cvtColor(img_tp1, final_canvas_img, CV_GRAY2BGR);
	//final_canvas_img = zeros(nY, nX, 3);
	//final_canvas_img(:, : , 1) = img_tp1;
	//final_canvas_img(:, : , 2) = img_tp1;
	//final_canvas_img(:, : , 3) = img_tp1;

	final_canvas_img.setTo(cv::Scalar(255, 0, 0), draw_bimg_tp1);

	if (bVerbose)
	{

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

	cv::Mat new_bimg;
	std::vector<cv::Point> new_lidx, app_lidx;

	GrowVesselUsingFastMarching(ivessel_tp1, all_vessel_pt, p.thre_ivessel, p, &new_bimg, &new_lidx, &app_lidx);


	p2p.thin(new_bimg, new_bimg);
	new_bimg.copyTo(bimg_tp1_post_processed);
	if (bVerbose)
	{
		sprintf(str, "%s%d-th_frame_final_post_processed_b.png", savePath, fidx_tp1);
		cv::imwrite(str, bimg_tp1_post_processed);



		cv::cvtColor(img_tp1, final_canvas_img, CV_GRAY2BGR);

		final_canvas_img.setTo(cv::Scalar(255, 0, 0), new_bimg);


		sprintf(str, "%s%d-th_frame_final_post_processed_rgb.png", savePath, fidx_tp1);
		cv::imwrite(str, final_canvas_img);

	}
	cv::Mat tmp_bimg_tp1;
	draw_bimg_tp1.convertTo(tmp_bimg_tp1, CV_64FC1);
	cv::Mat tmp_bimg_tp1_post_processed;
	bimg_tp1_post_processed.convertTo(tmp_bimg_tp1_post_processed, CV_64FC1);

	double* arr_tmp_bimg_tp1 = new double[tmp_bimg_tp1_post_processed.rows*tmp_bimg_tp1_post_processed.cols];
	double* arr_tmp_bimg_tp1_post_processed = new double[tmp_bimg_tp1_post_processed.rows*tmp_bimg_tp1_post_processed.cols];
	for (int y = 0; y < tmp_bimg_tp1_post_processed.rows; y++)
	for (int x = 0; x < tmp_bimg_tp1_post_processed.cols; x++)
	{
		arr_tmp_bimg_tp1[y*tmp_bimg_tp1_post_processed.cols + x] = tmp_bimg_tp1.at<double>(y, x);
		arr_tmp_bimg_tp1_post_processed[y*tmp_bimg_tp1_post_processed.cols + x] = tmp_bimg_tp1_post_processed.at<double>(y, x);
	}
	*arr_bimg_tp1 = arr_tmp_bimg_tp1;
	*arr_bimg_tp1_post_processed = arr_tmp_bimg_tp1_post_processed;

	//*arr_bimg_tp1 = NULL;
	//*arr_bimg_tp1_post_processed = NULL;

	return;
}

TESTDLL_NKJ void GrowVesselUsingFastMarching(cv::Mat ivessel, std::vector<cv::Point> lidx, double thre, cParam p,
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

		cv::Mat tmp(nY, nX, CV_8UC1);
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

TESTDLL_NKJ void GetLineLength(std::vector<cv::Point> L, bool IS3D, double *o_ll)
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

TESTDLL_NKJ void getBoundaryDistance(cv::Mat I, bool IS3D, cv::Mat *o_BoundaryDistance)
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

TESTDLL_NKJ void maxDistancePoint(cv::Mat BoundaryDistance, cv::Mat I, bool IS3D, cv::Point *o_posD, double *o_maxD)
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
//TESTDLL_NKJ void maxDistancePoint(cv::Mat BoundaryDistance, cv::Mat I, bool IS3D, double *o_maxD)
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
