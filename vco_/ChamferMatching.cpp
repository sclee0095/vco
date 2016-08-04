#include "ChamferMatching.h"


cChamferMatching::cChamferMatching()
{
}


cChamferMatching::~cChamferMatching()
{
}

cv::Mat cChamferMatching::computeChamferMatch(cv::Mat bimg_t, cv::Mat bimg_tp1,cVCOParams p,int &t_x, int &t_y)
{
	int nY = bimg_tp1.rows;
	int nX = bimg_tp1.cols;


	cv::Mat dt_tp1;
	cv::distanceTransform(~bimg_tp1, dt_tp1, cv::DistanceTypes::DIST_L1, cv::DistanceTransformMasks::DIST_MASK_3,CV_32FC1);
	cv::Mat idx;
	cv::findNonZero(bimg_t, idx);
	cv::Point *pts = ((cv::Point*)idx.data);
	cv::Mat rows(idx.rows, 1, CV_32SC1);
	cv::Mat cols(idx.rows, 1, CV_32SC1);

	//cv::Mat dt_tp1_8u;
	//dt_tp1.convertTo(dt_tp1_8u, CV_8UC1,255);
	//cv::imshow("dt_tp1_8u", dt_tp1_8u);
	//cv::waitKey();
	for (int i = 0; i < idx.rows; i++)
	{
		cols.at<int>(i, 0) = pts[i].x;
		rows.at<int>(i, 0) = pts[i].y;
	}

	double minX, minY;
	double maxX, maxY;

	cv::minMaxIdx(cols, &minX, &maxX);
	cv::minMaxIdx(rows, &minY, &maxY);

	if ((((int)maxX) - ((int)minX)) % 2 == 1)
	{
		if (maxX < nX)
			maxX = maxX + 1;
		else
			minX = minX - 1;
	}



	if ((((int)maxY) - ((int)minY)) % 2 == 1)
	{
		if (maxY < nY)
			maxY = maxY + 1;
		else
			minY = minY - 1;
	}


	minY = std::fmax(1, minY);
	maxY = std::fmin(nY, maxY);
	minX = std::fmax(1, minX);
	maxX = std::fmin(nX, maxX);

	cv::Rect crop(minX, minY, maxX - minX + 1, maxY - minY + 1);
	cv::Mat cropped_bimg_t = bimg_t(crop);

	cv::Mat matchImg = ChamferMatch(dt_tp1, cropped_bimg_t);

	
	/// added for the sequence test
	int cpt_yy = round((minY + maxY) / 2);
	int cpt_xx = round((minX + maxX) / 2);
	cv::Mat cand(nY, nX,CV_8UC1);
	cand = 0;

	crop = cv::Rect(std::max(1, cpt_yy - p.thre_dist_step1), std::max(1, cpt_xx - p.thre_dist_step1),
		std::min(nY, cpt_yy + p.thre_dist_step1) - std::max(1, cpt_yy - p.thre_dist_step1),
		std::min(nX, cpt_xx + p.thre_dist_step1) - std::max(1, cpt_xx - p.thre_dist_step1));
	//cand(std::max(1, cpt_yy - p.thre_dist_step1) :std::min(nY, cpt_yy + p.thre_dist_step1), std::max(1, cpt_xx - p.thre_dist_step1) : std::min(nX, cpt_xx + p.thre_dist_step1)) = true;
	cv::Mat roi = cand(crop);
	roi = 255;
	cv::Mat reverCand = ~cand;
	
	cv::findNonZero(reverCand, idx);
	pts = ((cv::Point*)idx.data);

	for (int i = 0; i < idx.rows; i++)
	{
		matchImg.at<double>(pts[i].y,pts[i].x) = DBL_MAX;
	}
	
	/// added for the sequence test

	double minval;
	int minIdx[2];
	cv::minMaxIdx(matchImg, &minval, 0,minIdx);
	
	int yy = minIdx[0];
	int xx = minIdx[1];
	int hy = floor(cropped_bimg_t.rows / 2);
	int hx = floor(cropped_bimg_t.cols / 2);

	t_x = xx - hx - minX;
	t_y = yy - hy - minY;
	if (minval == DBL_MAX)
	{
		t_x = 0;
		t_y = 0;
	}

	
	cv::findNonZero(bimg_t,idx);
	//[byy, bxx] = find(bimg_t);
	cv::Mat gt_bimg_t(nY, nX, CV_8UC1);
	gt_bimg_t = 0;
	for (int i = 0; i < idx.rows; i++)
	{
		idx.at<cv::Point>(i, 0).x += t_x;
		idx.at<cv::Point>(i, 0).y += t_y;

		if (idx.at<cv::Point>(i, 0).x >= 0 && idx.at<cv::Point>(i, 0).y >= 0 && idx.at<cv::Point>(i, 0).x < nX && idx.at<cv::Point>(i, 0).y < nY)
			gt_bimg_t.at<uchar>(idx.at<cv::Point>(i, 0).y, idx.at<cv::Point>(i, 0).x) = 255;
	}
	idx.release();
	
	
	//imwrite(gt_bimg_t, strcat(save_path, sprintf('%d-th_frame_gc_b.bmp', fidx_tp1)));
	//gc_canvas_img(sub2ind([nY, nX, 3], byy, bxx, ones(length(byy), 1))) = true;
	//gc_canvas_img(sub2ind([nY, nX, 3], byy, bxx, 2 * ones(length(byy), 1))) = false;
	//gc_canvas_img(sub2ind([nY, nX, 3], byy, bxx, 3 * ones(length(byy), 1))) = false;
	//imwrite(gc_canvas_img, strcat(save_path, sprintf('%d-th_frame_gc_rgb.bmp', fidx_tp1)));

	//delete[] pts;
	
	return gt_bimg_t;
}

cv::Mat cChamferMatching::ChamferMatch(cv::Mat dtImg, cv::Mat tempImg)
{
	dtImg.convertTo(dtImg, CV_64F);
	int r1 = dtImg.rows;
	int c1 = dtImg.cols;

	int r2 = tempImg.rows;
	int c2 = tempImg.cols;

	cv::Mat matchImg(r1, c1, CV_64FC1);

	matchImg = DBL_MAX;

	
	int umax = c1 - c2;
	int vmax = r1 - r2;
	int uc = floor(c2 / 2);
	int vc = floor(r2 / 2);
	int u_sp = 0;
	int v_sp = 0;

	int idxSz;
	//cv::Point addr = GetAddressTable(tempImg, r1, idxSz);

	std::vector<cv::Point> idx;
	cv::findNonZero(tempImg, idx);
	//cv::Point *pts = new cv::Point[idx.rows];
	/*cv::Point *pts ;
	pts = ((cv::Point*)idx.data);*/

	idxSz = idx.size();

	// offsets = sub2ind([r1, c1], 1:vmax, 1 : umax);
	// scores = ones(1, length(offsets))*flintmax;
	// for i = 1:length(offsets)
		// scores(i) = sum(dtImg(offsets(i) + addr));
	// end
		// matchImg(1:umax, 1 : vmax) = reshape(scores, [umax, vmax]);

	//double *tmpAddr = new double[addr.rows];
	//double tmpAddr = (double*)addr.data;

	//double *tmpDtImg = (double*)dtImg.data;
	//for (int u = u_sp ; u < umax; u++)
	//for (int v = v_sp ; v < vmax; v++)
	//{
	//	int offset = (u - 1)*r1 + v - 1;
	//	double score = 0;
	//	
	//	for (int i = 0; i < idxSz; i++)
	//	{
	//		//score += tmpDtImg[offset + (int)(tmpAddr[i])];
	//		score += dtImg.at<double>(v + pts[i].y, u + pts[i].x);
	//	}
	//	
	//	matchImg.at<double>(vc + v, uc + u) = score;
	//}

	for (int y = v_sp; y < vmax; y++)
	for (int x = u_sp; x < umax; x++)
	{
		//int offset = (u - 1)*r1 + v - 1;
		double score = 0;

		for (int i = 0; i < idxSz; i++)
		{
			//score += tmpDtImg[offset + (int)(tmpAddr[i])];
			score += dtImg.at<double>(y + idx[i].y, x + idx[i].x);
		}

		if (uc + x == 154 && vc+y == 129)
		{
			int a = 0;
		}

		matchImg.at<double>(vc + y, uc + x) = score;
	}

	double minV=DBL_MAX;
	int minIdx[2];
	for (int y = 0; y < vmax; y++)
	for (int x = 0; x < umax; x++)
	{
		if (minV > matchImg.at<double>(y, x))
		{
			minV = matchImg.at<double>(y, x);
			minIdx[0] = x;
			minIdx[1] = y;
		}
	}
	//delete &pts;
	idx.clear();
	
	

	return matchImg;
}

cv::Point* cChamferMatching::GetAddressTable(cv::Mat tempImg, int stride, int &idxSz)
{
	int r2 = tempImg.rows;
	int c2 = tempImg.cols;
	cv::Mat idx;
	cv::findNonZero(tempImg, idx);
	cv::Point *pts = new cv::Point[idx.rows];
	pts = ((cv::Point*)idx.data);

	idxSz = idx.rows;
	//cv::Mat rr(idx.rows,1,CV_32SC1);
	//cv::Mat cc(idx.rows, 1, CV_32SC1);
	//
	//
	//for (int i = 0; i < idx.rows; i++)
	//{
	//	cc.at<int>(i, 0) = pts[i].x;
	//	rr.at<int>(i, 0) = pts[i].y;

	//	printf("%d, %d\n", pts[i].x, pts[i].y);
	//}


	return pts;
}
