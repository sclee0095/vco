#include "cFastMarching.h"


cFastMarching::cFastMarching()
{
}


cFastMarching::~cFastMarching()
{
}


//#include "mex.h"
void cFastMarching::fast_marching(double* i_W, int i_Ww, int i_Wh, double* sp, int nsp, double* ep, int nep, double i_nb_iter_max,
	double *i_H, double *i_S, double* i_D, double *i_Q, double *i_PD,
	double** o_D, double** o_S)
{
	///* retrive arguments */
	//if (nrhs<4)
	//	mexErrMsgTxt("4 - 7 input arguments are required.");
	//if (nlhs<1)
	//	mexErrMsgTxt("1, 2, 3 or 4 output arguments are required.");

	// first argument : weight list
	n = i_Ww;
	p = i_Wh;
	W = i_W;
	// second argument : start_points
	start_points = sp;
	//start_points = mxGetPr(prhs[1]);

	int tmp = nsp;
	nb_start_points = nsp;
	//if (nb_start_points == 0 || tmp != 2)
		//mexErrMsgTxt("start_points must be of size 2 x nb_start_poins.");
		// third argument : end_points
		end_points = ep;
	tmp = nep;
	nb_end_points = nep;
	//if (nb_end_points != 0 && tmp != 2)
		//mexErrMsgTxt("end_points must be of size 2 x nb_end_poins.");
		//  argument 4: nb_iter_max
		nb_iter_max = (int)i_nb_iter_max;
	//  argument 5: heuristic
	//if (nrhs >= 5)
	//{
	//	H = mxGetPr(prhs[4]);
	//	if (mxGetM(prhs[4]) == 0 && mxGetN(prhs[4]) == 0)
	//		H = NULL;
	//	if (H != NULL && (mxGetM(prhs[4]) != n || mxGetN(prhs[4]) != p))
	//		mexErrMsgTxt("H must be of size n x p.");
	//}
	//else
	//	H = NULL;
	//// argument 6: constraint map
	//if (nrhs >= 6)
	//{
	//	L = mxGetPr(prhs[5]);
	//	if (mxGetM(prhs[5]) == 0 && mxGetN(prhs[5]) == 0)
	//		H = NULL;
	//	if (L != NULL && (mxGetM(prhs[5]) != n || mxGetN(prhs[5]) != p))
	//		mexErrMsgTxt("L must be of size n x p.");
	//}
	//else
	//	L = NULL;
	// argument 7: value list
	//if (nrhs >= 7)
	//{
	//values = H;
	//if (mxGetM(prhs[6]) == 0 && mxGetN(prhs[6]) == 0)
	//	values = NULL;
	//if (values != NULL && (mxGetM(prhs[6]) != nb_start_points || mxGetN(prhs[6]) != 1))
	//	mexErrMsgTxt("values must be of size nb_start_points x 1.");
	//}
	//else
	//	values = NULL;

	H = i_H;
	L = NULL;
	values = NULL;

	// first ouput : distance
	//D = new double[n*p];
	D = i_D;
	//plhs[0] = mxCreateDoubleMatrix(n, p, mxREAL);
	//D = mxGetPr(plhs[0]);
	// second output : state
	//if (nlhs >= 2)
	//{
	//	plhs[1] = mxCreateDoubleMatrix(n, p, mxREAL);
	//	S = mxGetPr(plhs[1]);
	//}
	//else
	//{
	//	S = new double[n*p];
	//}
	//// third output : index
	//if (nlhs >= 3)
	//{
	//	plhs[2] = mxCreateDoubleMatrix(n, p, mxREAL);
	//	Q = mxGetPr(plhs[2]);
	//}
	//else
	//{
	//	Q = new double[n*p];
	//}
	//// SCLEE: PD
	//if (nlhs >= 4)
	//{
	//	plhs[3] = mxCreateDoubleMatrix(n, p, mxREAL);
	//	PD = mxGetPr(plhs[3]);
	//}
	//else
	//{
	//	PD = new double[n*p];
	//}

	S = i_S;
	//Q = new double[n*p];
	//PD = new double[n*p];
	Q = i_Q;
	PD = i_PD;


	// launch the propagation
	//perform_front_propagation_2d_addpd(NULL,n,p,D,S,W,Q,L,PD,start_points,end_points,H,values,nb_iter_max,nb_start_points,nb_end_points);
	//perform_front_propagation_2d_addpd();
	perform_front_propagation_2d_addpd(n, p, D, S, W, Q, L, PD, start_points, end_points, H, values, nb_iter_max, nb_start_points, nb_end_points,&D);

	//int i_n,
	//	int i_p,
	//	double* i_D,
	//	double* i_S,
	//	double* i_W,
	//	double* i_Q,
	//	double* i_L,
	//	double* i_PD,
	//	double* i_start_points,
	//	double* i_end_points,
	//	double* i_H,
	//	double* i_values,
	//	int i_nb_iter_max,
	//	int i_nb_start_points,
	//	int i_nb_end_points,


	*o_D = D;
	*o_S = S;

	//delete[] Q;
	//delete[] PD;
	//if (nlhs<2)
	//	GW_DELETEARRAY(S);
	//if (nlhs<3)
	//	GW_DELETEARRAY(Q);
	//if (nlhs<4)
	//	GW_DELETEARRAY(PD);
	return;
}

//void cFastMarching::compute_geodesic(double *D, double *x, double **path)
//{
//	path = extract_path_2d(D, x);
//}
//void cFastMarching::extract_path_2d(double *D, double *end_point)
//{
//
//	cv::Mat D_mat(512, 512, CV_64FC1, D);
//	int trim_path = 1;
//	double stepsize = 0.1;
//	int maxverts = 10000;
//
//	// gradient computation
//
//	std::vector<cv::Point> I;
//	cv::findNonZero(D_mat == 1e9, I);
//	std::vector<cv::Point> J;
//	cv::findNonZero(D_mat != 1e9, J);
//	cv::Mat A1;
//	D_mat.copyTo(A1);
//	double maxv = 0;
//	//cv::minMaxIdx(D_mat, NULL, &maxv);
//	for (int i = 0; i < J.size(); i++)
//	{
//		if (A1.at<double>(J[i]) > maxv)
//			maxv = A1.at<double>(J[i]);
//	}
//	for (int i = 0; i < I.size(); i++)
//	{
//		A1.at<double>(I[i]) = maxv;
//	}
//	//A1(I) = mmax(A(J));
//	cv::Mat grad_x, grad_y;
//	cv::Sobel(A1, grad_x, CV_64FC1, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
//	cv::Sobel(A1, grad_y, CV_64FC1, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
//
//	//global grad;
//	//grad = compute_grad(A1);
//
//	cv::Mat combine(A1.rows, A1.cols, CV_64FC1);
//	for (int y = 0; y < combine.rows; y++)
//	for (int x = 0; x < combine.cols; x++)
//	{
//		combine.at<double>(y, x) = grad_x.at<double>(y, x)*grad_x.at<double>(y, x) + grad_y.at<double>(y, x)*grad_y.at<double>(y, x);
//	}
//	//n = sum(v1. ^ 2, 3);
//	double eps = 2.2204e-16;
//	cv::findNonZero(n < eps, I);
//	for (int i = 0; i < I.size(); i++)
//	{
//		combine.at<double>(I[i]) = 1;
//	}
//	//n(I) = 1;
//
//	cv::Mat v2[2];
//	v2[0] = cv::Mat(A1.rows, A1.cols, CV_64FC1);
//	v2[1] = cv::Mat(A1.rows, A1.cols, CV_64FC1);
//
//	cv::Mat tmp;
//	cv::sqrt(combine, tmp);
//	for (int i = 0; i < 2; i++)
//	{
//		v2[i] = combine * (1 / tmp);
//		v2[i] = -v2[i];
//	}
//	
//	//v2 = prod_vf_sf(v1, 1. / sqrt(n));
//	//grad = -perform_vf_normalization(grad);
//}

void cFastMarching::compute_geodesic(double *D, int w, int h, double *ep, std::vector<cv::Point> *path)
{
	cv::Mat D_mat(h, w, CV_64FC1, D);

	cv::Mat diffX(h, w, CV_64FC1);
	cv::Mat diffY(h, w, CV_64FC1);

	for (int y = 0; y < h; y++)
	for (int x = 0; x < w; x++)
	{
		if (x == 0 )
		{
			diffX.at<double>(y, x) = (D_mat.at<double>(y, x) - D_mat.at<double>(y, x + 1)) / (double)2;
		}
		else if (x == w - 1)
		{
			diffX.at<double>(y, x) = (D_mat.at<double>(y, x-1) - D_mat.at<double>(y, x)) / (double)2;
		}
		else
		{

			diffX.at<double>(y, x) = (D_mat.at<double>(y, x - 1) - D_mat.at<double>(y, x + 1)) / (double)2;
		}
		if (y == 0)
		{
			diffY.at<double>(y, x) = (D_mat.at<double>(y, x) - D_mat.at<double>(y + 1, x)) / (double)2;
		}
		else if (y == h - 1)
		{
			diffY.at<double>(y, x) = (D_mat.at<double>(y - 1, x) - D_mat.at<double>(y , x)) / (double)2;
		}
		else
		{
			diffY.at<double>(y, x) = (D_mat.at<double>(y - 1, x) - D_mat.at<double>(y + 1, x)) / (double)2;
		}
	}
	
	double min;
	int minPt[2];

	cv::minMaxIdx(D_mat, &min, 0, minPt);

	cv::Point sp = cv::Point(minPt[1], minPt[0]);
	cv::Point cur_pt = sp;

	cv::Mat D_cost(h, w, CV_64FC1);
	D_cost = DBL_MAX;
	D_cost.at<double>(sp) = 0;

	std::vector<std::vector<cv::Point>> layer;

	int nLayer = 0;
	while (true)
	{
		nLayer++;

		int fixed_y = nLayer;
		int fixed_x = nLayer;
		std::vector<cv::Point> cur_layer;

		
		for (int x = -nLayer + sp.x; x <= nLayer + sp.x; x++)
		{
			if (x >= 1 && x < w - 1 && fixed_y >= 1 && fixed_y < h - 1)
				cur_layer.push_back(cv::Point(x,fixed_y));
		}
		
		for (int y = -nLayer + sp.y; y <= nLayer + sp.y; y++)
		{
			if (fixed_x >= 1 && fixed_x < w - 1 && y >= 1 && y < h - 1)
				cur_layer.push_back(cv::Point(-fixed_x, y));
		}

		for (int x = -nLayer + sp.x; x <= nLayer + sp.x; x++)
		{
			if (x >= 1 && x < w - 1 && fixed_y >= 1 && fixed_y < h - 1)
				cur_layer.push_back(cv::Point(x, -fixed_y));
		}

		for (int y = -nLayer + sp.y; y <= nLayer + sp.y; y++)
		{
			if (fixed_x >= 1 && fixed_x < w - 1 && y >= 1 && y < h - 1)
				cur_layer.push_back(cv::Point(fixed_x, y));
		}
		layer.push_back(cur_layer);
		if (nLayer - sp.x >= 1 && nLayer - sp.y >= 1 && nLayer < w - 1 && nLayer < h - 1)
		{
			break;
		}
	}
	for (int i = 0; i < layer.size(); i++)
	{
		cv::Mat tmp_D_cost;
		D_cost.copyTo(tmp_D_cost);

		
		for (int j = 0; j < layer[i].size(); j++)
		{
			int y1 = layer[i][j].y;
			int x1 = layer[i][j].x;

			double curV = std::sqrt(diffY.at<double>(cur_pt + cv::Point(x1, y1))*diffY.at<double>(cur_pt + cv::Point(x1, y1)) + 
				diffX.at<double>(cur_pt + cv::Point(x1, y1))*diffX.at<double>(cur_pt + cv::Point(x1, y1)));
			double cur_range_min = DBL_MAX;
			cv::Point cur_range_min_pt = cv::Point(0,0);
			for (int y2 = -1; y2 <= 1; y2++)
			for (int x2 = -1; x2 <= 1; x2++)
			{
				if (y1 == x1 == 0)
				{
					continue;
				}
				//D_cost.at<double>(cur_pt.y + y1 + y2, cur_pt.x + x1 + x2)
				
				double cur_neiborV = D_cost.at<double>(cur_pt.y + y1 + y2, cur_pt.x + x1 + x2);
				if (cur_neiborV == DBL_MAX)
					continue;
				if (cur_range_min > cur_neiborV)
				{
					cur_range_min = cur_neiborV ;
					cur_range_min_pt = cv::Point(cur_pt.x + x1 + x2,cur_pt.y + y1 + y2);
				}
			}

			tmp_D_cost.at<double>(cur_pt + cv::Point(x1, y1)) = D_cost.at<double>(cur_range_min_pt) +curV;
			
			cur_pt = cur_pt + cv::Point(x1, y1);

			
		}
		tmp_D_cost.copyTo(D_cost);

		if (cur_pt == cv::Point(ep[0], ep[1]))
		{
			break;
		}

		
	}




	cur_pt = sp;

	std::vector<cv::Point> vPath;
	while (true)
	{
		vPath.push_back(cur_pt);
		cv::Point minCostPt;
		double minCost = DBL_MAX;
		for (int y1 = -1; y1 <= 1; y1++)
		for (int x1 = -1; x1 <= 1; x1++)
		{
			if (y1 == x1 == 0)
			{
				continue;
			}
			if (minCost >D_cost.at<double>(cur_pt + cv::Point(x1, y1)))
			{
				minCost = D_cost.at<double>(cur_pt + cv::Point(x1, y1));
				minCostPt = cv::Point(cur_pt + cv::Point(x1, y1));
			}
		}

		

		if (cur_pt == cv::Point(ep[0], ep[1]))
		{
			break;
		}

		cur_pt = minCostPt;
	}

	path = &vPath;
}

//void cFastMarching::perform_front_propagation_2d_addpd(T_callback_intert_node callback_insert_node, int ffm_n,
//	int ffm_p,			// height
//	double* ffm_D,
//	double* ffm_S,
//	double* ffm_W,
//	double* ffm_Q,
//	double* ffm_L,
//	double* ffm_PD, // SCLEE: distance in pixels
//	double* ffm_start_points,
//	double* ffm_end_points,
//	double* ffm_H,
//	double* ffm_values,
//	int ffm_nb_iter_max,
//	int ffm_nb_start_points,
//	int ffm_nb_end_points)
//{
//
//	int n = ffm_n;
//	int p = ffm_p;	// height
//	double* D = ffm_D;
//	double* S = ffm_S;
//	double* W = ffm_W;
//	double* Q = ffm_Q;
//	double* L = ffm_L;
//	double* PD = ffm_PD;
//	double* start_points = ffm_start_points;
//	double* end_points = ffm_end_points;
//	double* H = ffm_H;
//	double* values = ffm_values;
//	int nb_iter_max = ffm_nb_iter_max;
//	int nb_start_points = ffm_nb_start_points;
//	int nb_end_points = ffm_nb_end_points;
//
//	// create the Fibonacci heap
//	struct fibheap* open_heap = fh_makeheap();
//	fh_setcmp(open_heap, (voidcmp)compare_points);
//
//	double h = 1.0 / n;
//
//	// initialize points
//	for (int i = 0; i<n; ++i)
//	for (int j = 0; j<p; ++j)
//	{
//		D_(i, j) = GW_INFINITE;
//		S_(i, j) = kFar;
//		Q_(i, j) = -1;
//		PD_(i, j) = -1; // SCLEE: distance in pixels
//	}
//
//	// record all the points
//	heap_pool = new fibheap_el*[n*p];
//	memset(heap_pool, NULL, n*p*sizeof(fibheap_el*));
//
//	// inialize open list
//	point_list existing_points;
//	for (int k = 0; k<nb_start_points; ++k)
//	{
//		int i = (int)start_points_(0, k);
//		int j = (int)start_points_(1, k);
//
//		if (D_(i, j) == 0)
//			ERROR_MSG("start_points should not contain duplicates.");
//
//		point* pt = new point(i, j);
//		existing_points.push_back(pt);			// for deleting at the end
//		heap_pool_(i, j) = fh_insert(open_heap, pt);			// add to heap
//		if (values == NULL)
//			D_(i, j) = 0;
//		else
//			D_(i, j) = values[k];
//		S_(i, j) = kOpen;
//		Q_(i, j) = k;
//		PD_(i, j) = 0;  // SCLEE: distance in pixels
//	}
//
//	// perform the front propagation
//	int num_iter = 0;
//	bool stop_iteration = GW_False;
//	while (!::fh_isempty(open_heap) && num_iter<nb_iter_max && !stop_iteration)
//	{
//		num_iter++;
//
//		// current point
//		point& cur_point = *((point*)fh_extractmin(open_heap));
//		int i = cur_point.i;
//		int j = cur_point.j;
//		heap_pool_(i, j) = NULL;
//		S_(i, j) = kDead;
//		stop_iteration = end_points_reached(i, j);
//
//		/*
//		char msg[200];
//		sprintf(msg, "Cool %f", Q_(i,j) );
//		WARN_MSG( msg );
//		*/
//
//		CHECK_HEAP;
//
//		// recurse on each neighbor
//		int nei_i[4] = { i + 1, i, i - 1, i };
//		int nei_j[4] = { j, j + 1, j, j - 1 };
//		for (int k = 0; k<4; ++k)
//		{
//			int ii = nei_i[k];
//			int jj = nei_j[k];
//			bool bInsert = true;
//			if (callback_insert_node != NULL)
//				bInsert = callback_insert_node(i, j, ii, jj);
//			// check that the contraint distance map is ok
//			if (ii >= 0 && jj >= 0 && ii<n && jj<p && bInsert)
//			{
//				// SCLEE: WAHT IS P???
//				double P = h / W_(ii, jj);
//				// compute its neighboring values
//				double a1 = GW_INFINITE;
//				int k1 = -1;
//				int pd1 = -1; // SCLEE: distance in pixels
//				if (ii<n - 1)
//				{
//					bool bParticipate = true;
//					if (callback_insert_node != NULL)
//						bParticipate = callback_insert_node(ii, jj, ii + 1, jj);
//					if (bParticipate)
//					{
//						a1 = D_(ii + 1, jj);
//						k1 = Q_(ii + 1, jj);
//						pd1 = PD_(ii + 1, jj); // SCLEE: distance in pixels
//					}
//				}
//				if (ii>0)
//				{
//					bool bParticipate = true;
//					if (callback_insert_node != NULL)
//						bParticipate = callback_insert_node(ii, jj, ii - 1, jj);
//					if (bParticipate)
//					{
//						if (D_(ii - 1, jj)<a1) {
//							k1 = Q_(ii - 1, jj);
//							pd1 = PD_(ii - 1, jj);// SCLEE: distance in pixels
//						}
//						a1 = GW_MIN(a1, D_(ii - 1, jj));
//					}
//				}
//				double a2 = GW_INFINITE;
//				int k2 = -1;
//				int pd2 = -1; // SCLEE: distance in pixels
//				if (jj<p - 1)
//				{
//
//					bool bParticipate = true;
//					if (callback_insert_node != NULL)
//						bParticipate = callback_insert_node(ii, jj, ii, jj + 1);
//					if (bParticipate)
//					{
//						a2 = D_(ii, jj + 1);
//						k2 = Q_(ii, jj + 1);
//						pd2 = PD_(ii, jj + 1); // SCLEE: distance in pixels
//					}
//				}
//				if (jj>0)
//				{
//					bool bParticipate = true;
//					if (callback_insert_node != NULL)
//						bParticipate = callback_insert_node(ii, jj, ii, jj - 1);
//					if (bParticipate)
//					{
//						if (D_(ii, jj - 1)<a2) {
//							k2 = Q_(ii, jj - 1);
//							pd2 = PD_(ii, jj - 1);// SCLEE: distance in pixels
//						}
//						a2 = GW_MIN(a2, D_(ii, jj - 1));
//					}
//				}
//				if (a1>a2)	// swap so that a1<a2
//				{
//					double tmp = a1; a1 = a2; a2 = tmp;
//					int tmpi = k1; k1 = k2; k2 = tmpi;
//					int tmppd = pd1; pd1 = pd2; pd2 = tmppd; // SCLEE: distance in pixels
//				}
//				// update its distance
//				// now the equation is   (a-a1)^2+(a-a2)^2 = P, with a >= a2 >= a1.
//				double A1 = 0;
//				if (P*P > (a2 - a1)*(a2 - a1))
//				{
//					double delta = 2 * P*P - (a2 - a1)*(a2 - a1);
//					A1 = (a1 + a2 + sqrt(delta)) / 2.0;
//				}
//				else
//					A1 = a1 + P;
//				if (((int)S_(ii, jj)) == kDead)
//				{
//					// check if action has change. Should not happen for FM
//					// if( A1<D_(ii,jj) )
//					//	WARN_MSG("The update is not monotone");
//#if 1
//					if (A1<D_(ii, jj))	// should not happen for FM
//					{
//						D_(ii, jj) = A1;
//						// update the value of the closest starting point
//						//if( GW_ABS(a1-A1)<GW_ABS(a2-A1) && k1>=0  )
//						Q_(ii, jj) = k1;
//						PD_(ii, jj) = pd1 + 1; // SCLEE: distance in pixels
//						//else
//						//	Q_(ii,jj) = k2;
//						//Q_(ii,jj) = Q_(i,j);
//					}
//#endif
//				}
//				else if (((int)S_(ii, jj)) == kOpen)
//				{
//					// check if action has change.
//					if (A1<D_(ii, jj))
//					{
//						D_(ii, jj) = A1;
//						// update the value of the closest starting point
//						//if( GW_ABS(a1-A1)<GW_ABS(a2-A1) && k1>=0  )
//						Q_(ii, jj) = k1;
//						PD_(ii, jj) = pd1 + 1; // SCLEE: distance in pixels
//						//else
//						//	Q_(ii,jj) = k2;
//						//Q_(ii,jj) = Q_(i,j);
//						// Modify the value in the heap
//						fibheap_el* cur_el = heap_pool_(ii, jj);
//						if (cur_el != NULL)
//							fh_replacedata(open_heap, cur_el, cur_el->fhe_data);	// use same data for update
//						else
//							ERROR_MSG("Error in heap pool allocation.");
//					}
//				}
//				else if (((int)S_(ii, jj)) == kFar)
//				{
//					if (D_(ii, jj) != GW_INFINITE)
//						ERROR_MSG("Distance must be initialized to Inf");
//					if (L == NULL || A1 <= L_(ii, jj))
//					{
//						S_(ii, jj) = kOpen;
//						// distance must have change.
//						D_(ii, jj) = A1;
//						// update the value of the closest starting point
//						//if( GW_ABS(a1-A1)<GW_ABS(a2-A1) && k1>=0 )
//						Q_(ii, jj) = k1;
//						PD_(ii, jj) = pd1 + 1; // SCLEE: distance in pixels
//						//else
//						//	Q_(ii,jj) = k2;
//						//Q_(ii,jj) = Q_(i,j);
//						// add to open list
//						point* pt = new point(ii, jj);
//						existing_points.push_back(pt);
//						heap_pool_(ii, jj) = fh_insert(open_heap, pt);			// add to heap	
//					}
//				}
//				else
//					ERROR_MSG("Unkwnown state.");
//
//			}	// end switch
//		}		// end for
//	}			// end while
//
//	//				char msg[200];
//	//				sprintf(msg, "Cool %f", Q_(100,100) );
//	//				 WARN_MSG( msg ); 
//
//	// free heap
//	fh_deleteheap(open_heap);
//	// free point pool
//	for (point_list::iterator it = existing_points.begin(); it != existing_points.end(); ++it)
//		GW_DELETE(*it);
//	// free fibheap pool
//	GW_DELETEARRAY(heap_pool);
//}
//
//
//
//inline
//bool cFastMarching::end_points_reached(const int i, const int j)
//{
//	for (int k = 0; k<nb_end_points; ++k)
//	{
//		if (i == ((int)end_points_(0, k)) && j == ((int)end_points_(1, k)))
//			return true;
//	}
//	return false;
//}
//
//inline
//int cFastMarching::compare_points(void *x, void *y)
//{
//	point& a = *((point*)x);
//	point& b = *((point*)y);
//	if (H == NULL)
//		return cmp(D_(a.i, a.j), D_(b.i, b.j));
//	else
//		return cmp(D_(a.i, a.j) + H_(a.i, a.j), D_(b.i, b.j) + H_(b.i, b.j));
//}
//
//
//// test the heap validity
//void cFastMarching::check_heap(int i, int j)
//{
//	for (int x = 0; x<n; ++x)
//	for (int y = 0; y<p; ++y)
//	{
//		if (heap_pool_(x, y) != NULL)
//		{
//			point& pt = *(point*)heap_pool_(x, y)->fhe_data;
//			if (H == NULL)
//			{
//				if (D_(i, j)>D_(pt.i, pt.j))
//					ERROR_MSG("Problem with heap.\n");
//			}
//			else
//			{
//				if (D_(i, j) + H_(i, j)>D_(pt.i, pt.j) + H_(pt.i, pt.j))
//					ERROR_MSG("Problem with heap.\n");
//			}
//		}
//	}
//}

void cFastMarching::compute_geodesic2()
{

}

void cFastMarching::compute_discrete_geodesic(cv::Mat D_mat,cv::Point x, std::vector<cv::Point> *o_path)
{
	//function path = compute_discrete_geodesic(D, x)

	// compute_discrete_geodesic - extract a discrete geodesic in 2D and 3D
	//
	//   path = compute_discrete_geodesic(D, x);
	//
	//   Same as extract_path_xd but less precise and more robust.
	//
	//   Copyright(c) 2007 Gabriel Peyre

	int nd = 2;
	
	std::vector<cv::Point> path;
		
	path.push_back(x);

	// admissible moves
	double dx[] = { 1, -1, 0, 0 };
	double dy[] = { 0 ,0, 1, -1 };
	
	cv::Mat d(2,4,CV_32SC1);
	for (int cy = 0; cy < d.rows; cy++)
	for (int cx = 0; cx < d.cols; cx++)
	{
		if (cy == 0)
			d.at<int>(cy, cx) = dx[cx];
		else
			d.at<int>(cy, cx) = dy[cx];

	}
	//double vprev = D[x(1)x(2)];
	double vprev = D_mat.at<double>(x);
	
	//double minval;
	//double maxval;
	//int minidx1[2], maxidx[2];
	//cv::minMaxIdx(D_mat, &minval, &maxval,minidx1,maxidx);
	//cv::Mat view = ((D_mat - minval) / (maxval - minval)) * 255;
	//view.convertTo(view, CV_8UC1);
	//cv::imshow("view", view);
	//cv::waitKey();

	//s = size(D);
	int sy = D_mat.rows;
	int sx = D_mat.cols;

	
	while (true)
	{
		cv::Point x0 = path[path.size()-1];
		cv::Mat x0_mat(2, 1, CV_32SC1);
		x0_mat.at<int>(0, 0) = x0.x;
		x0_mat.at<int>(1, 0) = x0.y;
		cv::Mat pts = cv::repeat(x0_mat, 1, d.cols);
		pts = pts+d;
		//x = repmat(x0, 1, size(d, 2)) + d;


		cv::Mat find_mat =pts.row(0) >= 0 & pts.row(1) >= 0 & pts.row(0) < sx & pts.row(1) < sy;
		std::vector<cv::Point> idx;
		cv::findNonZero(find_mat, idx);
		//I = find(x(1, :) > 0 & x(2, :) > 0 & x(1, :) <= s(1) & x(2, :) <= s(2));
		cv::Mat tmp_pts(2,idx.size(),CV_32SC1);
		
		for (int i = 0; i < idx.size(); i++)
		{
			tmp_pts.at<int>(0, i) = pts.at<int>(0, idx[i].x);
			tmp_pts.at<int>(1, i) = pts.at<int>(1, idx[i].x);
		}
		tmp_pts.copyTo(pts);
		//x = x(:, I);

		//I = x(1, :) + (x(2, :) - 1)*s(1);

		double minv = DBL_MAX;
		int minIdx;
		for (int i = 0; i < pts.cols; i++)
		{
			if (minv > D_mat.at<double>(pts.at<int>(1, i), pts.at<int>(0, i)))
			{
				minv = D_mat.at<double>(pts.at<int>(1, i), pts.at<int>(0, i));
				minIdx = i;
			}
		}
		//[v, J] = min(D(I));

		cv::Point selected = cv::Point(pts.at<int>(0, minIdx), pts.at<int>(1, minIdx));
		//x = x(:, J);
		//if v>vprev
		if (minv >= vprev) // revised by syshin
		{
			*o_path = path;
			return;
		}

		vprev = minv;
		path.push_back(selected);
	}

	
}