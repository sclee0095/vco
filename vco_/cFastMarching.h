#pragma once

//#ifndef _PERFORM_FRONT_PROPAGATION_2D_ADDPD_H_
//#define _PERFORM_FRONT_PROPAGATION_2D_ADDPD_H_
//#endif // _PERFORM_FRONT_PROPAGATION_2D_ADDPD_H_
#include "std_include.h"
//#include "fast_marching/perform_front_propagation_2d_addpd.h"
//#include "fast_marching/perform_front_propagation_2d_addpd.cpp"

#include "fast_marching2/perform_front_propagation_2d_addpd.h"

//#include <math.h>
////#include "fast_marching/config.h"
//#include "fast_marching/config.h"
//#include <stdio.h>
//#include <string.h>
//#include <vector>
//#include <algorithm>

//#include "fast_marching/fheap/fib.h"
//#include "fast_marching/fheap/fibpriv.h"

//#define kDead -1
//#define kOpen 0
//#define kFar 1
//
//#define ACCESS_ARRAY(a,i,j) a[(i)+n*(j)]
//#define D_(i,j) ACCESS_ARRAY(D,i,j)
//#define S_(i,j) ACCESS_ARRAY(S,i,j)
//#define W_(i,j) ACCESS_ARRAY(W,i,j)
//#define H_(i,j) ACCESS_ARRAY(H,i,j)
//#define Q_(i,j) ACCESS_ARRAY(Q,i,j)
//#define L_(i,j) ACCESS_ARRAY(L,i,j)
//#define PD_(i,j) ACCESS_ARRAY(PD,i,j)  // SCLEE: distance in pixels
//#define heap_pool_(i,j) ACCESS_ARRAY(heap_pool,i,j)
//#define start_points_(i,k) start_points[(i)+2*(k)]
//#define end_points_(i,k) end_points[(i)+2*(k)]
//
//// select to test or not to test (debug purpose)
//// #define CHECK_HEAP check_heap(i,j,k);
//#ifndef CHECK_HEAP
//#define CHECK_HEAP
//#endif
//// error display
//// #define ERROR_MSG(a) mexErrMsgTxt(a)
//#ifndef ERROR_MSG
//#define ERROR_MSG(a) 
//#endif
//// #define WARN_MSG(a)  mexWarnMsgTxt(a) 
//#ifndef WARN_MSG
//#define WARN_MSG(a)
//#endif


class cFastMarching
{
public:
	cFastMarching();
	~cFastMarching();

	void fast_marching(double* i_W, int i_Ww, int i_Wh, double* sp, int nsp, double* ep, int nep, double i_nb_iter_max,
		double *i_H, double *i_S, double* i_D, double *i_Q, double *i_PD,
		double** o_D, double** o_S);

	//void compute_geodesic(double *D, double *x);
	//void extract_path_2d(double *D, double *end_point);
	void compute_geodesic(double *D, int w, int h, double *ep, std::vector<cv::Point> *path);
	void compute_geodesic2();
	void compute_discrete_geodesic(cv::Mat D_mat, cv::Point x, std::vector<cv::Point> *path);
	////int ffm_n;			// width
	////int ffm_p;			// height
	////double* ffm_D;
	////double* ffm_S;
	////double* ffm_W;
	////double* ffm_Q;
	////double* ffm_L;
	////double* ffm_PD; // SCLEE: distance in pixels
	////double* ffm_start_points;
	////double* ffm_end_points;
	////double* ffm_H;
	////double* ffm_values;
	////int ffm_nb_iter_max;
	////int ffm_nb_start_points;
	////int ffm_nb_end_points;

	//int n;			// width
	//int p;			// height
	//double* D;
	//double* S;
	//double* W;
	//double* Q;
	//double* L;
	//double* PD; // SCLEE: distance in pixels
	//double* start_points;
	//double* end_points;
	//double* H;
	//double* values;
	//int nb_iter_max;
	//int nb_start_points;
	//int nb_end_points;

	//fibheap_el** heap_pool = NULL;

	//typedef bool(*T_callback_intert_node)(int i, int j, int ii, int jj);

	//// main function
	//void perform_front_propagation_2d_addpd(T_callback_intert_node callback_insert_node, int ffm_n,
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
	//	int ffm_nb_end_points);

	//bool end_points_reached(const int i, const int j);
	//int compare_points(void *x, void *y);
	//void check_heap(int i, int j);

	//struct point
	//{
	//	point(int ii, int jj)
	//	{
	//		i = ii; j = jj;
	//	}
	//	int i, j;
	//};
	//typedef std::vector<point*> point_list;

	// some global variables
	int n;			// width
	int p;			// height
	double* D;
	double* S;
	double* W;
	double* Q;
	double* L;
	double* PD; // SCLEE: distance in pixels
	double* start_points;
	double* end_points;
	double* H;
	double* values;
	int nb_iter_max;
	int nb_start_points;
	int nb_end_points;

	//void perform_front_propagation_2d_addpd(T_callback_intert_node callback_insert_node = NULL);

};

