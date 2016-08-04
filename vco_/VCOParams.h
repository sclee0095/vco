#pragma once

#include "std_include.h"

class VCO_EXPORTS cVCOParams
{
public:
	cVCOParams();
	~cVCOParams();

	// parameters
	bool use_global_chamfer;
	double thre_ivessel;
	int thre_dist_step1;
	int thre_dist_step2;
	int n_junction_cands;
	int n_junction_cc_cands;
	int n_cands;
	int n_all_cands;
	int sampling_period;

	//parameters for fast marching
	double pfm_nb_iter_max;

	// parameters for SIFT
	double psift_scale;
	double psift_magnif;
	double psift_binSize;
	double psift_nnorm; // l - 2 norm

	// parameters for frangi filter
	int pfrangi_FrangiScaleRange[2];
	int pfrangi_FrangiScaleRatio;
	double pfrangi_FrangiBetaOne;
	int pfrangi_FrangiBetaTwo;
	bool pfrangi_verbose;
	bool pfrangi_BlackWhite;
};
