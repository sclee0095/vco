#include "VCOParams.h"

cVCOParams::cVCOParams()
{
	use_global_chamfer = true;
	//thre_ivessel = 0.05;
	thre_ivessel = 0.05; //0.6
	thre_dist_step1 = 50;
	thre_dist_step2 = 10;
	n_junction_cands = 10;
	n_junction_cc_cands = 2;
	n_cands = 5;
	n_all_cands = (1 + 2 * n_junction_cc_cands)*n_cands;
	sampling_period = 5;

	//parameters for fast marching
	pfm_nb_iter_max = INFINITY;

	// parameters for SIFT
	psift_scale = 3.0;
	psift_magnif = 3;
	psift_binSize = psift_magnif*psift_scale;
	psift_nnorm = 2; // l - 2 norm

	// parameters for frangi filter
	pfrangi_FrangiScaleRange[0] = 2;
	pfrangi_FrangiScaleRange[1] = 7;
	pfrangi_FrangiScaleRatio = 1;
	pfrangi_FrangiBetaOne = 0.5;
	pfrangi_FrangiBetaTwo = 15;
	pfrangi_verbose = true;
	pfrangi_BlackWhite = false;
}

cVCOParams::~cVCOParams()
{
}