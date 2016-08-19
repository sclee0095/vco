#include "std_include.h" // include neccesary header files

#include "cParam.h" // for vco parameter
#include <Windows.h>
#include "../vco_/VCO.h"
//#include "VCO.h"



#ifdef _DEBUG 
#pragma comment(lib,"vco64d.lib")
#else
#pragma comment(lib,"vco64.lib")
#endif

// if you want to make center line mask to use only first frame mask, 
// put in true to TEST_CONTINUE_MASK
#define TEST_CONTINUE_MASK false


typedef std::wstring str_t;

// read in folder
std::vector<std::string> get_file_in_folder(std::string folder, std::string file_type = "*.*");

// if you want detail, bVerbose put in true;
bool bVerbose = false;
int main()
{
	
	// set image root path
	std::string root_path = "F:/VCO/vco/IMG2";

	// set catheter mask root path
	std::string catheter_mask_root_path = "F:/VCO/vco/catheter_mask";

	// set vessel mask root path
	std::string vessel_mask_root_path = "F:/VCO/vco/vessel_mask";

	// set stored root path
	std::string result_root_path = "F:/VCO/vco/result";

	// create image case lists and get llists
	std::vector<std::string> case_list;
	case_list = get_file_in_folder(root_path);

	int n_case = case_list.size();

	// set vco parameter(defalte)
	cParam p;

	// compute vco each image
	for (int i = 0; i <1; i++)
	{
		std::vector<std::string> seq_list = get_file_in_folder(root_path + '/' + case_list[i]);

		int n_seq = seq_list.size();

		for (int j = 0; j <1; j++)
		{
			std::vector<std::string> frame_list;
			std::string cur_str = root_path + '/' + case_list[i] + '/' + seq_list[j];
			frame_list = get_file_in_folder(cur_str);

			cur_str = vessel_mask_root_path + '/' + case_list[i] + '/' + seq_list[j];
			std::vector<std::string> mask_list = get_file_in_folder(cur_str);


			int n_masks = mask_list.size();
			std::string buf = mask_list[0];
			buf.pop_back(); buf.pop_back(); buf.pop_back(); buf.pop_back();
			int start_frame = atoi(buf.c_str());;

			buf = mask_list[n_masks - 1];
			buf.pop_back(); buf.pop_back(); buf.pop_back(); buf.pop_back();
			int end_frame = atoi(buf.c_str());;


			double **t_quan_res = new double*[end_frame - start_frame + 1];
			for (int p1 = 0; p1 < end_frame - start_frame + 1; p1++)
			{
				t_quan_res[p1] = new double[24];
				for (int q1 = 0; q1 < 24; q1++)
				{
					t_quan_res[p1][q1] = 0;
				}
			}
			cv::Mat seq_bimg_t;
			cv::Mat new_t_vscl_mask;
			for (int k = start_frame; k < end_frame - 1; k++)
			{
				// check running time
				clock_t start_time = clock();

				if (k == start_frame+4)
				{
					int asd = 0;
				}
				printf("%d of %d in seq(%d)\n", k - start_frame + 1, end_frame - start_frame + 1, k+1);

				
				// read to t frame image
				cur_str = root_path + '/' + case_list[i] + '/' + seq_list[j] + '/' + frame_list[k - start_frame];
				cv::Mat img_t = cv::imread(cur_str, 0);

				// read to  t+1 frame image
				cur_str = root_path + '/' + case_list[i] + '/' + seq_list[j] + '/' + frame_list[k - start_frame + 1];
				cv::Mat img_tp1 = cv::imread(cur_str, 0);

				cv::Mat bimg_t;
				bimg_t = 0;
				if (TEST_CONTINUE_MASK && k != start_frame)
				{
					new_t_vscl_mask.copyTo(bimg_t);
				}
				else
				{
					// read to t frame vessel mask
					cur_str = vessel_mask_root_path + "/" + case_list[i] + "/" + seq_list[j] + "/" + mask_list[k - start_frame];
					bimg_t = cv::imread(cur_str, 0);
				}
				


				// make stored path and stored folder
				std::string save_dir_path = result_root_path + "/" + case_list[i] + "/";
				std::string save_path = result_root_path + "/" + case_list[i] + "/" + seq_list[j] + "/";

				_mkdir(save_dir_path.data());
				_mkdir(save_path.data());
				

				// create image for vco input & output using opencv
				cv::Mat img_t_64f, img_tp1_64f, bimg_t_64f;
				img_t.convertTo(img_t_64f, CV_64FC1);
				img_tp1.convertTo(img_tp1_64f, CV_64FC1);
				bimg_t.convertTo(bimg_t_64f, CV_64FC1);
				double* arr_img_t, *arr_img_tp1, *arr_bimg_t;
				arr_img_t = ((double*)img_t_64f.data);
				arr_img_tp1 = ((double*)img_tp1_64f.data);
				arr_bimg_t = ((double*)bimg_t_64f.data);

				
				char spath[200];
				sprintf(spath, "%s", save_path.data());

				// compute vco, input & output data tpye is doulbe
				cVCO vco(arr_img_t, arr_bimg_t, arr_img_tp1, k + 1, img_t.cols, img_t.rows, bVerbose, spath);
				vco.VesselCorrespondenceOptimization();
				
				
				// get center line mask of cv::Mat form
				// get_tp1_vescl_mask() function is returned pre-post processing mask of cv::Mat form
				// get_tp1_vescl_mask_pp() function is returned post-post processing mask of cv::Mat form
				cv::Mat tp1_vmask = vco.get_tp1_vescl_mask();
				cv::Mat tp1_vmask_pp = vco.get_tp1_vescl_mask_pp();


				// get cente line mask of double form(512x512)
				// get_p_tp1_mask_8u() function is returned pre-post processing mask of unsigned char array pointer
				// get_p_tp1_mask_pp_8u() function is returned post-post processing mask of unsigned char  array pointer
				unsigned char*tp_p_vmask_8u = vco.get_p_tp1_mask_8u();
				unsigned char*tp_p_vmask_pp_8u = vco.get_p_tp1_mask_pp_8u();

				// get each segment 2d array vector
				// getVsegVpts2dArr() function is returned pre-post processing each segment 2d array vector
				// getVsegVpts2dArr_pp() function is returned post-post processing each segment 2d array vector
				std::vector<std::vector<cv::Point>> tp1_vsegm_vpt_2darr = vco.getVsegVpts2dArr();
				std::vector<std::vector<cv::Point>> tp1_vsegm_vpt_2darr_pp = vco.getVsegVpts2dArr_pp();

				// get to the selected as uniform term point
				// get_t_vpts_arr() function is returned points to get to the selecte for uniform of t frame
				// get_tp1_vpts_arr() function is returned points to get to the selecte for uniform of tp1 frame
				std::vector<cv::Point> t_vpt_arr = vco.get_t_vpts_arr();
				std::vector<cv::Point> tp1_vpt_arr = vco.get_tp1_vpts_arr();
				
				// get displacement vector. this is based selected as uniform term point
				std::vector<cv::Point> disp_vec_arr = vco.get_disp_vec_arr();

				// get feature points. if type is 0, it is end feature. another(1) is junction points 
				std::vector<ves_feat_pt> t_features = vco.get_t_VesFeatPts();
				std::vector<ves_feat_pt> tp1_features = vco.get_tp1_VesFeatPts();

				// get vesselness to frangi filter
				float* p_vesselnessFrangi = vco.get_tp1_p_FrangiVesselnessMask();
				cv::Mat vesselnessFrangi = vco.get_tp1_FrangiVesselnessMask();

				// get to linked information form after post processing segmentations of the t+1 frame
				// note : this is not tree. Just linked information to relation at each segmentation
				std::vector<std::vector<std::vector<std::vector<int>>>> tp1_segm_linked 
					= vco.get_tp1_segm_linked_information();

				
				// if TEST_CONTINUE_MASK is true, write center line mask of t+1 frame
				if (TEST_CONTINUE_MASK)
				{
					new_t_vscl_mask = vco.get_tp1_adjusted_vescl_mask_pp();
					cv::imshow("new_t_vscl_mask", new_t_vscl_mask);

					
					std::string save_mask_root = "F:/VCO/vco/continued_mask";
					std::string save_mask = save_mask_root + "/" + case_list[i] + "/" + seq_list[j] + "/" + mask_list[k - start_frame+1].data();
					save_mask.pop_back(); save_mask.pop_back(); save_mask.pop_back();
					save_mask += "png";
					std::string save_dir1 = save_mask_root + "/" + case_list[i];
					std::string save_dir2 = save_mask_root + "/" + case_list[i] + "/" + seq_list[j];

					_mkdir(save_dir1.data());
					_mkdir(save_dir2.data());

					
					cv::imwrite(save_mask, new_t_vscl_mask);
				}

				// write center line mask of t+1 frame for using VCO
				if (!bVerbose)
				{
					cur_str = result_root_path + '/' + case_list[i] + '/' + seq_list[j] + '/' + frame_list[k - start_frame + 1];
					cur_str.pop_back(); cur_str.pop_back(); cur_str.pop_back();
					cur_str += "png";
					cv::imwrite(cur_str.data(), tp1_vmask_pp);
				}

			
				// for visualiazation using opencv library
				cv::imshow("tp1_vmask", tp1_vmask);
				cv::imshow("tp1_vmask_pp", tp1_vmask_pp);
				cv::waitKey(500);

				// check running time
				clock_t end_time = clock();
				printf("Elapsed: %f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);


				// release
				delete[] tp_p_vmask_8u, tp_p_vmask_pp_8u;
				
			}

			for (int p1 = 0; p1 < end_frame - start_frame + 1; p1++)
			{

				delete t_quan_res[p1];

			}
			delete[] t_quan_res;
		}

	}





}

std::vector<std::string> get_file_in_folder(std::string folder, std::string file_type)
{
	std::vector<std::string> folder_names;

	std::string search_path;
	char a[200];
	wsprintf(a, "%s/%s", folder.c_str(), file_type.c_str());

	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(a, &fd);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do{
			if ((fd.cFileName[0]) != ('.'))
				folder_names.push_back((std::string)fd.cFileName);
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}

	return folder_names;
}
