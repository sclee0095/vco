#include "std_include.h" // include neccesary header files

#include "cParam.h" // for vco parameter
#include <Windows.h>

#ifdef _DEBUG 
#pragma comment(lib,"vco64d.lib")
#else
#pragma comment(lib,"vco64.lib")
#endif

// make function using dll
//extern "C" 
__declspec(dllimport) void VesselCorrespondenceOptimization(double*, double*, double*, int, int,
	cParam, std::string, int, double**, double**, int, char*, bool=false);



typedef std::wstring str_t;

// read in folder
std::vector<std::string> get_file_in_folder(std::string folder, std::string file_type = "*.*");

bool bVerbose = true;
int main()
{
	// set image root path
	std::string root_path = "E:/300_autoSeg/IMG2";

	// set catheter mask root path
	std::string catheter_mask_root_path = "E:/300_autoSeg/shin/dataset/catheter_mask";

	// set vessel mask root path
	std::string vessel_mask_root_path = "E:/300_autoSeg/shin/dataset/vessel_mask";

	// set stored root path
	std::string result_root_path = "E:/300_autoSeg/c++_porting_result";

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

			for (int k = start_frame; k < end_frame - 1; k++)
			{
				// check running time
				clock_t start_time = clock();


				printf("%d of %d in seq(%d,%d)\n", k - start_frame + 1, end_frame - start_frame + 1, i, j);

				// read to t frame image
				cur_str = root_path + '/' + case_list[i] + '/' + seq_list[j] + '/' + frame_list[k - start_frame];
				cv::Mat img_t = cv::imread(cur_str, 0);

				// read to  t+1 frame image
				cur_str = root_path + '/' + case_list[i] + '/' + seq_list[j] + '/' + frame_list[k - start_frame + 1];
				cv::Mat img_tp1 = cv::imread(cur_str, 0);

				// read to t frame vessel mask
				cur_str = vessel_mask_root_path + "/" + case_list[i] + "/" + seq_list[j] + "/" + mask_list[k - start_frame];
				cv::Mat bimg_t = cv::imread(cur_str, 0);

				// make stored path and stored folder
				std::string save_dir_path = result_root_path + "/" + case_list[i] + "/";
				std::string save_path = result_root_path + "/" + case_list[i] + "/" + seq_list[j] + "/";
				if (bVerbose)
				{
					_mkdir(save_dir_path.data());
					_mkdir(save_path.data());
				}

				// create image for vco input & output using opencv
				cv::Mat img_t_64f, img_tp1_64f, bimg_t_64f;
				img_t.convertTo(img_t_64f, CV_64FC1);
				img_tp1.convertTo(img_tp1_64f, CV_64FC1);
				bimg_t.convertTo(bimg_t_64f, CV_64FC1);
				double* arr_img_t, *arr_img_tp1, *arr_bimg_t, *arr_bimg_tp1 = 0, *arr_bimg_tp1_post_processed = 0;
				arr_img_t = ((double*)img_t_64f.data);
				arr_img_tp1 = ((double*)img_tp1_64f.data);
				arr_bimg_t = ((double*)bimg_t_64f.data);

				// compute vco, input & output data tpye is doulbe
				VesselCorrespondenceOptimization(arr_img_t, arr_img_tp1, arr_bimg_t, img_t.cols, img_t.rows, p, save_path, k + 1,
					&arr_bimg_tp1, &arr_bimg_tp1_post_processed, k + 1, (char*)save_path.data(), bVerbose);

				// for visualiazation
				cv::Mat bimg_tp1(img_t.rows, img_t.cols, CV_64FC1, arr_bimg_tp1),
					bimg_tp1_post_processed(img_t.rows, img_t.cols, CV_64FC1, arr_bimg_tp1_post_processed);

				bimg_tp1_post_processed.copyTo(seq_bimg_t);

				cv::Mat view;
				seq_bimg_t.convertTo(view, CV_8UC1);

				cv::imshow("veiw", view);
				cv::waitKey();

				// check running time
				clock_t end_time = clock();
				printf("Elapsed: %f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);


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
