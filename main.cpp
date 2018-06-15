
#include "MBS.h"

// #define DEBUG_YL

int main(int argc, char** argv)
{
#ifdef DEBUG_YL
	cv::Mat img = cv::imread("F:/KITTI/data_scene_flow/training/image_2/000011_10.png");
	int spSize = 400;
	double alpha = 0.1;
#else
	if (argc < 2){
		printf("USAGE: MBS.exe imageName [SuperPixelSize Alpha]\n");
		return -1;
	}
	cv::Mat img = cv::imread(argv[1]);
	int spSize = 400;
	double alpha = 0.1;
	if (argc >= 3){
		spSize = atoi(argv[2]);
	}
	if (argc >= 4){
		alpha = atof(argv[3]);
	}
#endif

	MBS mbs;
	mbs.SetSuperpixelSize(spSize);
	mbs.SetAlpha(alpha);
	
	int spCnt = mbs.SuperpixelSegmentation(img);

	// Visualization
	// cv::Mat spVisual = mbs.Visualization();
    cv::Mat spVisual = mbs.Visualization(img);
#ifdef DEBUG_YL
	printf("%d\n", spCnt);
	cv::imshow("MBS out", spVisual);
	cv::waitKey(0);
#else
	cv::imwrite("MBS_out.png", spVisual);
#endif

	return 0;
}
