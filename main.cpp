
#include "MBS.h"

int main(int argc, char** argv)
{
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

	int* labels = new int[img.cols * img.rows];

	MBS mbs;
	mbs.SetSuperpixelSize(spSize);
	mbs.SetAlpha(alpha);
	
	int spCnt = mbs.SuperpixelSegmentation(img, labels);
	// printf("%d\n", spCnt);

	// Visualization
	cv::Mat spVisual = MBS::SuperpixelVisualization(img, labels);
	cv::imwrite("MBS_out.png", spVisual);

	delete[] labels;
	return 0;
}
