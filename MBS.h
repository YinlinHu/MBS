// [10/26/2017 Yinlin]
// Implementation of "Minimum Barrier Superpixel Segmentation"
// huyinlin@gmail.com

#ifndef _MBS_H_
#define _MBS_H_

#include "opencv2/opencv.hpp"

// Minimum Barrier Superpixel Segmentation
class MBS
{
public:
	MBS();
	~MBS();

	/************************************************************************/
	/* parameter setting functions                                          */
	/************************************************************************/
	// control compactness, small alpha leads to more compact superpixels,
	// [0-1] is fine, the default is 0.1 which is suitable for most cases
	void SetAlpha(double alpha);
	
	// set the average size of superpixels
	void SetSuperpixelSize(int spSize);

	/************************************************************************/
	/* do the over-segmentation                                             */
	/************************************************************************/
	int SuperpixelSegmentation(cv::Mat& image);

	/************************************************************************/
	/* utility functions                                                    */
	/************************************************************************/
	int* GetSuperpixelLabels();
	cv::Mat GetSeeds();
	cv::Mat GetSuperpixelElements();
	
	cv::Mat Visualization();
	cv::Mat Visualization(cv::Mat& image);

private:
	void DistanceTransform_MBD(cv::Mat& image, float* seedsX, float* seedsY, 
		int cnt, int* labels, float* dmap, float factor, int iter = 4);
	int FastMBD(cv::Mat& img, int* labels, int spSize, int outIter, int inIter, 
		float alpha, float* seedsX, float* seedsY, int cnt);
	void MergeComponents(int* ioLabels, int w, int h);
	double _alpha;
	int _spSize;

	int* _labels;

	cv::Mat _seeds;

	int _imgWidth;
	int _imgHeight;
	int _spCnt;

};

#endif // _MBS_H_
