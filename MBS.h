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
	// [0-1] is fine, the default is 0.1 which is suitable for most cases.
	void SetAlpha(double alpha);
	
	// set the average size of superpixels, 
	// please use number larger than 20.
	void SetSuperpixelSize(int spSize);

	/************************************************************************/
	/* do the over-segmentation                                             */
	/************************************************************************/
	int SuperpixelSegmentation(cv::Mat& image, int* outLabels);

	/************************************************************************/
	/* utility functions                                                    */
	/************************************************************************/
	static cv::Mat SuperpixelVisualization(cv::Mat& image, int* inLabels);

private:
	void DistanceTransform_MBD(cv::Mat& image, float* seedsX, float* seedsY, 
		int cnt, int* labels, float* dmap, float factor, int iter = 4);
	int FastMBD(cv::Mat& img, int* labels, int spSize, int outIter, int inIter, 
		float alpha, float* seedsX, float* seedsY, int cnt);
	int RemoveOutliers(int* inLabels, int* outLabels, int w, int h, int spSize);

	double _alpha;
	int _spSize;
};

#endif // _MBS_H_
