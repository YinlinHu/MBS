
#include "MBS.h"
#include <algorithm>

MBS::MBS()
{
	_alpha = 0.1;
	_spSize = 400;

	_labels = NULL;

	_imgWidth = 0;
	_imgHeight = 0;
	_spCnt = 0;
}

MBS::~MBS()
{
	if (_labels){
		delete[] _labels;
		_labels = NULL;
	}
}

void MBS::SetAlpha(double alpha)
{
	_alpha = alpha;
}

void MBS::SetSuperpixelSize(int spSize)
{
	_spSize = spSize;
}

int MBS::SuperpixelSegmentation(cv::Mat& image)
{
	cv::Mat sImg; // smoothed image
	cv::boxFilter(image, sImg, -1, cv::Size(5, 5));

	int w = sImg.cols;
	int h = sImg.rows;
	_imgWidth = w;
	_imgHeight = h;

	// generate grid seeds
/*	int spSize = (w*h) / _spSize;*/
	int step = sqrt((double)_spSize) + 0.5;
	int seedsWidth = w / step;
	int seedsHeight = h / step;
	_seeds.create(seedsHeight, seedsWidth, CV_32SC2);

	int xoffset = (w - (seedsWidth - 1)*step) / 2;
	int yoffset = (h - (seedsHeight - 1)*step) / 2;
	int numV = seedsWidth * seedsHeight;
	for (int i = 0; i < seedsHeight; i++){
		cv::Vec2i* pi = _seeds.ptr<cv::Vec2i>(i);
		for (int j = 0; j < seedsWidth; j++){
			pi[j][0] = j*step + xoffset;
			pi[j][1] = i*step + yoffset;
		}
	}

	// refine seeds on pyramid
	float scaleRatio = 0.5;
	int minSpSize = 10;
	int pydLevels = 1;

	if (_spSize > minSpSize){
		pydLevels = int(0.5 * log((float)minSpSize / _spSize) / log(scaleRatio)) + 1;
	}

	cv::Mat* pyd = new cv::Mat[pydLevels];
	pyd[0] = sImg.clone();
	for (int i = 1; i < pydLevels; i++){
		cv::pyrDown(pyd[i - 1], pyd[i]);
	}
	//printf("%d levels\n", pyd.nlevels());

	// from top to bottom
	if (_labels){
		delete[] _labels;
	}
	_labels = new int[w*h];

	float* tmpSeedsX = new float[numV];
	float* tmpSeedsY = new float[numV];
	for (int i = 0; i < numV; i++){
		int sx = i % seedsWidth;
		int sy = i / seedsWidth;
		tmpSeedsX[i] = _seeds.at<cv::Vec2i>(sy, sx)[0] * pow(scaleRatio, pydLevels - 1);
		tmpSeedsY[i] = _seeds.at<cv::Vec2i>(sy, sx)[1] * pow(scaleRatio, pydLevels - 1);
	}

	for (int k = pydLevels - 1; k >= 0; k--){
		int size = _spSize*pow(scaleRatio, 2 * k);
		FastMBD(pyd[k], _labels, size, 1, 4, _alpha, tmpSeedsX, tmpSeedsY, numV);
		if (k > 0){
			for (int i = 0; i < numV; i++){
				tmpSeedsX[i] /= scaleRatio;
				tmpSeedsY[i] /= scaleRatio;
			}
		}
	}

	// remove outliers
	_spCnt = numV;
	MergeComponents(_labels, w, h);

	delete[] tmpSeedsX;
	delete[] tmpSeedsY;
	delete[] pyd;

	return _spCnt;
}

int* MBS::GetSuperpixelLabels()
{
	return _labels;
}

cv::Mat MBS::GetSeeds()
{
	return _seeds;
}

cv::Mat MBS::GetSuperpixelElements()
{
	// get the maximum number of elements
	int* elemCnts = new int[_spCnt];
	memset(elemCnts, 0, _spCnt*sizeof(int));
	for (int i = 0; i < _imgHeight; i++){
		for (int j = 0; j < _imgWidth; j++){
			int lab = _labels[i*_imgWidth + j];
			elemCnts[lab]++;
		}
	}
	int maxNum = -1;
	for (int i = 0; i < _spCnt; i++){
		if (elemCnts[i] > maxNum){
			maxNum = elemCnts[i];
		}
	}

	// get the elements of each superpixel
	cv::Mat elem(_spCnt, maxNum + 1, CV_32SC1);
	elem.setTo(-1);
	memset(elemCnts, 0, _spCnt*sizeof(int));
	for (int i = 0; i < _imgHeight; i++){
		for (int j = 0; j < _imgWidth; j++){
			int pixIdx = i*_imgWidth + j;
			int lab = _labels[pixIdx];
			elem.at<int>(lab, elemCnts[lab]) = pixIdx;
			elemCnts[lab]++;
		}
	}

	delete[] elemCnts;
	return elem;
}

// Fast Approximate MBD (Minimum Barrier Distance) Transform
void MBS::DistanceTransform_MBD(cv::Mat& image, float* seedsX, float* seedsY, int cnt, int* labels, float* dmap, float factor, int iter)
{
	int w = image.cols;
	int h = image.rows;
	int ch = image.channels();
	assert(ch == 3);

	cv::Mat U = image.clone();
	cv::Mat L = image.clone();

	memset(labels, 0xFF, w*h*sizeof(int)); // -1
	memset(dmap, 0x7F, w*h*sizeof(float)); // MAX_FLT
	//FImage tmp(w, h);
	for (int i = 0; i < cnt; i++){
		int x = seedsX[i] + 0.5;
		int y = seedsY[i] + 0.5;
		int cIdx = y*w + x;
		dmap[cIdx] = 0;
		labels[cIdx] = i;
		//
		//tmp[cIdx] = 1;
	}
	//tmp.imshow("center", 0);

	for (int n = 0; n < iter; n++)
	{
		int startX = 0, endX = w;
		int startY = 0, endY = h;
		int step = 1;
		int ox[2], oy[2]; //offset
		ox[0] = 0; oy[0] = -1;
		ox[1] = -1; oy[1] = 0;
		if (n % 2 == 1){
			startX = w - 1;	endX = -1;
			startY = h - 1;	endY = -1;
			step = -1;
			ox[0] = 0; oy[0] = 1;
			ox[1] = 1; oy[1] = 0;
		}

		for (int i = startY; i != endY; i += step){
			cv::Vec3b* pi = image.ptr<cv::Vec3b>(i);
			cv::Vec3b* pu = U.ptr<cv::Vec3b>(i);
			cv::Vec3b* pl = L.ptr<cv::Vec3b>(i);

			for (int j = startX; j != endX; j += step){
				int idx = i*w + j;
				for (int k = 0; k <= 1; k++){
					int candix = j + ox[k];
					int candiy = i + oy[k];
					if (candix >= 0 && candix < w && candiy >= 0 && candiy < h){
						int canIdx = candiy*w + candix;
						int sIdx = labels[canIdx];
						if (sIdx >= 0){
							cv::Vec3b cd1 = U.at<cv::Vec3b>(candiy, candix); // candidates
							cv::Vec3b cd2 = L.at<cv::Vec3b>(candiy, candix);

							uchar maxCost[3], minCost[3];
							maxCost[0] = std::max(cd1[0], pi[j][0]);
							maxCost[1] = std::max(cd1[1], pi[j][1]);
							maxCost[2] = std::max(cd1[2], pi[j][2]);
							minCost[0] = std::min(cd2[0], pi[j][0]);
							minCost[1] = std::min(cd2[1], pi[j][1]);
							minCost[2] = std::min(cd2[2], pi[j][2]);

							int colorDis[3];
							colorDis[0] = maxCost[0] - minCost[0];
							colorDis[1] = maxCost[1] - minCost[1];
							colorDis[2] = maxCost[2] - minCost[2];
							float cDis = std::max(std::max(colorDis[0], colorDis[1]), colorDis[2]) / 255.;

							float sDis2 = (seedsX[sIdx] - j)*(seedsX[sIdx] - j) + (seedsY[sIdx] - i)*(seedsY[sIdx] - i);
							float dis = cDis*cDis + factor*sDis2;

							if (dis < dmap[idx]){
								labels[idx] = sIdx;
								dmap[idx] = dis;
								memcpy(pu + j, maxCost, 3 * sizeof(uchar));
								memcpy(pl + j, minCost, 3 * sizeof(uchar));
							}
						}
					}
				}
			}
		}
	}
}

int MBS::FastMBD(cv::Mat& img, int* labels, int spSize, int outIter, int inIter, float alpha, float* seedsX, float* seedsY, int cnt)
{
	int w = img.cols;
	int h = img.rows;

	float* cx = new float[cnt];
	float* cy = new float[cnt];
	float* sumx = new float[cnt];
	float* sumy = new float[cnt];
	float* sumn = new float[cnt];

	memcpy(cx, seedsX, cnt*sizeof(float));
	memcpy(cy, seedsY, cnt*sizeof(float));

	float* dmap = new float[w*h];

	for (int k = 0; k < outIter; k++)
	{
		DistanceTransform_MBD(img, cx, cy, cnt, labels, dmap, alpha*alpha / spSize, inIter);

		memset(sumx, 0, cnt*sizeof(float));
		memset(sumy, 0, cnt*sizeof(float));
		memset(sumn, 0, cnt*sizeof(float));
		for (int i = 0; i < h; i++){
			for (int j = 0; j < w; j++){
				int idx = i*w + j;
				float wt = dmap[idx];
				int sIdx = labels[idx];
				if (sIdx >= 0){
					sumx[sIdx] += (wt*j);
					sumy[sIdx] += (wt*i);
					sumn[sIdx] += wt;
				}
			}
		}
		for (int i = 0; i < cnt; i++){
			float wt = sumn[i];
			float oldx = cx[i];
			float oldy = cy[i];
			if (wt > 0){
				cx[i] = sumx[i] / wt;
				cy[i] = sumy[i] / wt;
			}
		}
	}

	// update clustering centers
	memcpy(seedsX, cx, cnt*sizeof(float));
	memcpy(seedsY, cy, cnt*sizeof(float));

	delete[] cx;
	delete[] cy;
	delete[] sumx;
	delete[] sumy;
	delete[] sumn;

	delete[] dmap;
	return 0;
}

// Modified from SLIC
// ===========================================================================
// /		1. finding an adjacent label for each new component at the start
// /		2. if a certain component is too small, assigning the previously found
// /		    adjacent label to this component.
// /        3. if the component is the last component for some label, reserve it.
// /            guarantee that no label will disappear after merge,
// /            this is important for some applications (Yinlin)
// /		4. after merge, each label have one and only one component
// ===========================================================================
void MBS::MergeComponents(int* ioLabels, int w, int h)
{
	const int dx4[4] = { -1, 0, 1, 0 };
	const int dy4[4] = { 0, -1, 0, 1 };

	// get the raw element count for each superpixel
	int* elemCnts = new int[_spCnt];
	memset(elemCnts, 0, _spCnt*sizeof(int));
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){
			int lab = ioLabels[i*w + j];
			elemCnts[lab]++;
		}
	}

	int* visited = new int[w*h];
	memset(visited, 0, w*h*sizeof(int));

	int* xvec = new int[w*h];
	int* yvec = new int[w*h];

	int currLabel = -1; //current superpixel label
	int adjLabel = -1; //adjacent superpixel label

	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){
			int pixIdx = i*w + j;
			if (!visited[pixIdx]){
				currLabel = ioLabels[pixIdx];
				//--------------------
				// Start a new segment
				//--------------------
				xvec[0] = j;
				yvec[0] = i;
				//-------------------------------------------------------
				// Quickly find an adjacent label for use later if needed
				//-------------------------------------------------------
				for (int n = 0; n < 4; n++){
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if ((x >= 0 && x < w) && (y >= 0 && y < h)){
						int nbIdx = y*w + x;
						if (visited[nbIdx]){
							adjLabel = ioLabels[nbIdx];
						}
					}
				}

				// get current component size
				int count = 1;
				visited[pixIdx] = 1;
				for (int c = 0; c < count; c++){
					for (int n = 0; n < 4; n++){
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];
						if ((x >= 0 && x < w) && (y >= 0 && y < h)){
							int nbIdx = y*w + x;
							if (!visited[nbIdx] 
								&& currLabel == ioLabels[nbIdx]){
								xvec[count] = x;
								yvec[count] = y;
								visited[nbIdx] = 1;
								count++;
							}
						}
					}
				}
				//-------------------------------------------------------
				// If segment size is less then a limit, assign an
				// adjacent label found before. Furthermore, make sure that
				// no label will disappear after merge
				//-------------------------------------------------------
				if (count <= (_spSize >> 3)){
					int remainCnt = elemCnts[currLabel] - count;
					if (remainCnt > 0){
						for (int c = 0; c < count; c++){
							int ind = yvec[c] * w + xvec[c];
							ioLabels[ind] = adjLabel;
						}
						elemCnts[currLabel] = remainCnt;
					}
				}
			}
			//
		}
	}
	delete[] xvec;
	delete[] yvec;

	delete[] visited;
	delete[] elemCnts;
}

cv::Mat MBS::Visualization(cv::Mat& image)
{
	int w = image.cols;
	int h = image.rows;

	// generate boundaries
	int* bound = new int[w*h];
	memset(bound, 0, w*h*sizeof(int));
	int nbOffset[4][2] = { { 0, -1 }, { 0, 1 }, { 1, 0 }, { -1, 0 } };
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){
			int diffCnt = 0;
			for (int k = 0; k < 4; k++){
				int nbi = i + nbOffset[k][0];
				int nbj = j + nbOffset[k][1];
				if (nbi >= 0 && nbi < h&&nbj >= 0 && nbj < w){
					if (_labels[i*w + j] != _labels[nbi*w + nbj]){
						diffCnt++;
					}
				}
			}
			if (diffCnt > 0){
				bound[i*w + j] = 1;
			}
		}
	}

	// draw green boundaries
	cv::Mat visImg = image.clone();
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){
			if (bound[i*w + j]){
				visImg.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 0);
			}
		}
	}

	delete[] bound;
	return visImg;
}

cv::Mat MBS::Visualization()
{
	cv::Mat visImg(_imgHeight, _imgWidth, CV_8UC3);
	cv::Mat elem = GetSuperpixelElements();
	uchar color[3];
	srand(0);
	for (int i = 0; i < _spCnt; i++){
		// generate random color for each superpixel
		color[0] = rand() % 200 + 50;
		color[1] = rand() % 200 + 50;
		color[2] = rand() % 200 + 50;
		for (int j = 0; j < elem.cols; j++){
			int pixIdx = elem.at<int>(i, j);
			if (pixIdx < 0){
				break;
			}
			int x = pixIdx % _imgWidth;
			int y = pixIdx / _imgWidth;
			cv::Vec3b* pRow = visImg.ptr<cv::Vec3b>(y);
			memcpy(pRow + x, color, 3 * sizeof(uchar));
		}
	}
	return visImg;
}
