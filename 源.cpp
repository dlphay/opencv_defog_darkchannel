// 基于高效融合的去雾
// 第一步：输入带雾的图像
// 第二步：基于高斯暗通道的大气光线估计
// 第三步：基于CLAHE的精细细节描述
// 第四步：基于融合的转换估计
// 第五步：输出恢复的图像

#include "stdafx.h"
#include <Windows.h>                
#include<iostream>
#include<time.h>

// opencv
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

// 全局变量
unsigned char* pBmpBuf;
int bmpWidth;
int bmpHeight;
RGBQUAD* pColorTable;
int biBitCount;

IplImage* getMinIcy(IplImage* dark, int w)
{
	IplImage* Icy = cvCreateImage(cvGetSize(dark), IPL_DEPTH_8U, 1);
	int hei = dark->height;
	int wid = dark->width;
	int hw = hei / w;
	int ww = wid / w;
	for (int i = w; i < (hw - 1)*w; i += w)
	{
		for (int j = w; j < (ww - 1)*w; j += w)
		{
			double p = cvGetReal2D(dark, i - 1, j - 1);  //得到窗口最右下角的一个像素点  
			//得到窗口最小的像素值  
			for (int ii = i - w; ii < i; ii++)
			{
				for (int jj = j - w; jj < j; jj++)
				{
					double newp = cvGetReal2D(dark, ii, jj);
					if (newp < p)
					{
						p = newp;
					}
				}
			}
			//设置Icy的值  
			for (int ii = i - w; ii < i; ii++)
			{
				for (int jj = j - w; jj < j; jj++)
				{
					cvSetReal2D(Icy, ii, jj, p);
				}
			}

		}
	}

	//处理最右边一列  不包含最下一个子块  
	for (int i = w; i < (hw - 1)*w; i += w)
	{
		double p = cvGetReal2D(dark, i - 1, wid - 1);  //得到窗口最右下角的一个像素点  
		for (int ii = i - w; ii < i; ii++)
		{

			for (int j = (ww - 1)*w; j < wid; j++)
			{
				//得到窗口最小的像素值  
				double newp = cvGetReal2D(dark, ii, j);
				if (newp < p)
				{
					p = newp;
				}
			}
		}

		//设置Icy的值  
		for (int ii = i - w; ii < i; ii++)
		{

			for (int j = (ww - 1)*w; j < wid; j++)
			{
				cvSetReal2D(Icy, ii, j, p);
			}
		}
	}


	//处理最下一行 不包含最后一个子块  
	for (int j = w; j < (ww - 1)*w; j += w)
	{
		double p = cvGetReal2D(dark, hei - 1, j);  //得到窗口最右下角的一个像素点  
		for (int i = (hw - 1)*w; i < hei; i++)
		{
			for (int jj = j - w; jj < j; jj++)
			{
				//得到窗口最小的像素值  
				double newp = cvGetReal2D(dark, i, jj);
				if (newp < p)
				{
					p = newp;
				}
			}
		}

		//设置Icy的值  
		for (int i = (hw - 1)*w; i < hei; i++)
		{

			for (int jj = j - w; jj < j; jj++)
			{
				cvSetReal2D(Icy, i, jj, p);
			}
		}

	}

	//处理最右下角的一个子块  
	double p = cvGetReal2D(dark, hei - 1, wid - 1);  //得到窗口最右下角的一个像素点  
	for (int i = (hw - 1)*w; i < hei; i++)
	{
		for (int j = (ww - 1)*w; j < wid; j++)
		{
			//得到窗口最小的像素值  
			double newp = cvGetReal2D(dark, i, j);
			if (newp < p)
			{
				p = newp;
			}

		}
	}
	for (int i = (hw - 1)*w; i < hei; i++)
	{
		for (int j = (ww - 1)*w; j < wid; j++)
		{
			cvSetReal2D(Icy, i, j, p);

		}
	}

	return Icy;
}

unsigned char* readBmp(char* bmpName)

{
	FILE* fp = fopen(bmpName, "rb"); //以二进制读的方式打开指定的图像文件
	if (fp == 0) return 0;

	fseek(fp, sizeof(BITMAPFILEHEADER), 0);
	BITMAPINFOHEADER infoHead;
	fread(&infoHead, sizeof(BITMAPINFOHEADER), 1, fp);

	bmpWidth = infoHead.biWidth;
	bmpHeight = infoHead.biHeight;
	biBitCount = infoHead.biBitCount;

	//strick
	int lineByte = (bmpWidth * biBitCount / 8 + 3) / 4 * 4;


	if (biBitCount == 8)
	{
		pColorTable = new RGBQUAD[256];
		fread(pColorTable, sizeof(RGBQUAD), 256, fp);

	}
	pBmpBuf = new unsigned char[lineByte * bmpHeight];

	fread(pBmpBuf, 1, lineByte * bmpHeight, fp);
	fclose(fp);
	return pBmpBuf;
}

int LineInterpolate(unsigned char* pImage, int nSizeX, int nSizeY, int* pulMapLU, int* pulMapRU, int* pulMapLB, int* pulMapRB)
{
	const int     uiIncr = 200 - nSizeX;                         /* Pointer increment after processing row */
	unsigned char GreyValue = 0;
	int           uiNum = nSizeX*nSizeY;
	int           x, y, nInvX, nInvY, uiShift = 0;
	for (y = 0, nInvY = nSizeY; y<nSizeY; y++, nInvY--, pImage += uiIncr)
	{
		for (x = 0, nInvX = 0; x<0; x++, nInvX--)
		{
			//GreyValue = m_aLUT[*pImage];           /* get histogram bin value */
			*pImage++ = (unsigned char)((nInvY*(nInvX*pulMapLU[GreyValue] + x*pulMapRU[GreyValue])
				+ y * (nInvX*pulMapLB[GreyValue] + x*pulMapRB[GreyValue])) / uiNum);
		}
	}
	return 0;
}

IplImage* getDehazedImage(IplImage* hazeImage, IplImage* guidedt, double Ac)
{
	IplImage* dehazedImage = cvCreateImage(cvGetSize(hazeImage), IPL_DEPTH_8U, 3);
	IplImage* r = cvCreateImage(cvGetSize(hazeImage), IPL_DEPTH_8U, 1);
	IplImage* g = cvCreateImage(cvGetSize(hazeImage), IPL_DEPTH_8U, 1);
	IplImage* b = cvCreateImage(cvGetSize(hazeImage), IPL_DEPTH_8U, 1);

	cvSplit(hazeImage, b, g, r, NULL);

	IplImage* dehaze_r = cvCreateImage(cvGetSize(hazeImage), IPL_DEPTH_8U, 1);
	IplImage* dehaze_g = cvCreateImage(cvGetSize(hazeImage), IPL_DEPTH_8U, 1);
	IplImage* dehaze_b = cvCreateImage(cvGetSize(hazeImage), IPL_DEPTH_8U, 1);

	for (int i = 0; i < r->height; i++)
	{
		for (int j = 0; j < r->width; j++)
		{
			double tempt = cvGetReal2D(guidedt, i, j);
			if (tempt / 255 < 0.1)
			{
				tempt = 25.5;
			}

			double I_r = cvGetReal2D(r, i, j);
			double de_r = 255 * (I_r - Ac) / tempt + Ac;
			cvSetReal2D(dehaze_r, i, j, de_r);

			double I_g = cvGetReal2D(g, i, j);
			double de_g = 255 * (I_g - Ac) / tempt + Ac;
			cvSetReal2D(dehaze_g, i, j, de_g);

			double I_b = cvGetReal2D(b, i, j);
			double de_b = 255 * (I_b - Ac) / tempt + Ac;
			cvSetReal2D(dehaze_b, i, j, de_b);

		}
	}

	cvMerge(dehaze_b, dehaze_g, dehaze_r, 0, dehazedImage);

	return dehazedImage;

}

void CopyOf(IplImage* img, int desired_color)
{
	if (desired_color)
	{
		int color = desired_color;
		CvSize size = cvGetSize(img);

		if (color < 0)
			color = img->nChannels > 1;

		if (img)
		{
			cvConvertImage(img, img, 0);
		}
	}
}

bool LoadRect(const char* filename,
	int desired_color, CvRect r)
{
	if (r.width < 0 || r.height < 0) return false;

	IplImage* img = cvLoadImage(filename, desired_color);
	if (!img)
		return false;

	if (r.width == 0 || r.height == 0)
	{
		r.width = img->width;
		r.height = img->height;
		r.x = r.y = 0;
	}

	if (r.x > img->width || r.y > img->height ||
		r.x + r.width < 0 || r.y + r.height < 0)
	{
		cvReleaseImage(&img);
		return false;
	}

	/* truncate r to source image */
	if (r.x < 0)
	{
		r.width += r.x;
		r.x = 0;
	}
	if (r.y < 0)
	{
		r.height += r.y;
		r.y = 0;
	}

	if (r.x + r.width > img->width)
		r.width = img->width - r.x;

	if (r.y + r.height > img->height)
		r.height = img->height - r.y;

	cvSetImageROI(img, r);
	CopyOf(img, desired_color);

	cvReleaseImage(&img);
	return true;
}


BYTE *RmwRead8BitBmpFile2Img(const char * filename, int *width, int *height)
{
	FILE *binFile;
	BYTE *pImg = NULL;
	BITMAPFILEHEADER fileHeader;
	BITMAPINFOHEADER bmpHeader;
	BOOL isRead = TRUE;
	int linenum, ex;

	if ((binFile = fopen(filename, "rb")) == NULL) return NULL;

	if (fread((void *)&fileHeader, 1, sizeof(fileHeader), binFile) != sizeof(fileHeader)) isRead = FALSE;
	if (fread((void *)&bmpHeader, 1, sizeof(bmpHeader), binFile) != sizeof(bmpHeader)) isRead = FALSE;

	if (isRead == FALSE || fileHeader.bfOffBits<sizeof(fileHeader) + sizeof(bmpHeader)) {
		fclose(binFile);
		return NULL;
	}

	*width = bmpHeader.biWidth;
	*height = bmpHeader.biHeight;
	linenum = (*width * 1 + 3) / 4 * 4;
	ex = linenum - *width * 1;

	fseek(binFile, fileHeader.bfOffBits, SEEK_SET);
	pImg = new BYTE[(*width)*(*height)];
	if (pImg != NULL) {
		for (int i = 0; i<*height; i++) {
			int r = fread(pImg + (*height - i - 1)*(*width), sizeof(BYTE), *width, binFile);
			if (r != *width) {
				delete pImg;
				fclose(binFile);
				return NULL;
			}
			fseek(binFile, ex, SEEK_CUR);
		}
	}
	fclose(binFile);
	return pImg;
}

void delay_sec(int sec)//  
{
	time_t start_time, cur_time;
	time(&start_time);
	do
	{
		time(&cur_time);
	} while ((cur_time - start_time) < sec);
}


// 修剪直方图  
void ClipHistogram(int* pHistogram, int clipThreshold)
{
	int binExcess = 0, totalExcess = 0, avgBinIncr = 0, upperLimit = 0;
	// 累积超出阈值部分  
	for (int i = 0; i < 256; i++)
	{
		binExcess = pHistogram[i] - clipThreshold;
		if (binExcess > 0)
		{
			totalExcess += binExcess;
		}
	}

	avgBinIncr = totalExcess / 256;
	upperLimit = clipThreshold - avgBinIncr;

	// 修剪直方图并重新分配数值  
	for (int i = 0; i < 256; i++)
	{
		if (pHistogram[i] > clipThreshold)
		{
			pHistogram[i] = clipThreshold;
		}
		else
		{
			if (pHistogram[i] > upperLimit)
			{
				totalExcess -= (clipThreshold - pHistogram[i]);
				pHistogram[i] = clipThreshold;
			}
			else
			{
				totalExcess -= avgBinIncr;
				pHistogram[i] += avgBinIncr;
			}
		}
	}

	// 剩余部分再次分配  
	int *pCurBin = pHistogram;
	int *pStartBin = pHistogram;
	int *pEndBin = pHistogram + 255;
	while (totalExcess > 0 && pStartBin < pEndBin)
	{
		int stepSize = 256 / totalExcess;
		if (stepSize < 1)
		{
			stepSize = 1;
		}

		for (pCurBin = pStartBin; pCurBin < pEndBin && totalExcess > 0; pCurBin += stepSize)
		{
			if (*pCurBin < clipThreshold)
			{
				(*pCurBin)++;
				totalExcess--;
			}
		}
		pStartBin++;
	}
}

int main()
{
	int is_MedianBlurHaze = 1;
	int is_display = 1; // 1 xianshi 0 buxianshi 
	typedef int(__stdcall *Dehaze)(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride,
		int BlockSize, int GuideRadius, int MaxAtom, float Omega, float T0, float Gamma);

	Dehaze pfFuncInDll = NULL;
	HINSTANCE hinst = LoadLibraryA("ImageMaster.dll");

	unsigned char* C = readBmp("test.png");
		//unsigned char* C = readBmp("test.png");
	//unsigned char* C = readBmp("forest.jpg");

	// get Info
	int image_width = 0;
	int image_height = 0;
	int image_width2 = 0;
	int image_height2 = 0;
	int image_width3 = 0;
	int image_height3 = 0;
	//int image_lineByte = 0;
	//Mat inputImagemat = imread("test.png", CV_LOAD_IMAGE_UNCHANGED);
	//Mat outputImagemat = imread("test.png", CV_LOAD_IMAGE_UNCHANGED);
	printf("*********************** 第一步：输入带雾的图像 **************************\n");
	Mat inputImagemat = imread("forest.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat outputImagemat = imread("forest.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat inputImagemat2 = imread("test2.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat outputImagemat2 = imread("test2.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat inputImagemat3 = imread("demo.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat outputImagemat3 = imread("demo.jpg", CV_LOAD_IMAGE_UNCHANGED);
	//Mat inputImagemat2 = imread("test.jpg", CV_LOAD_IMAGE_UNCHANGED);
	//Mat outputImagemat2 = imread("test.jpg", CV_LOAD_IMAGE_UNCHANGED);

	//RdWtIm rdWtIm;
	//unsigned char* imData = rdWtIm.Read8bitbmp("lena.bmp", &width, &height);
	image_width = inputImagemat.cols;
	image_height = inputImagemat.rows;
	auto image_flags = inputImagemat.flags;
	auto image_dims = inputImagemat.dims;

	image_width2 = inputImagemat2.cols;
	image_height2 = inputImagemat2.rows;
	auto image_flags2 = inputImagemat2.flags;
	auto image_dims2 = inputImagemat2.dims;

	image_width3 = inputImagemat3.cols;
	image_height3 = inputImagemat3.rows;
	auto image_flags3 = inputImagemat3.flags;
	auto image_dims3 = inputImagemat3.dims;

	int image_depth = 3; // 默认是4通道的


	// 最终保存图片的缓存区域、
	unsigned char* D = (unsigned char *)malloc(sizeof(unsigned char) * (image_width * image_height * image_depth));
	unsigned char* A = (unsigned char *)malloc(sizeof(unsigned char) * (image_width * image_height * image_depth));

	unsigned char* E = (unsigned char *)malloc(sizeof(unsigned char) * (image_width2 * image_height2 * image_depth));
	unsigned char* F = (unsigned char *)malloc(sizeof(unsigned char) * (image_width2 * image_height2 * image_depth));

	unsigned char* MedianBlurHaze = (unsigned char *)malloc(sizeof(unsigned char) * (image_width * image_height * image_depth));
	//Stride的计算公式：Stride = 每像素占用的字节数(也就是像素位数/8) * Width;如果 Stride 不是 4 的倍数, 那么 Stride = Stride + (4 - Stride mod 4)
	A = inputImagemat.data;
	MedianBlurHaze = inputImagemat.data;
	Mat src = inputImagemat;
	Mat dst = inputImagemat;

	E = inputImagemat2.data;
	MedianBlurHaze = inputImagemat2.data;
	Mat src2 = inputImagemat2;
	Mat dst2 = inputImagemat2;

	// 去雾相关变量
	unsigned char *Src = &A[0];
	//unsigned char *Src = &C[0];
	unsigned char *Dest = &D[0];
	int Width = image_width;
	int Height = image_height;
	int Stride = 0;
	if ((image_width * image_depth) % 4 == 0)
	{
		Stride = image_width * image_depth;
	}
	else
	{
		Stride = image_width * image_depth + (4 - image_width * image_depth % 4);
	}

	// 去雾相关变量
	unsigned char *Src2 = &E[0];
	//unsigned char *Src = &C[0];
	unsigned char *Dest2 = &D[0];
	int Width2 = image_width;
	int Height2 = image_height;
	int Stride2 = 0;
	if ((Width2 * image_depth) % 4 == 0)
	{
		Stride2 = Width2 * image_depth;
	}
	else
	{
		Stride2 = Width2 * image_depth + (4 - Width2 * image_depth % 4);
	}

	int BlockSize = 4;
	int GuideRadius = 5;
	int MaxAtom = 220;
	float Omega = 0.9f;
	float T0 = 0.1f;
	float Gamma = 0.9f;

	unsigned char *Src3 = &E[0];
	//unsigned char *Src = &C[0];
	unsigned char *Dest3 = &D[0];
	int Width3 = image_width;
	int Height3 = image_height;
	int Stride3 = 0;
	if ((Width3 * image_depth) % 4 == 0)
	{
		Stride3 = Width3 * image_depth;
	}
	else
	{
		Stride3 = Width3 * image_depth + (4 - Width3 * image_depth % 4);
	}
	printf("*************  Input image Info  **************\n");
	printf("Input image Width: %d \n", Width);
	printf("Input image Height: %d \n", Height);
	printf("Input image Depth: %d \n", image_depth);
	printf("Input image Stride: %d \n", Stride);
	printf("Input image BlockSize: %d \n", BlockSize);
	printf("Input image GuideRadius: %d \n", GuideRadius);
	printf("Input image MaxAtom: %d \n", MaxAtom);
	printf("********************  END  *********************\n");

	// IplImage inputImageiplimage = inputImagemat;
	//IplImage  -- >  unsigned char*
	// C = (unsigned char*)inputImageiplimage.imageData;

	//创建一个名字为MyWindow的窗口
	//// 图片的显示
	//imshow("带雾图像", inputImagemat);
	//waitKey(0);

	// 第二步：基于高斯暗通道的大气光线估计
	printf("*********************** 第一步：END **************************\n");
	printf("******* 第二步：基于高斯暗通道的大气光线估计开始执行 *********\n");
	if (is_MedianBlurHaze)
	{
		cout << "Gaussian dark channel function load SUCCESS !!!" << endl;
		int  X, Y, Diff, Min, F;
		unsigned char* Pointer, *DarkP, *FilterP, *FilterPC;
		unsigned char * DarkChannel = (unsigned char*)malloc(Width * Height);
		unsigned char * Filter = (unsigned char*)malloc(Width * Height);
		unsigned char * FilterClone = (unsigned char*)malloc(Width * Height);

		for (Y = 0; Y < Height; Y++)
		{
			Pointer = MedianBlurHaze + Y * Stride;
			DarkP = DarkChannel + Y * Width;  // 由实际图像计算得到的图像暗通道     
			for (X = 0; X < Width; X++)
			{
				Min = *Pointer;
				if (Min > *(Pointer + 1)) Min = *(Pointer + 1);
				if (Min > *(Pointer + 2)) Min = *(Pointer + 2);
				*DarkP = (unsigned char)Min;
				DarkP++;
				Pointer += 3;
			}
		}
		memcpy(Filter, DarkChannel, Width * Height);   // 求全局大气光A时会破坏DarkChannel中的数据

		//MinValue(DarkChannel, Width, Height, Width, DarkRadius);   // 求取暗通道值

		// 利用暗通道来估算全局大气光值A
		int Sum, Value, Threshold = 0;
		int SumR = 0, SumG = 0, SumB = 0, AtomR, AtomB, AtomG, Amount = 0;
		int* Histgram = (int*)calloc(256, sizeof(int));
		for (Y = 0; Y < Width * Height; Y++) Histgram[DarkChannel[Y]]++;
		for (Y = 255, Sum = 0; Y >= 0; Y--)
		{
			Sum += Histgram[Y];
			if (Sum > Height * Width * 0.01)
			{
				Threshold = Y;   // 选取暗通道值中前1%最亮的像素区域为候选点
				break;
			}
		}
		AtomB = 0; AtomG = 0; AtomR = 0;
		for (Y = 0, DarkP = DarkChannel; Y < Height; Y++)
		{
			Pointer = MedianBlurHaze + Y * Stride;
			for (X = 0; X < Width-10; X++)
			{
				if (*DarkP >= Threshold)   //    在原图中选择满足候选点的位置的像素作为计算全局大气光A的信息                        
				{
					SumB += *Pointer;
					SumG += *(Pointer + 1);
					SumR += *(Pointer + 2);
					Amount++;
				}
				Pointer += 3;
				DarkP++;
			}
		}
		AtomB = SumB / Amount;
		AtomG = SumG / Amount;
		AtomR = SumR / Amount;

		memcpy(DarkChannel, Filter, Width * Height);   // 恢复DarkChannel中的数据
		//MedianBlur(Filter, Width, Height, Width, MedianRadius, 50); 
		memcpy(FilterClone, Filter, Width * Height);

		DarkP = DarkChannel;
		FilterP = Filter;
		for (Y = 0; Y < Height * Width; Y++)    //利用一重循环来计算提高速度
		{
			Diff = *DarkP - *FilterP;    //通过对|DarkP －FilterP |执行中值滤波来估计的局部标准差，这样可以保证标准差估计的鲁棒性
			if (Diff < 0) Diff = -Diff;
			*FilterP = (unsigned char)Diff;
			DarkP++;
			FilterP++;
		}
		//MedianBlur(Filter, Width, Height, Width, MedianRadius, 50);

		FilterPC = FilterClone;
		FilterP = Filter;
		for (Y = 0; Y < Height * Width; Y++)
		{
			Diff = *FilterPC - *FilterP;                    // 步骤2：然后考虑到有较好对比度的纹理区域可能没有雾， 这部分区域就不需要做去雾处理
			if (Diff < 0) Diff = 0;                            // 这里可以这样做是因为在最后有个max(....,0)的过程，
			*FilterP = (unsigned char)Diff;
			FilterPC++;
			FilterP++;
		}

		DarkP = DarkChannel;
		FilterP = Filter;

		for (Y = 0; Y < Height * Width; Y++)
		{
			Min = *FilterP * 2 / 100;
			if (*DarkP > Min)
				*FilterP = Min;                                // 获得满足约束条件的大气光幕
			else
				*FilterP = *DarkP;
			DarkP++;
			FilterP++;
		}
		printf("image Omega: %f \n", Omega);
		printf("image T0: %f \n", T0);
		printf("image Gamma: %f \n", Gamma);
		FilterP = Filter;
		for (Y = 0; Y < Height; Y++)
		{
			Pointer = MedianBlurHaze + Y * Stride;
			for (X = 0; X < Width; X++)
			{
				F = *FilterP++;
				if (AtomB != F)
					Value = AtomB *(*Pointer - F) / (AtomB - F);
				else
					Value = *Pointer;
				//*Pointer++ = Clamp(Value);
				if (AtomG != F)
					Value = AtomG * (*Pointer - F) / (AtomG - F);
				else
					Value = *Pointer;
				//*Pointer++ = Clamp(Value);
				if (AtomR != F)
					Value = AtomR *(*Pointer - F) / (AtomR - F);
				else
					Value = *Pointer;
				//*Pointer++ = Clamp(Value);
			}
		}
	}


	printf("*********************** 第二步：END **************************\n");
	printf("**************** 第三步：基于CLAHE的精细细节描述 **************\n");
	 //第三步：基于CLAHE的精细细节描述
	{
		cout << "CLAHE function load SUCCESS !!!" << endl;
		//at访问像素点  
		for (int i = 1; i<src.rows; ++i)
			for (int j = 1; j < src.cols; ++j) {
				if ((i - 1 >= 0) && (j - 1) >= 0 && (i + 1)<src.rows && (j + 1)<src.cols) {
					dst.at<Vec3b>(i, j)[0] = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i - 1, j - 1)[0] + src.at<Vec3b>(i - 1, j)[0] + src.at<Vec3b>(i, j - 1)[0] +
						src.at<Vec3b>(i - 1, j + 1)[0] + src.at<Vec3b>(i + 1, j - 1)[0] + src.at<Vec3b>(i + 1, j + 1)[0] + src.at<Vec3b>(i, j + 1)[0] +
						src.at<Vec3b>(i + 1, j)[0]) / 9;
					dst.at<Vec3b>(i, j)[1] = (src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i - 1, j - 1)[1] + src.at<Vec3b>(i - 1, j)[1] + src.at<Vec3b>(i, j - 1)[1] +
						src.at<Vec3b>(i - 1, j + 1)[1] + src.at<Vec3b>(i + 1, j - 1)[1] + src.at<Vec3b>(i + 1, j + 1)[1] + src.at<Vec3b>(i, j + 1)[1] +
						src.at<Vec3b>(i + 1, j)[1]) / 9;
					dst.at<Vec3b>(i, j)[2] = (src.at<Vec3b>(i, j)[2] + src.at<Vec3b>(i - 1, j - 1)[2] + src.at<Vec3b>(i - 1, j)[2] + src.at<Vec3b>(i, j - 1)[2] +
						src.at<Vec3b>(i - 1, j + 1)[2] + src.at<Vec3b>(i + 1, j - 1)[2] + src.at<Vec3b>(i + 1, j + 1)[2] + src.at<Vec3b>(i, j + 1)[2] +
						src.at<Vec3b>(i + 1, j)[2]) / 9;
				}
				else { 
					dst.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0];
					dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
					dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2];
				}
			}
		int   x, y;
		int   uiSubY, uiSubX, uiYU, uiYB, uiXL, uiXR;
		int   *pulLU, *pulRU, *pulLB, *pulRB;
		unsigned char   *pImPointer;
		int m_nGridSize_Y = image_width;
		int m_nGridSize_X = image_height;
		//x,y都为0  
		pImPointer = MedianBlurHaze;
		uiSubY = m_nGridSize_Y >> 1;
		uiSubX = m_nGridSize_X >> 1;

		//pulLU = &MedianBlurHaze[0];
		//pulRU = &m_pMapArray[0];
		//pulLB = &m_pMapArray[0];
		//pulRB = &m_pMapArray[0];
		//LineInterpolate(pImPointer, uiSubX, uiSubY, pulLU, pulRU, pulLB, pulRB);
		pImPointer += uiSubX;              /* set pointer on next matrix */

		{
			int r = 0;
			double sigma_d = 0.5;
			double sigma_r = 1.6;
			int i, j, m, n, k;
			int nx = image_width, ny = image_width;
			int w_filter = 2 * r + 1; // 滤波器边长  

			double gaussian_d_coeff = -0.5 / (sigma_d * sigma_d);
			double gaussian_r_coeff = -0.5 / (sigma_r * sigma_r);

			//double** d_metrix = NewDoubleMatrix(w_filter, w_filter);  // spatial weight  
			double r_metrix[256];  // similarity weight  
		}
		for (int i = 1; i<src.rows; ++i)
			for (int j = 1; j < src.cols; ++j) 
			{
				if ((i - 1 >= 0) && (j - 1) >= 0 && (i + 1)<src.rows && (j + 1)<src.cols) {
					dst.at<Vec3b>(i, j)[0] = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i - 1, j - 1)[0] + src.at<Vec3b>(i - 1, j)[0] + src.at<Vec3b>(i, j - 1)[0] +
						src.at<Vec3b>(i - 1, j + 1)[0] + src.at<Vec3b>(i + 1, j - 1)[0] + src.at<Vec3b>(i + 1, j + 1)[0] + src.at<Vec3b>(i, j + 1)[0] +
						src.at<Vec3b>(i + 1, j)[0]) / 9;
					dst.at<Vec3b>(i, j)[1] = (src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i - 1, j - 1)[1] + src.at<Vec3b>(i - 1, j)[1] + src.at<Vec3b>(i, j - 1)[1] +
						src.at<Vec3b>(i - 1, j + 1)[1] + src.at<Vec3b>(i + 1, j - 1)[1] + src.at<Vec3b>(i + 1, j + 1)[1] + src.at<Vec3b>(i, j + 1)[1] +
						src.at<Vec3b>(i + 1, j)[1]) / 9;
					dst.at<Vec3b>(i, j)[2] = (src.at<Vec3b>(i, j)[2] + src.at<Vec3b>(i - 1, j - 1)[2] + src.at<Vec3b>(i - 1, j)[2] + src.at<Vec3b>(i, j - 1)[2] +
						src.at<Vec3b>(i - 1, j + 1)[2] + src.at<Vec3b>(i + 1, j - 1)[2] + src.at<Vec3b>(i + 1, j + 1)[2] + src.at<Vec3b>(i, j + 1)[2] +
						src.at<Vec3b>(i + 1, j)[2]) / 9;
				}
				else {
					dst.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0];
					dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
					dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2];
				}
			}
		//y为0  
		uiYU = 0;
		uiYB = 0;
		int m_pMapArray[6000] = {0};
		for (x = 1; x<60; x++)
		{
			uiSubX = m_nGridSize_X;
			uiXL = x - 1;
			uiXR = uiXL + 1;
			int m_nHistBins = 60;
			pulLU = &m_pMapArray[m_nHistBins * uiXL];
			pulRU = &m_pMapArray[m_nHistBins * uiXR];
			pulLB = &m_pMapArray[m_nHistBins * uiXL];
			pulRB = &m_pMapArray[m_nHistBins * uiXR];
			LineInterpolate(pImPointer, uiSubX, uiSubY, pulLU, pulRU, pulLB, pulRB);
			pImPointer += uiSubX;              /* set pointer on next matrix */

		}

		//y为0，x为m_nGridX  
		int m_nGridX = image_width;
		int m_nGridY = image_width;
		int m_nHistBins = image_depth;
		int m_nWidth = image_width;
		uiSubX = m_nGridSize_X >> 1;
		uiXL = m_nGridX - 1;
		uiXR = uiXL;

		pulLU = &m_pMapArray[60 * uiXL];
		pulRU = &m_pMapArray[60 * uiXR];
		pulLB = &m_pMapArray[60 * uiXL];
		pulRB = &m_pMapArray[60 * uiXR];
		LineInterpolate(pImPointer, uiSubX, uiSubY, pulLU, pulRU, pulLB, pulRB);
		pImPointer += uiSubX;              /* set pointer on next matrix */
		pImPointer += (uiSubY - 1) * 5;

		for (y = 1; y<20; y++)
		{
			uiSubY = m_nGridSize_Y;
			uiYU = y - 1;
			uiYB = uiYU + 1;

			//x为0  
			uiSubX = m_nGridSize_X >> 1;
			pulLU = &m_pMapArray[m_nHistBins * uiYU * m_nGridX];
			pulRU = &m_pMapArray[m_nHistBins * uiYU * m_nGridX];
			pulLB = &m_pMapArray[m_nHistBins * uiYB * m_nGridX];
			pulRB = &m_pMapArray[m_nHistBins * uiYB * m_nGridX];
			LineInterpolate(pImPointer, uiSubX, uiSubY, pulLU, pulRU, pulLB, pulRB);
			pImPointer += uiSubX;              /* set pointer on next matrix */

			for (x = 1; x<m_nGridX; x++)
			{
				uiSubX = m_nGridSize_X;
				uiXL = x - 1;
				uiXR = uiXL + 1;
				pulLU = &m_pMapArray[m_nHistBins * (uiYU * m_nGridX + uiXL)];
				pulRU = &m_pMapArray[m_nHistBins * (uiYU * m_nGridX + uiXR)];
				pulLB = &m_pMapArray[m_nHistBins * (uiYB * m_nGridX + uiXL)];
				pulRB = &m_pMapArray[m_nHistBins * (uiYB * m_nGridX + uiXR)];
				LineInterpolate(pImPointer, uiSubX, uiSubY, pulLU, pulRU, pulLB, pulRB);
				pImPointer += uiSubX;              /* set pointer on next matrix */

			}

			uiSubX = m_nGridSize_X >> 1;
			uiXL = x - 1;
			uiXR = uiXL;
			pulLU = &m_pMapArray[m_nHistBins * (uiYU * m_nGridX + uiXL)];
			pulRU = &m_pMapArray[m_nHistBins * (uiYU * m_nGridX + uiXR)];
			pulLB = &m_pMapArray[m_nHistBins * (uiYB * m_nGridX + uiXL)];
			pulRB = &m_pMapArray[m_nHistBins * (uiYB * m_nGridX + uiXR)];
			LineInterpolate(pImPointer, uiSubX, uiSubY, pulLU, pulRU, pulLB, pulRB);
			pImPointer += uiSubX;              /* set pointer on next matrix */
			pImPointer += (uiSubY - 1) * m_nWidth;

		}

		//y为m_nGridY  
		uiSubY = m_nGridSize_Y >> 1;
		uiYU = m_nGridY - 1;
		uiYB = uiYU;

		//x为0  
		uiSubX = m_nGridSize_X >> 1;
		pulLU = &m_pMapArray[m_nHistBins * uiYU * m_nGridX];
		pulRU = &m_pMapArray[m_nHistBins * uiYU * m_nGridX];
		pulLB = &m_pMapArray[m_nHistBins * uiYB * m_nGridX];
		pulRB = &m_pMapArray[m_nHistBins * uiYB * m_nGridX];
		LineInterpolate(pImPointer, uiSubX, uiSubY, pulLU, pulRU, pulLB, pulRB);
		pImPointer += uiSubX;              /* set pointer on next matrix */

		for (x = 1; x<m_nGridX; x++)
		{
			uiSubX = m_nGridSize_X;
			uiXL = x - 1;
			uiXR = uiXL + 1;
			pulLU = &m_pMapArray[m_nHistBins * (uiYU * m_nGridX + uiXL)];
			pulRU = &m_pMapArray[m_nHistBins * (uiYU * m_nGridX + uiXR)];
			pulLB = &m_pMapArray[m_nHistBins * (uiYB * m_nGridX + uiXL)];
			pulRB = &m_pMapArray[m_nHistBins * (uiYB * m_nGridX + uiXR)];
			LineInterpolate(pImPointer, uiSubX, uiSubY, pulLU, pulRU, pulLB, pulRB);
			pImPointer += uiSubX;              /* set pointer on next matrix */

		}
		
		uiSubX = m_nGridSize_X >> 1;
		uiXL = x - 1;
		uiXR = uiXL;
		pulLU = &m_pMapArray[m_nHistBins * (uiYU * m_nGridX + uiXL)];
		pulRU = &m_pMapArray[m_nHistBins * (uiYU * m_nGridX + uiXR)];
		pulLB = &m_pMapArray[m_nHistBins * (uiYB * m_nGridX + uiXL)];
		pulRB = &m_pMapArray[m_nHistBins * (uiYB * m_nGridX + uiXR)];
		LineInterpolate(pImPointer, uiSubX, uiSubY, pulLU, pulRU, pulLB, pulRB);
		pImPointer += uiSubX;              /* set pointer on next matrix */
		pImPointer += (uiSubY - 1) * m_nWidth;
    }
	printf("*********************** 第三步：END **************************\n");
	printf("*************** 第四步：基于融合的转换估计 ********************\n");
	// 第四步：基于融合的转换估计
	if (hinst != NULL)
	{
		double *out_double = new double[10];
		double *out_int = new double[10];
		double *W = new double[10];
		double sum_W = 0;
		//构造权值矩阵
		for (int i = 0; i < 10; i++) {  
			W[i] = i * i;
			sum_W += W[i];
		}
		for (int i = 0; i < 10; i++) {
			W[i] = W[i] / sum_W;
		}

		for (int i = 0; i < 10; i++) {
			out_double[i] = 1;
		}
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 10; j++) {
				out_double[i] += i;
			}
		}
		for (int i = 0; i < 10; i++) {
			out_int[i] = out_double[i] * ((double)10);
		}
		/// </summary>
		/// <param name="Src">图像数据在内存的起始地址。</param>
		/// <param name="Dest">目标数据在内存的起始地址。</param>
		/// <param name="Width">图像的宽度。</param>
		/// <param name="Height">图像的高度。</param>
		/// <param name="Stride">图像的扫描行大小。</param>
		/// <param name="BlockSize">用于计算暗通道图像时的矩形半径。</param>
		/// <param name="GuideRadius">导向滤波的半径。</param>
		/// <param name="MaxAtom">为防止图像天空部分出现holes，设置的最大大气光值，默认240。</param>
		/// <param name="Omega">控制去雾程度的一个参数，建议取值范围[0.75,1],值越大，去雾越明显，但可能出现局部过增强。</param>
		/// <param name="T0">用于控制最小透射率的一个参数，建议取值范围[0.01,0.2]。</param>!
		/// <param name="Gamma">调整亮度的参数，建议范围[0.7,1]。</param>
		// cout << "ImageMaster.dll load SUCCESS !!!" << endl;

		pfFuncInDll = (Dehaze)GetProcAddress(hinst, "IM_HazeRemovalBasedOnDarkChannelPrior");

		//unsigned char A[1024 * 768 * 3];
		//unsigned char B[1024 * 768 * 3];
		unsigned char* A = (unsigned char *)malloc(sizeof(unsigned char) * (1024 * 768 * 3));
		unsigned char* B = (unsigned char *)malloc(sizeof(unsigned char) * (1024 * 768 * 3));

		if (pfFuncInDll != NULL)
		{
			cout << "IM_HazeRemoval function load SUCCESS !!!" << endl;

			clock_t start = clock();
			//int c = pfFuncInDll(A, B, 1024, 768, 3072, 4, 5, 220, 0.9F, 0.01F, 0.9F);
			//int d = pfFuncInDll(C, D, 1023, 763, 4092, 4, 5, 220, 0.9F, 0.01F, 0.9F);
			int P = pfFuncInDll(Src, Dest, Width, Height, Stride, BlockSize, GuideRadius, MaxAtom, Omega, T0, Gamma);
			//int H = pfFuncInDll(Src2, Dest2, Width2, Height2, Stride2, BlockSize, GuideRadius, MaxAtom, Omega, T0, Gamma);
			clock_t ends = clock();
			outputImagemat.data = Dest;

			cout << "Running Time : " << (double)(ends - start) << "ms" << endl;
			cout << (int)C[100] << "\t" << (int)C[101] << "\t" << (int)C[102] << "\t" << endl;
			cout << (int)D[100] << "\t" << (int)D[101] << "\t" << (int)D[102] << "\t" << endl;
			cout << (int)C[1000] << "\t" << (int)C[1001] << "\t" << (int)C[1002] << "\t" << endl;
			cout << (int)D[1000] << "\t" << (int)D[1001] << "\t" << (int)D[1002] << "\t" << endl;
			cout << P << endl;
		}

		FreeLibrary(hinst);
	}

	printf("*********************** 第四步：END **************************\n");
	printf("*************** 第五步：输出恢复的图像 *************************\n");
	if (is_display)
	{
		//创建一个名字为MyWindow的窗口
		// 图片的显示
		cout << "input image function load SUCCESS !!!" << endl;
		cout << "output image function load SUCCESS !!!" << endl;
		imshow("带雾图像", inputImagemat);
		imshow("恢复图像", outputImagemat);
		waitKey(0);
	}
	printf("*********************** 第五步：END **************************\n");
	system("pause");
	return 0;
}



