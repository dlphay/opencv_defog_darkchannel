// ���ڸ�Ч�ںϵ�ȥ��
// ��һ������������ͼ��
// �ڶ��������ڸ�˹��ͨ���Ĵ������߹���
// ������������CLAHE�ľ�ϸϸ������
// ���Ĳ��������ںϵ�ת������
// ���岽������ָ���ͼ��

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

// ȫ�ֱ���
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
			double p = cvGetReal2D(dark, i - 1, j - 1);  //�õ����������½ǵ�һ�����ص�  
			//�õ�������С������ֵ  
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
			//����Icy��ֵ  
			for (int ii = i - w; ii < i; ii++)
			{
				for (int jj = j - w; jj < j; jj++)
				{
					cvSetReal2D(Icy, ii, jj, p);
				}
			}

		}
	}

	//�������ұ�һ��  ����������һ���ӿ�  
	for (int i = w; i < (hw - 1)*w; i += w)
	{
		double p = cvGetReal2D(dark, i - 1, wid - 1);  //�õ����������½ǵ�һ�����ص�  
		for (int ii = i - w; ii < i; ii++)
		{

			for (int j = (ww - 1)*w; j < wid; j++)
			{
				//�õ�������С������ֵ  
				double newp = cvGetReal2D(dark, ii, j);
				if (newp < p)
				{
					p = newp;
				}
			}
		}

		//����Icy��ֵ  
		for (int ii = i - w; ii < i; ii++)
		{

			for (int j = (ww - 1)*w; j < wid; j++)
			{
				cvSetReal2D(Icy, ii, j, p);
			}
		}
	}


	//��������һ�� ���������һ���ӿ�  
	for (int j = w; j < (ww - 1)*w; j += w)
	{
		double p = cvGetReal2D(dark, hei - 1, j);  //�õ����������½ǵ�һ�����ص�  
		for (int i = (hw - 1)*w; i < hei; i++)
		{
			for (int jj = j - w; jj < j; jj++)
			{
				//�õ�������С������ֵ  
				double newp = cvGetReal2D(dark, i, jj);
				if (newp < p)
				{
					p = newp;
				}
			}
		}

		//����Icy��ֵ  
		for (int i = (hw - 1)*w; i < hei; i++)
		{

			for (int jj = j - w; jj < j; jj++)
			{
				cvSetReal2D(Icy, i, jj, p);
			}
		}

	}

	//���������½ǵ�һ���ӿ�  
	double p = cvGetReal2D(dark, hei - 1, wid - 1);  //�õ����������½ǵ�һ�����ص�  
	for (int i = (hw - 1)*w; i < hei; i++)
	{
		for (int j = (ww - 1)*w; j < wid; j++)
		{
			//�õ�������С������ֵ  
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
	FILE* fp = fopen(bmpName, "rb"); //�Զ����ƶ��ķ�ʽ��ָ����ͼ���ļ�
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


// �޼�ֱ��ͼ  
void ClipHistogram(int* pHistogram, int clipThreshold)
{
	int binExcess = 0, totalExcess = 0, avgBinIncr = 0, upperLimit = 0;
	// �ۻ�������ֵ����  
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

	// �޼�ֱ��ͼ�����·�����ֵ  
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

	// ʣ�ಿ���ٴη���  
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
	printf("*********************** ��һ������������ͼ�� **************************\n");
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

	int image_depth = 3; // Ĭ����4ͨ����


	// ���ձ���ͼƬ�Ļ�������
	unsigned char* D = (unsigned char *)malloc(sizeof(unsigned char) * (image_width * image_height * image_depth));
	unsigned char* A = (unsigned char *)malloc(sizeof(unsigned char) * (image_width * image_height * image_depth));

	unsigned char* E = (unsigned char *)malloc(sizeof(unsigned char) * (image_width2 * image_height2 * image_depth));
	unsigned char* F = (unsigned char *)malloc(sizeof(unsigned char) * (image_width2 * image_height2 * image_depth));

	unsigned char* MedianBlurHaze = (unsigned char *)malloc(sizeof(unsigned char) * (image_width * image_height * image_depth));
	//Stride�ļ��㹫ʽ��Stride = ÿ����ռ�õ��ֽ���(Ҳ��������λ��/8) * Width;��� Stride ���� 4 �ı���, ��ô Stride = Stride + (4 - Stride mod 4)
	A = inputImagemat.data;
	MedianBlurHaze = inputImagemat.data;
	Mat src = inputImagemat;
	Mat dst = inputImagemat;

	E = inputImagemat2.data;
	MedianBlurHaze = inputImagemat2.data;
	Mat src2 = inputImagemat2;
	Mat dst2 = inputImagemat2;

	// ȥ����ر���
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

	// ȥ����ر���
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

	//����һ������ΪMyWindow�Ĵ���
	//// ͼƬ����ʾ
	//imshow("����ͼ��", inputImagemat);
	//waitKey(0);

	// �ڶ��������ڸ�˹��ͨ���Ĵ������߹���
	printf("*********************** ��һ����END **************************\n");
	printf("******* �ڶ��������ڸ�˹��ͨ���Ĵ������߹��ƿ�ʼִ�� *********\n");
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
			DarkP = DarkChannel + Y * Width;  // ��ʵ��ͼ�����õ���ͼ��ͨ��     
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
		memcpy(Filter, DarkChannel, Width * Height);   // ��ȫ�ִ�����Aʱ���ƻ�DarkChannel�е�����

		//MinValue(DarkChannel, Width, Height, Width, DarkRadius);   // ��ȡ��ͨ��ֵ

		// ���ð�ͨ��������ȫ�ִ�����ֵA
		int Sum, Value, Threshold = 0;
		int SumR = 0, SumG = 0, SumB = 0, AtomR, AtomB, AtomG, Amount = 0;
		int* Histgram = (int*)calloc(256, sizeof(int));
		for (Y = 0; Y < Width * Height; Y++) Histgram[DarkChannel[Y]]++;
		for (Y = 255, Sum = 0; Y >= 0; Y--)
		{
			Sum += Histgram[Y];
			if (Sum > Height * Width * 0.01)
			{
				Threshold = Y;   // ѡȡ��ͨ��ֵ��ǰ1%��������������Ϊ��ѡ��
				break;
			}
		}
		AtomB = 0; AtomG = 0; AtomR = 0;
		for (Y = 0, DarkP = DarkChannel; Y < Height; Y++)
		{
			Pointer = MedianBlurHaze + Y * Stride;
			for (X = 0; X < Width-10; X++)
			{
				if (*DarkP >= Threshold)   //    ��ԭͼ��ѡ�������ѡ���λ�õ�������Ϊ����ȫ�ִ�����A����Ϣ                        
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

		memcpy(DarkChannel, Filter, Width * Height);   // �ָ�DarkChannel�е�����
		//MedianBlur(Filter, Width, Height, Width, MedianRadius, 50); 
		memcpy(FilterClone, Filter, Width * Height);

		DarkP = DarkChannel;
		FilterP = Filter;
		for (Y = 0; Y < Height * Width; Y++)    //����һ��ѭ������������ٶ�
		{
			Diff = *DarkP - *FilterP;    //ͨ����|DarkP ��FilterP |ִ����ֵ�˲������Ƶľֲ���׼��������Ա�֤��׼����Ƶ�³����
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
			Diff = *FilterPC - *FilterP;                    // ����2��Ȼ���ǵ��нϺöԱȶȵ������������û���� �ⲿ������Ͳ���Ҫ��ȥ����
			if (Diff < 0) Diff = 0;                            // �����������������Ϊ������и�max(....,0)�Ĺ��̣�
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
				*FilterP = Min;                                // �������Լ�������Ĵ�����Ļ
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


	printf("*********************** �ڶ�����END **************************\n");
	printf("**************** ������������CLAHE�ľ�ϸϸ������ **************\n");
	 //������������CLAHE�ľ�ϸϸ������
	{
		cout << "CLAHE function load SUCCESS !!!" << endl;
		//at�������ص�  
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
		//x,y��Ϊ0  
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
			int w_filter = 2 * r + 1; // �˲����߳�  

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
		//yΪ0  
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

		//yΪ0��xΪm_nGridX  
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

			//xΪ0  
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

		//yΪm_nGridY  
		uiSubY = m_nGridSize_Y >> 1;
		uiYU = m_nGridY - 1;
		uiYB = uiYU;

		//xΪ0  
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
	printf("*********************** ��������END **************************\n");
	printf("*************** ���Ĳ��������ںϵ�ת������ ********************\n");
	// ���Ĳ��������ںϵ�ת������
	if (hinst != NULL)
	{
		double *out_double = new double[10];
		double *out_int = new double[10];
		double *W = new double[10];
		double sum_W = 0;
		//����Ȩֵ����
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
		/// <param name="Src">ͼ���������ڴ����ʼ��ַ��</param>
		/// <param name="Dest">Ŀ���������ڴ����ʼ��ַ��</param>
		/// <param name="Width">ͼ��Ŀ�ȡ�</param>
		/// <param name="Height">ͼ��ĸ߶ȡ�</param>
		/// <param name="Stride">ͼ���ɨ���д�С��</param>
		/// <param name="BlockSize">���ڼ��㰵ͨ��ͼ��ʱ�ľ��ΰ뾶��</param>
		/// <param name="GuideRadius">�����˲��İ뾶��</param>
		/// <param name="MaxAtom">Ϊ��ֹͼ����ղ��ֳ���holes�����õ���������ֵ��Ĭ��240��</param>
		/// <param name="Omega">����ȥ��̶ȵ�һ������������ȡֵ��Χ[0.75,1],ֵԽ��ȥ��Խ���ԣ������ܳ��־ֲ�����ǿ��</param>
		/// <param name="T0">���ڿ�����С͸���ʵ�һ������������ȡֵ��Χ[0.01,0.2]��</param>!
		/// <param name="Gamma">�������ȵĲ��������鷶Χ[0.7,1]��</param>
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

	printf("*********************** ���Ĳ���END **************************\n");
	printf("*************** ���岽������ָ���ͼ�� *************************\n");
	if (is_display)
	{
		//����һ������ΪMyWindow�Ĵ���
		// ͼƬ����ʾ
		cout << "input image function load SUCCESS !!!" << endl;
		cout << "output image function load SUCCESS !!!" << endl;
		imshow("����ͼ��", inputImagemat);
		imshow("�ָ�ͼ��", outputImagemat);
		waitKey(0);
	}
	printf("*********************** ���岽��END **************************\n");
	system("pause");
	return 0;
}



