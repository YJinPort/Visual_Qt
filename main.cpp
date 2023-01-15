#include "QtWidgetsApplication_test.h"
#include <QtWidgets/QApplication>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//ũ�� ���� �Լ�
void scaling(Mat img, Mat& dst, Size size)
{
    dst = Mat(size, img.type(), Scalar(0));
    
    double ratioY = (double)size.height / img.rows;
    double ratioX = (double)size.width / img.cols;

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            int x = (int)(j * ratioX);
            int y = (int)(i * ratioY);
            dst.at<uchar>(y, x) = img.at<uchar>(i, j);
        }
    }
}

//�ֱ��� �̿� ����
void scaling_nearest(Mat img, Mat& dst, Size size)
{
    dst = Mat(size, CV_8U, Scalar(0));

    double ratioY = (double)size.height / img.rows;
    double ratioX = (double)size.width / img.cols;

    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            int x = (int)cvRound(j / ratioX);
            int y = (int)cvRound(i / ratioY);
            dst.at<uchar>(i, j) = img.at<uchar>(y, x);
        }
    }
}

//���� ȭ�� �缱�� ����
uchar bilinear_value(Mat img, double x, double y)
{
    if (x >= img.cols - 1) x--;
    if (y >= img.rows - 1) y--;

    Point pt((int)x, (int)y);
    int A = img.at<uchar>(pt);
    int B = img.at<uchar>(pt + Point(0, 1));
    int C = img.at<uchar>(pt + Point(1, 0));
    int D = img.at<uchar>(pt + Point(1, 1));

    double alpha = y - pt.y;
    double beta = x - pt.x;

    int M1 = A + (int)cvRound(alpha * (B - A));
    int M2 = C + (int)cvRound(alpha * (D - C));
    int P = M1 + (int)cvRound(beta * (M2 - M1));

    return saturate_cast<uchar>(P);
}

//ũ�⺯�� - �缱�� ����
void scaling_bilinear(Mat img, Mat& dst, Size size)
{
    dst = Mat(size, img.type(), Scalar(0));

    double ratioY = (double)size.height / img.rows;
    double ratioX = (double)size.width / img.cols;

    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            double y = i / ratioY;
            double x = j / ratioX;

            dst.at<uchar>(i, j) = bilinear_value(img, x, y);
        }
    }
}

void translation(Mat img, Mat& dst, Point pt)
{
    Rect rect(Point(0, 0), img.size());
    dst = Mat(img.size(), img.type(), Scalar(0));

    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            Point dst_pt(j, i);
            Point img_pt = dst_pt - pt;
            if (rect.contains(img_pt))
            {
                dst.at<uchar>(dst_pt) = img.at<uchar>(img_pt);
            }
        }
    }
}

void rotation(Mat img, Mat& dst, double dgree)
{
    double radian = dgree / 180 * CV_PI;
    double sin_value = sin(radian);
    double cos_value = cos(radian);

    Rect rect(Point(0, 0), img.size());
    dst = Mat(img.size(), img.type(), Scalar(0));

    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            double x = (j * cos_value) + (i * sin_value);
            double y = (-j * sin_value) + (i * cos_value);

            if (rect.contains(Point2d(x, y)))
            {
                dst.at<uchar>(i, j) = bilinear_value(img, x, y);
            }
        }
    }
}

void rotation(Mat img, Mat& dst, double dgree, Point pt)
{
    double radian = dgree / 180 * CV_PI;
    double sin_value = sin(radian);
    double cos_value = cos(radian);

    Rect rect(Point(0, 0), img.size());
    dst = Mat(img.size(), img.type(), Scalar(0));

    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            int jj = j - pt.x;
            int ii = i - pt.y;

            double x = (jj * cos_value) + (ii * sin_value) + pt.x;
            double y = (-jj * sin_value) + (ii * cos_value) + pt.y;

            if (rect.contains(Point2d(x, y)))
            {
                dst.at<uchar>(i, j) = bilinear_value(img, x, y);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QtWidgetsApplication_test w;

    Mat image = imread("./lena.bmp", 0);
    CV_Assert(image.data);

#if 0
    //ħ��(erosion)
    Mat binImage, erosionImage;
    threshold(image, binImage, 128, 255, THRESH_BINARY);

    uchar sample[] = { 0, 1, 0,
                       1, 1, 1,
                       0, 1, 0 };
    Mat mask(3, 3, CV_8UC1, sample);

    morphologyEx(binImage, erosionImage, MORPH_ERODE, mask);

    imshow("image", image);
    imshow("binImage", binImage);
    imshow("erosionImage", erosionImage);

#elif 0
    //��â(dilation)
    Mat binImage, dilationImage;
    threshold(image, binImage, 128, 255, THRESH_BINARY);

    uchar sample[] = { 0, 1, 0,
                       1, 1, 1,
                       0, 1, 1 };
    Mat mask(3, 3, CV_8UC1, sample);

    morphologyEx(binImage, dilationImage, MORPH_DILATE, mask);

    imshow("image", image);
    imshow("binImage", binImage);
    imshow("dilationImage", dilationImage);

#elif 0
    //����(opening), ����(closing)
    Mat binImage, openingImage, closingImage;
    threshold(image, binImage, 128, 255, THRESH_BINARY);

    uchar sample[] = { 0, 1, 0,
                       1, 1, 1,
                       0, 1, 1 };
    Mat mask(3, 3, CV_8UC1, sample);

    morphologyEx(binImage, openingImage, MORPH_OPEN, mask);
    morphologyEx(binImage, closingImage, MORPH_CLOSE, mask);

    /*imshow("image", image);
    imshow("binImage", binImage);*/
    imshow("openingImage", openingImage);
    imshow("closingImage", closingImage);

#elif 0
    Mat dst1, dst2;
    scaling(image, dst1, Size(300, 300));
    scaling(image, dst2, Size(600, 600));

    imshow("image", image);
    imshow("���", dst1);
    imshow("Ȯ��", dst2);

#elif 0
    Mat dst1, dst2;
    scaling(image, dst1, Size(600, 600));
    scaling_nearest(image, dst2, Size(600, 600));

    imshow("�������� Ȯ��", dst1);
    imshow("�ֱ��� �̿������� Ȯ��", dst2);

#elif 0
    Mat dst1, dst2;
    scaling_nearest(image, dst1, Size(600, 600));
    scaling_bilinear(image, dst2, Size(600, 600));

    imshow("�ֱ��� �̿������� Ȯ��", dst1);
    imshow("�缱�� ������ Ȯ��", dst2);

#elif 0
    Mat dst1, dst2;
    translation(image, dst1, Point(30, 80));
    translation(image, dst2, Point(-80, -50));

    imshow("(30, 80)�̵�", dst1);
    imshow("(-80, -50)�̵�", dst2);

#elif 0
    Mat dst1, dst2;
    Point center = image.size() / 2;

    rotation(image, dst1, 20);
    rotation(image, dst2, 20, center);

    imshow("20�� ȸ��(����)", dst1);
    imshow("20�� ȸ��(�߽���)", dst2);

#elif 1
    double shear_value = 0.3;
    Mat shearing = Mat_<double>({ 2, 3 }, { 1, shear_value, 0, 0, 1, 0 });
    Mat shearingImage;

    warpAffine(image, shearingImage, shearing, Size(cvRound(image.cols + (image.rows * shear_value)), image.rows));

    imshow("����", image);
    imshow("shearingImage", shearingImage);

#endif

    waitKey();

    w.show();
    return a.exec();
}
