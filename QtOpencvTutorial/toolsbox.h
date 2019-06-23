#ifndef TOOLSBOX_H
#define TOOLSBOX_H

#include <QObject>
#include <QWidget>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/features2d.hpp"
//#include "opencv2/xfeatures2d.hpp"





class toolsBox
{
public:
    toolsBox();
    ~toolsBox();
    void saltAndPepper(cv::Mat Image);
    cv::Mat histogramOfImage(cv::Mat);
    cv::Mat morphologyOperation(cv::Mat, int, int, int);
    cv::Mat imageErode(cv::Mat);
    cv::Mat imageOpen(cv::Mat);
    cv::Mat imageClose(cv::Mat);
    cv::Mat histEqualization(cv::Mat);
    cv::Mat imageBlur(cv::Mat,int,int);
    cv::Mat sobelOperation(cv::Mat, int);
    cv::Mat laplacianOperation(cv::Mat, int);
    cv::Mat cannyEdgeDetection(cv::Mat, int);
    cv::Mat HoughLines(cv::Mat, int);
    cv::Mat HoughCircles(cv::Mat, int);
    cv::Mat imageContours(cv::Mat,int thresh);
    cv::Mat shapeDescriptor(cv::Mat, int);
    cv::Mat harrisCornerDetector(cv::Mat, int);
    cv::Mat extractSURF(cv::Mat);
};

#endif // TOOLSBOX_H
