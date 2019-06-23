#ifndef EXTRACT_FEATURES_H
#define EXTRACT_FEATURES_H

#include <QApplication>
#include <QWidget>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <QDebug>


class Features
{
private:
    std::vector<cv::KeyPoint> keypoints; // to hold keypoints
    std::vector<cv::KeyPoint> keypoints2; // to hold keypoints of second image in image matching
    cv::Mat imageFeatures; // to hold feature image
    cv::Ptr<cv::FastFeatureDetector> fastDetector; // pointer of fastfeaturedetector
    cv::Ptr<cv::xfeatures2d::SURF> surfDetector; // pointer to surf detector
    cv::Ptr<cv::xfeatures2d::SIFT> siftDetector; // pointer to SIFT detector
    cv::Mat descriptors, descriptors2; // descriptors for first and second image
    cv::FlannBasedMatcher flannBasedMatcher; // object for flann based matcher
    std::vector<cv::DMatch> matches; // variable to store matches
    std::vector<cv::DMatch> selMatches; //to select matches for fundamnetal matrix

public:
    Features();

    // function to get FAST features
    cv::Mat getFastFeatures(cv::Mat image, int thresholdFast);

    //function to get SURF features
    cv::Mat getSurfFeatures(cv::Mat image, double thresholdSurf);

    //function to get SIFT features
    cv::Mat getSiftFeatures(cv::Mat image, double edgeThreshold);

    //function to get matched features bewteen two images
    cv::Mat getMatchFeatures(cv::Mat image, cv::Mat image2, int firstNMatches);

};



#endif // EXTRACT_FEATURES_H
