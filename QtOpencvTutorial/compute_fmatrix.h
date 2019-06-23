#ifndef COMPUTE_FMATRIX_H
#define COMPUTE_FMATRIX_H

#include <vector>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <find_matches.h>
#include <stdio.h>
#include <stdarg.h>

class FMatrix
{
private:
    std::vector<cv::DMatch> selMatches; //to get selected matches
    std::vector<cv::DMatch> matches; //to get all matches
    std::vector<int> pointIndexes1;
    std::vector<int> pointIndexes2;
    std::vector<cv::Point2f> selPoints1, selPoints2; // all three for conversions of points to calculate F
    cv::Mat fundamental; //for fundamental matrix
    std::vector<cv::Vec3f> lines1; //epipolar lines
    std::vector<cv::Vec3f> lines2; //for epipolar lines
    std::vector<cv::Point2f> points1, points2; //for RANSAC
    cv::Ptr<cv::xfeatures2d::SURF> surfDetector; // pointer to surf detector
    std::vector<cv::KeyPoint> keypoints1; // to hold keypoints
    std::vector<cv::KeyPoint> keypoints2; // to hold keypoints of second image in image matching
    cv::Mat descriptors, descriptors2; // descriptors for first and second image
    cv::Mat imageFeatures; // to hold feature image
    std::vector<cv::Point2f> points1In, points2In; //Inlier points RANSAC
    cv::Mat imageMatches;
    cv::Mat displayImages(std::vector<cv::Mat> outImage);

public:

    FMatrix();

    cv::FlannBasedMatcher flannBasedMatcher; // object for flann based matcher
    FindMatches matcher; //object for robust matcher class


        // function to get fundamental matrix from 7 point algo
        void getFundamental7PointAndDrawEpilines(cv::Mat image, cv::Mat image2, cv::Mat &img, cv::Mat &fundamental);

        // function to get fundamental matrix from 7 point algo
        void getFundamental8PointAndDrawEpilines(cv::Mat image, cv::Mat image2, cv::Mat &img, cv::Mat &fundamental);


        // function to get fundamental matrix from 7 point algo
        void getFundamentalRANSACAndDrawEpilines(cv::Mat image, cv::Mat image2, cv::Mat &img, cv::Mat &fundamental);

        //function to get homography
        void getHomography(cv::Mat image, cv::Mat image2, cv::Mat &img, cv::Mat &fundamental);


};

#endif // COMPUT_FMATRIX_H
