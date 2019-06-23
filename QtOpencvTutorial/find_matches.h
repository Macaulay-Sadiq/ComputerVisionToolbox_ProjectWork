#ifndef FINDMATCHES_H
#define FINDMATCHES_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <QDebug>

class FindMatches
{
public:

    // pointer to the feature point detector object
        cv::Ptr<cv::FeatureDetector> detector;
        // pointer to the feature descriptor extractor object
        cv::Ptr<cv::DescriptorExtractor> extractor;
        float ratio; // max ratio between 1st and 2nd NN
        bool refineF; // if true will refine the F matrix
        double distance; // min distance to epipolar
        double confidence; // confidence level (probability)

        std::vector<cv::KeyPoint> keypoints1;
        std::vector<cv::KeyPoint> keypoints2;

        cv::Ptr<cv::xfeatures2d::SURF> surfDetector; // pointer to surf detector
        cv::FlannBasedMatcher flannBasedMatcher; // object for flann based matcher

    public:
        FindMatches();


        // Set the minimum distance to epipolar in RANSAC
        void setMinDistanceToEpipolar(double d);

        // Set confidence level in RANSAC
        void setConfidenceLevel(double c) ;

        // Set the NN ratio
        void setRatio(float r) ;

        // if you want the F matrix to be recalculated
        void refineFundamental(bool flag) ;

        // Clear matches for which NN ratio is > than threshold
        // return the number of removed points
        // (corresponding entries being cleared, i.e. size will be 0)
        int ratioTest(std::vector<std::vector<cv::DMatch>> &matches);

        // Insert symmetrical matches in symMatches vector
        void symmetryTest(const std::vector<std::vector<cv::DMatch> > &matches1,
                          const std::vector<std::vector<cv::DMatch> > &matches2,
                          std::vector<cv::DMatch>& symMatches);

        // Identify good matches using RANSAC
        // Return fundemental matrix
        cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,
                           const std::vector<cv::KeyPoint>& keypoints1,
                           const std::vector<cv::KeyPoint>& keypoints2,
                           std::vector<cv::DMatch>& outMatches) ;

        // Match feature points using symmetry test and RANSAC
        // returns fundemental matrix
        cv::Mat match(cv::Mat image1, cv::Mat image2, // input images
            std::vector<cv::DMatch>& matches, // output matches and keypoints
            std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2);
};

#endif // FINDMATCHES_H
