#include "extract_features.h"

Features::Features()
{

}

// function to get FAST
cv::Mat Features::getFastFeatures(cv::Mat image, int thresholdFast){

    cv::Mat src_gray = image.clone();    // copy image to image features

    if (image.type() != CV_8UC1){
        /// Convert the image to grayscale
        cvtColor( image, src_gray, CV_BGR2GRAY );
    }

    // fastfeature detector create
    fastDetector = cv::FastFeatureDetector::create(thresholdFast);

    // to detect key points
    fastDetector->detect(src_gray, keypoints);

    //to draw keypoints
    cv::drawKeypoints(image, keypoints, image, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

    return image;
}


//function to get SURF features
cv::Mat Features::getSurfFeatures(cv::Mat image, double thresholdSurf){

    cv::Mat src_gray = image.clone();    // copy image to image features

    if (image.type() != CV_8UC1){
        /// Convert the image to grayscale
        cvtColor( image, src_gray, CV_BGR2GRAY );
    }

    // SURF detector create
    surfDetector = cv::xfeatures2d::SURF::create(thresholdSurf*100);

    // to detcect key points
    surfDetector->detect(src_gray, keypoints);

    // to draw keypoints
    cv::drawKeypoints(image, keypoints, image, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    return image;
}

//function to get SIFT features
cv::Mat Features::getSiftFeatures(cv::Mat image, double edgeThreshold){

    cv::Mat src_gray = image.clone();    // copy image to image features

    if (image.type() != CV_8UC1){
        /// Convert the image to grayscale
        cvtColor( image, src_gray, CV_BGR2GRAY );
    }

    //SIFT detector create
    siftDetector = cv::xfeatures2d::SIFT::create(0, 3, 0.04, edgeThreshold*10);

    // to detect keypoints
    siftDetector->detect(src_gray, keypoints);

    // to draw keypoints
    cv::drawKeypoints(image, keypoints, image, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    return image;
}

//function to get matched features bewteen two images
cv::Mat Features::getMatchFeatures(cv::Mat image, cv::Mat image2, int thresh){

    cv::Mat src_gray = image.clone();    // copy image
    cv::Mat src_gray2 = image2.clone();    // copy image

    if (image.type() != CV_8UC1){
        /// Convert the image to grayscale
        cvtColor( image, src_gray, CV_BGR2GRAY );
    }

    if (image2.type() != CV_8UC1){
        /// Convert the image to grayscale
        cvtColor( image2, src_gray2, CV_BGR2GRAY );
    }

    //compute keypoints for image
    surfDetector = cv::xfeatures2d::SURF::create(thresh);
    surfDetector->detect(src_gray, keypoints);

    //compute keypoints for image2
    surfDetector->detect(src_gray2, keypoints2);

    //compute descriptors for firsta and second images
    surfDetector->compute(src_gray, keypoints, descriptors);
    surfDetector->compute(src_gray2, keypoints2, descriptors2);

    //detect matches
    flannBasedMatcher.match(descriptors, descriptors2, matches);

    // deleter matches after first 'n' matches
    std::nth_element(matches.begin(), matches.begin() + 15 - 1, matches.end());
    matches.erase(matches.begin() + 15, matches.end());

    //to draw matches
    cv::drawMatches(image, keypoints, image2, keypoints2, matches, imageFeatures, cv::Scalar(0, 0, 255));


    return imageFeatures;

}
