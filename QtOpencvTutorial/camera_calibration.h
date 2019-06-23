#ifndef CAMERA_CALIBRATION_H
#define CAMERA_CALIBRATION_H

#include <vector>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>

class CameraCalibration
{
    // input points
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    // output Matrices
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    // flag to specify how calibration is done
    int flag;
    // used in image undistortion
    cv::Mat map1,map2;
    bool mustInitUndistort;

    // the points on the chessboard
    std::vector<cv::Point2f> imageCorners;
    std::vector<cv::Point3f> objectCorners;
    cv::Mat image; // to contain chessboard image
    cv::Mat undistorted; // for undistored image


public:
    CameraCalibration() : flag(0), mustInitUndistort(true) {}

    // Open the chessboard images and extract corner points
    std::vector<cv::Mat> addChessboardPoints(const std::vector<cv::String> &filelist, cv::Size & boardSize);

    // Add scene points and corresponding image points
    void addPoints(const std::vector<cv::Point2f>& imageCorners, const std::vector<cv::Point3f>& objectCorners);

    // Calibrate the camera
    double calibrate(cv::Size imageSize);

    // Set the calibration flag
    void setCalibrationFlag(bool radial8CoeffEnabled=false, bool tangentialParamEnabled=false);

    // Remove distortion in an image (after calibration)
    cv::Mat remap(const cv::Mat &image);

    // Getters
    cv::Mat getCameraMatrix();
    cv::Mat getDistCoeffs();

};
#endif // CAMERA_CALIBRATION_H
