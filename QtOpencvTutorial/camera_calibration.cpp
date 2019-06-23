#include "camera_calibration.h"




std::vector<cv::Mat> CameraCalibration::addChessboardPoints(const std::vector<cv::String>& filelist, cv::Size & boardSize){

    std::vector<cv::Mat> data;
    // 3D Scene Points:
    // Initialize the chessboard corners
    // in the chessboard reference frame
    // The corners are at 3D location (X,Y,Z)= (i,j,0)
    for (int i=0; i<boardSize.height; i++) {
        for (int j=0; j<boardSize.width; j++) {

            objectCorners.push_back(cv::Point3f(i, j, 0.0f));
        }
    }

    // 2D Image points:
    cv::Mat image; // to contain chessboard image
    int successes = 0;
    // for all viewpoints
    for (int i=0; i<30; i++) {  //considering the first 30 files

        // Open the image
        image = cv::imread(filelist[i],0);

        // Get the chessboard corners
        bool found = cv::findChessboardCorners(
                    image, boardSize, imageCorners);

        // Get subpixel accuracy on the corners
        cv::cornerSubPix(image, imageCorners,
                         cv::Size(5,5),
                         cv::Size(-1,-1),
                         cv::TermCriteria(cv::TermCriteria::MAX_ITER +
                                          cv::TermCriteria::EPS,
                                          30,		// max number of iterations
                                          0.1));     // min accuracy

        // If we have a good board, add it to our data
        if (imageCorners.size() == boardSize.area()) {

            // Add image and scene points from one view
            addPoints(imageCorners, objectCorners);
            successes++;
        }

        //Draw the corners
        cv::drawChessboardCorners(image, boardSize, imageCorners, found);
        data.push_back (image);
    }
    return data;
}

// Add scene points and corresponding image points
void CameraCalibration::addPoints(const std::vector<cv::Point2f>& imageCorners, const std::vector<cv::Point3f>& objectCorners){

    // 2D image points from one view
    imagePoints.push_back(imageCorners);
    // corresponding 3D scene points
    objectPoints.push_back(objectCorners);
}

// Calibrate the camera
double CameraCalibration::calibrate(cv::Size imageSize){
    // undistorter must be reinitialized
    mustInitUndistort= true;

    //Output rotations and translations
    std::vector<cv::Mat> rvecs, tvecs;

    // start calibration
    return cv::calibrateCamera(objectPoints, // the 3D points
                            imagePoints,  // the image points
                            imageSize,    // image size
                            cameraMatrix, // output camera matrix
                            distCoeffs,   // output distortion matrix
                            rvecs, tvecs, // Rs, Ts
                            flag);        // set options
}

// Set the calibration flag
void CameraCalibration::setCalibrationFlag(bool radial8CoeffEnabled, bool tangentialParamEnabled){

    // Set the flag used in cv::calibrateCamera()
    flag = 0;
    if (!tangentialParamEnabled) flag += CV_CALIB_ZERO_TANGENT_DIST;
    if (radial8CoeffEnabled) flag += CV_CALIB_RATIONAL_MODEL;
}

// Remove distortion in an image (after calibration)
cv::Mat CameraCalibration::remap(const cv::Mat &image){


    if (mustInitUndistort) { // called once per calibration

        cv::initUndistortRectifyMap(
                    cameraMatrix,  // computed camera matrix
                    distCoeffs,    // computed distortion matrix
                    cv::Mat(),     // optional rectification (none)
                    cv::Mat(),     // camera matrix to generate undistorted
                    //cv::Size(640,480),
                    image.size(),  // size of undistorted
                    CV_32FC1,      // type of output map
                    map1, map2);   // the x and y mapping functions

        mustInitUndistort= false;
    }

    // Apply mapping functions
    cv::remap(image, undistorted, map1, map2,
              cv::INTER_LINEAR); // interpolation type

    return undistorted;
}

// Getters
cv::Mat CameraCalibration::getCameraMatrix() {
    return cameraMatrix;
}
cv::Mat CameraCalibration::getDistCoeffs() {
    return distCoeffs;
}
