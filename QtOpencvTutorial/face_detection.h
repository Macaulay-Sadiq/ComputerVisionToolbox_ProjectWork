#ifndef FACE_H
#define FACE_H

#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include "opencv2/core/core.hpp"
//#include "opencv2/contrib/contrib.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;
using namespace cv::dnn;
using namespace cv::face;

class Face
{
private:


public:

    Face();
    Mat faceRecognition(Mat frameOpenCVDNN);
    Mat faceDetectionWithClassifier(Mat frame);
    int faceRecognitionWithPCA(Mat img);
    Mat faceRecognitionWithFisherFace(Mat frame);
    int faceRecognitionWithLBP(Mat testSample);

};

#endif // FACE_H
