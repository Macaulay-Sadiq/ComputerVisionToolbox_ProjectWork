#include "face_detection.h"
#include <unistd.h>
#define CAFFE




Face::Face(){


}


Mat Face::faceRecognition(Mat frameOpenCVDNN){
    // Adapted from: https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/face_detection_opencv_dnn.cpp


    const size_t inWidth = 300;
    const size_t inHeight = 300;
    const double inScaleFactor = 1.0;
    const float confidenceThreshold = 0.7;
    const cv::Scalar meanVal(104.0, 177.0, 123.0);

    string caffeConfigFile = "../QtOpencvTutorial/models/deploy.prototxt";
    string caffeWeightFile = "../QtOpencvTutorial/models/res10_300x300_ssd_iter_140000_fp16.caffemodel";

    Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);

    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;
#ifdef CAFFE
    cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
#else
    cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);
#endif

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > confidenceThreshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
        }
    }
    return  frameOpenCVDNN;
}


Mat Face::faceDetectionWithClassifier(Mat frame){

    // Adapted from: https://www.geeksforgeeks.org/opencv-c-program-face-detection/

    CascadeClassifier nestedCascade;
    CascadeClassifier cascade;
    double scale=1;

    // Load classifiers from "opencv/data/haarcascades" directory
    nestedCascade.load( "../QtOpencvTutorial/models/haarcascade_eye_tree_eyeglasses.xml" ) ;

    // Change path before execution
    cascade.load( "../QtOpencvTutorial/models/haarcascade_frontalface_alt2.xml" ) ;

    vector<Rect> faces, faces2;
    Mat gray = frame.clone ();
    Mat smallImg;

    if (frame.type()!=CV_8UC1)
        cvtColor( frame, gray, COLOR_BGR2GRAY ); // Convert to Gray Scale
    double fx = 1 / scale;

    // Resize the Grayscale Image
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    // Detect faces of different sizes using cascade classifier
    cascade.detectMultiScale( smallImg, faces, 1.1,
                              2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    // Draw circles around the faces
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = Scalar(255, 0, 0); // Color for Drawing tool
        int radius;

        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
            circle( frame, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( frame, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
                       cvPoint(cvRound((r.x + r.width-1)*scale),
                               cvRound((r.y + r.height-1)*scale)), color, 3, 8, 0);
        if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg( r );

        // Detection of eyes int the input image
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects, 1.1, 2,
                                        0|CASCADE_SCALE_IMAGE, Size(30, 30) );

        // Draw circles around eyes
        for ( size_t j = 0; j < nestedObjects.size(); j++ )
        {
            Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.25*scale);
            circle( frame, center, radius, color, 3, 8, 0 );
        }
    }
    return frame;
}

static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}


static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    chdir("../QtOpencvTutorial");
    std::ifstream file(filename.c_str(), ifstream::in);

    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, CV_LOAD_IMAGE_GRAYSCALE));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int Face::faceRecognitionWithPCA(Mat testSample){

    /*
     * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
     * Released to public domain under terms of the BSD Simplified license.
     *
     * Redistribution and use in source and binary forms, with or without
     * modification, are permitted provided that the following conditions are met:
     *   * Redistributions of source code must retain the above copyright
     *     notice, this list of conditions and the following disclaimer.
     *   * Redistributions in binary form must reproduce the above copyright
     *     notice, this list of conditions and the following disclaimer in the
     *     documentation and/or other materials provided with the distribution.
     *   * Neither the name of the organization nor the names of its contributors
     *     may be used to endorse or promote products derived from this software
     *     without specific prior written permission.
     *
     *   See <http://www.opensource.org/licenses/bsd-license>
     */

    // Get the path to CSV files containing the labels.

    string fn_csv = "../QtOpencvTutorial/labels.csv";

    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;

//    reading training image and label from .esv file
    read_csv(fn_csv, images, labels);

    int height = images[0].rows;

// check for test input color space
    cv::Mat gray = testSample.clone();
    if (testSample.type() != CV_8UC1){
        cvtColor(testSample, gray, CV_BGR2GRAY);
    }
    int num_components = 10;
    double threshold = 10.0;

    // cresting an object of the model
    Ptr<FaceRecognizer> model = EigenFaceRecognizer::create(num_components, threshold);

    model->train(images, labels);

    // make predictions of label
    int predictedLabel = model->predict(gray);

    return predictedLabel;
}


Mat Face::faceRecognitionWithFisherFace(Mat frame){
    chdir("../QtOpencvTutorial");

    // Get the path to CSV files containing the labels.
    string fn_csv = "labels.csv";


    // Get the path to your CSV:
    string fn_haar =  "models/haarcascade_frontalcatface.xml" ;

    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<int> labels;

    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
    read_csv(fn_csv, images, labels);

    // size AND we need to reshape incoming faces to this size:
    int im_width = images[0].cols;
    int im_height = images[0].rows;

    // Create a FaceRecognizer and train it on the given images:
    //Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
  //  model->train(images, labels);


    // That's it for learning the Face Recognition model. You now
    // need to create the classifier for the task of Face Detection.
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);

    // Clone the current frame:
    Mat original = frame.clone();

    // Convert the current frame to grayscale:
    Mat gray;
    cvtColor(original, gray, CV_BGR2GRAY);

    // Find the faces in the frame:
    vector< Rect_<int> > faces;
    haar_cascade.detectMultiScale(gray, faces);


    for(int i = 0; i < faces.size(); i++) {
        // Process face by face:
        Rect face_i = faces[i];

        // Crop the face from the image. So simple with OpenCV C++:
        Mat face = gray(face_i);


        Mat face_resized;
        cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
        // Now perform the prediction, see how easy that is:
       // int prediction = model->predict(face_resized);

        // And finally write all we've found out to the original image!
        // First of all draw a green rectangle around the detected face:
        rectangle(original, face_i, CV_RGB(0, 255,0), 1);

        // Create the text we will annotate the box with:
        //string box_text = format("Prediction = %d", prediction);

        // Calculate the position for annotated text (make sure we don't
        // put illegal values in there):
        int pos_x = std::max(face_i.tl().x - 10, 0);
        int pos_y = std::max(face_i.tl().y - 10, 0);
        // And now put it into the image:

        //putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
    }


    return original;
}

int Face::faceRecognitionWithLBP(Mat testSample){


    /*
     * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
     * Released to public domain under terms of the BSD Simplified license.
     *
     * Redistribution and use in source and binary forms, with or without
     * modification, are permitted provided that the following conditions are met:
     *   * Redistributions of source code must retain the above copyright
     *     notice, this list of conditions and the following disclaimer.
     *   * Redistributions in binary form must reproduce the above copyright
     *     notice, this list of conditions and the following disclaimer in the
     *     documentation and/or other materials provided with the distribution.
     *   * Neither the name of the organization nor the names of its contributors
     *     may be used to endorse or promote products derived from this software
     *     without specific prior written permission.
     *
     *   See <http://www.opensource.org/licenses/bsd-license>
     */


        // Get the path to your CSV.
        string fn_csv = "../QtOpencvTutorial/labels.csv";
        // These vectors hold the images and corresponding labels.
        vector<Mat> images;
        vector<int> labels;

        // Read in the data. This can fail if no valid
        // input filename is given.

        read_csv(fn_csv, images, labels);

        //
        Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
        model->train(images, labels);

        // check for test input color space
            cv::Mat gray = testSample.clone();
            if (testSample.type() != CV_8UC1){
                cvtColor(testSample, gray, CV_BGR2GRAY );
            }

        // The following line predicts the label of a given
        // test image:
        int predictedLabel = model->predict(gray);


//        model->setThreshold(0.0);
//        // Now the threshold of this model is set to 0.0. A prediction
//        // now returns -1, as it's impossible to have a distance below
//        // it
//        predictedLabel = model->predict(testSample);
//        cout << "Predicted class = " << predictedLabel << endl;
//        // Show some informations about the model, as there's no cool

        return predictedLabel;
}









