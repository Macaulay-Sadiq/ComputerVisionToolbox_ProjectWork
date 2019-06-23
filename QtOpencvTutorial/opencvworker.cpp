#include "opencvworker.h"
#include <QFileDialog>
#include <opencv2/imgproc/imgproc.hpp>
#include "toolsbox.h"
#include <unistd.h>
#include <iostream>
#include <cstdio>
#include <sys/stat.h>
#include <dirent.h>
#include <QString>
#include <iomanip>
#include "face_detection.h"


toolsBox tools;
CameraCalibration camCalibrnt;
Features features;
FMatrix fMat;
FindMatches match;
Face fc;



OpenCVWorker::OpenCVWorker(QWidget *parent) : QWidget(parent),toggledMorphology(false),
    status(false), toggleStream(false),toggleHistogram(false),structElemt(0),operationType(0),
    xOrigin(0), yOrigin(0), binaryThreshhold(20), binaryThresholdEnable(false), toggleGRAY(false),
    toggleCam(false), toggleImageSource(true), toggleVideostrem(false), toggleDisplayLogo(false),
    toggleSaltPepper(false), toggleHSV(false), toggleRGB(false),toggleLAB(false),toggleYCrCb(false),
    toggleHistEqual(false),toggleBlur(false),blurOperation(0),toggleSobel(false),toggleLaplacian(false),
    toggleCanny(false),toggleHoughLines(false),toggleHoughCircles(false), toggleContour(false),
    toggleShapeDescriptor(false),toggleHarrisCorner(false),toggleCalibration(false),toggleCalibCapture(false),
    toggleShowUdistort(false),toggleSURF(false),toggleFAST(false),toggleSIFT(false),toggleMatch(false),toggleStitch(false), fmMethod(1),toggleFmatrix(false),toggleHomogrphy(false), toggle7point(false), toggle8point(false),toggleRANSAC(false), toggleFacedetection(false), toggleFacecascade(false), toggleFRwithPCA(false),toggleFRwithFisherF(false),toggleFRwithLBP(false)
{
    cap = new cv::VideoCapture();

}

OpenCVWorker::~OpenCVWorker(){
    if (cap->isOpened()) cap->release();
    delete cap;
}

void OpenCVWorker::displayLogo(){

    QString logo = QFileDialog::getOpenFileName(
                this,
                tr("Logo"),
                "../QtOpencvTutorial",
                tr("Image Files (*.png  *.jpg  *.bmp *.tif)")
                );

    imgLogo = cv::imread(logo.toStdString(), CV_LOAD_IMAGE_COLOR);
    cv::Size size(100,100);
    cv::resize(imgLogo, imgLogo, size);
}

void OpenCVWorker::receiveImage(){
    cap->release();
    _frameOriginal.release ();
    _frameProcessed.release();
    toggleImageSource = true;

    if(toggleVideostrem){
        toggleVideostrem = false;
    }
    if(toggleMatch){
        toggleMatch = false;
    }
    if (toggleCam){
        toggleCam = false;
    }
    if(toggleCalibration){
        toggleCalibration = false;
    }

    _inputImage = QFileDialog::getOpenFileName(
                this,
                tr("Image diretory"),
                "../QtOpencvTutorial",
                tr("Image Files (*.png  *.jpg  *.bmp *.tif *.pgm)")
                );

    saltpepper_noise = cv::Mat::zeros(_frameOriginal.rows, _frameOriginal.cols,CV_8U);
    randu(saltpepper_noise,0,255);

}


void OpenCVWorker::receiveVideo(){

    cap->release();
    _frameOriginal.release ();
    _frameProcessed.release();
    toggleVideostrem = true;

    if(toggleImageSource){
        toggleImageSource = false;
    }
    if(toggleMatch){
        toggleMatch = false;
    }
    if (toggleCam){
        toggleCam = false;
    }
    if(toggleCalibration){
        toggleCalibration = false;
    }


    _inputVideo = QFileDialog::getOpenFileName(
                this,
                tr("Video diretory"),
                "../QtOpencvTutorial",
                tr("Image Files (*.mp4 *.avi)")
                );
    cap->open (_inputVideo.toStdString ());
}

void OpenCVWorker::receiveTriggerToOpenWebCam(){
    cap->release();
    _frameOriginal.release ();
    _frameProcessed.release();
    toggleCam =true;

    if(toggleImageSource){
        toggleImageSource = false;
    }
    if (toggleVideostrem){
        toggleVideostrem = false;
    }
    if(toggleCalibration)
        toggleCalibration = false;
    if(toggleMatch){
        toggleMatch = false;
}
    cap->open(0);

}


void OpenCVWorker::receiveGrabFrame(){

    if(toggleVideostrem){

        (*cap) >> _frameOriginal;
        if(_frameOriginal.empty()) return;
    }
    else if(toggleCalibration){
        _frameOriginal = cv::imread(_inputImage.toStdString(), cv::IMREAD_COLOR);
        if(_frameOriginal.empty()) return;
    }

    else if (toggleImageSource) {
        _frameOriginal = cv::imread(_inputImage.toStdString(), cv::IMREAD_COLOR);
        if (_frameOriginal.empty()) return;
    }

    else if(toggleCam) {
        (*cap) >>  _frameOriginal;
        if(_frameOriginal.empty()) return;
    }

    cv::cvtColor(_frameOriginal,_frameOriginal,  CV_BGR2RGB );

    QImage image((const uchar*) _frameOriginal.data,_frameOriginal.cols,_frameOriginal.rows,_frameOriginal.step,QImage::Format_RGB888);
    emit sendFirstFrame(image);
}

void OpenCVWorker::receiveProcessedGrabFrame(){


    if(!toggleStream) return;

    if (_frameOriginal.empty()) return;

    if(!toggleCalibration){
        _frameProcessed = _frameOriginal;
    }

        cv::Mat newImg =   _frameProcessed;
        if(toggleGRAY)
            cv::cvtColor(newImg, newImg,   cv::COLOR_RGB2GRAY);
        if(binaryThresholdEnable){
            cv::cvtColor(newImg, newImg,   cv::COLOR_RGB2GRAY);
            cv::threshold(newImg, newImg,  binaryThreshhold, 255, cv::THRESH_BINARY);
        }
        if (toggleHistEqual)
            newImg = tools.histEqualization (newImg);
        if (toggleHoughLines)
            newImg = tools.HoughLines (newImg, binaryThreshhold);
        if (toggleHoughCircles)
            newImg = tools.HoughCircles (newImg, binaryThreshhold);
        if(toggleContour)
            newImg = tools.imageContours (newImg,binaryThreshhold);
        if(toggleShapeDescriptor)
            newImg = tools.shapeDescriptor (newImg, binaryThreshhold);
        if (toggleHarrisCorner)
            newImg = tools.harrisCornerDetector (newImg, binaryThreshhold);
        if (toggleSURF)
            newImg = features.getSurfFeatures (newImg,binaryThreshhold);
        if(toggleFAST)
            newImg = features.getFastFeatures (newImg,binaryThreshhold);
        if (toggleSIFT)
            newImg = features.getSiftFeatures (newImg,binaryThreshhold);
        if(toggleFacedetection){
            newImg = fc.faceRecognition (newImg);
        }
        if (toggleFRwithFisherF){
             newImg = fc.faceRecognitionWithFisherFace (newImg);
        }
        if(toggleFacecascade){
            newImg = fc.faceDetectionWithClassifier(newImg);
        }
        if(toggleMatch)
            newImg = match;
        if (toggle7point)
            newImg = match;
        if (toggle8point)
            newImg = match;
        if (toggleRANSAC)
             newImg = match;
        if (toggleHomogrphy)
            newImg = match;
        if(toggleStitch)
            newImg = match;
        if(toggleSaltPepper)
            saltAndPepperNoise(newImg);
        if(toggleCanny)
            newImg = tools.cannyEdgeDetection (newImg, binaryThreshhold);
        if(toggleLaplacian)
            newImg = tools.laplacianOperation (newImg, binaryThreshhold);
        if(toggleSobel)
            newImg = tools.sobelOperation (newImg, binaryThreshhold);
        if (toggledMorphology)
            newImg = tools.morphologyOperation(newImg,structElemt,binaryThreshhold,operationType);
        if (toggleBlur)
            newImg = tools.imageBlur(newImg, blurOperation,binaryThreshhold);
        if(toggleHSV)
            convertToHSV (newImg);
        if(toggleLAB)
            convertToLAB (newImg);
        if(toggleYCrCb)
            convertToYCrCb (newImg);
        if(toggleRGB)
            convertToRGB (newImg);
        if(toggleHistogram)
            newImg =  tools.histogramOfImage (newImg);
        if (toggleHistEqual)
            newImg = tools.histEqualization (newImg);
        if(toggleDisplayLogo){
            cv::Mat newImg2 =   imgLogo;
            if (newImg2.empty()) return;
            if(binaryThresholdEnable){
                cv::cvtColor(newImg2, newImg2,   cv::COLOR_RGB2GRAY);
            }
            cv::Rect roi( cv::Point(xOrigin, yOrigin), newImg2.size());
            newImg2.copyTo( newImg(roi));
        }

        if (newImg.type()==CV_8UC1) {
            cv::cvtColor(newImg, newImg, cv::COLOR_GRAY2RGB);
        }

        QImage output((const uchar *)newImg.data, newImg.cols, newImg.rows, newImg.step, QImage::Format_RGB888);
        emit sendProcessedFrame(output);

}

void OpenCVWorker::receiveFaceRecgWithPCA (){
    toggleFRwithPCA = !toggleFRwithPCA;
    int label = fc.faceRecognitionWithPCA (_frameProcessed);

    emit sendLabel(label);
}

void OpenCVWorker::receiveFaceRecgWithFisherF (){
    toggleFRwithFisherF = !toggleFRwithFisherF;
}

void OpenCVWorker::receiveFaceRecgWithLBP (){
    toggleFRwithLBP = !toggleFRwithLBP;
    int label = fc.faceRecognitionWithLBP (_frameProcessed);

    emit sendLabel(label);
}

void OpenCVWorker::receiveFaceDetection (){
    toggleFacedetection = !toggleFacedetection;
}
void OpenCVWorker::receiveFaceCascade (){
    toggleFacecascade = !toggleFacecascade;
}

void OpenCVWorker::receiveFAST(){
    toggleFAST = !toggleFAST;
}
void OpenCVWorker::receiveSIFT(){
    toggleSIFT = !toggleSIFT;
}
void OpenCVWorker::receiveSURF(){
    toggleSURF = !toggleSURF;
}
void OpenCVWorker::receiveMatch(){
    getsecondImage();
    toggleMatch = true;
    image1 = _frameOriginal;
    match = features.getMatchFeatures (image1,image2,binaryThreshhold);
}

void OpenCVWorker::getsecondImage(){
 QString   imgfile = QFileDialog::getOpenFileName(
                this,
                tr("Image to be Matched"),
                "../QtOpencvTutorial",
                tr("Image Files (*.png  *.jpg  *.bmp  *.tif  *.pgm)")
                );
  image2 = cv::imread(imgfile.toStdString(), cv::IMREAD_COLOR);

}




void OpenCVWorker::saltAndPepperNoise(cv::Mat Image){

    saltpepper_noise = cv::Mat::zeros(_frameOriginal.rows, _frameOriginal.cols,CV_8U);
    randu(saltpepper_noise,0,255);

    cv::Mat black = saltpepper_noise < 30;
    cv::Mat white = saltpepper_noise > 225;
    Image.setTo(255,white);
    Image.setTo(0,black);
}

void OpenCVWorker::convertToHSV(cv::Mat img){
    if (img.type()==CV_8UC1) return;
    cv::cvtColor(img, img,  cv::COLOR_RGB2HSV);
}

void OpenCVWorker::convertToRGB(cv::Mat img){
    if (img.type()==CV_8UC1)
        cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
}

void OpenCVWorker::convertToLAB(cv::Mat img){
    if (img.type()==CV_8UC1) return;
    cv::cvtColor(img, img,  cv::COLOR_RGB2Lab);
}

void OpenCVWorker::convertToYCrCb(cv::Mat img){
    if (img.type()==CV_8UC1) return;
    cv::cvtColor(img, img,  cv::COLOR_RGB2YCrCb);
}

void OpenCVWorker::checkIfDeviceAlreadyOpen(const int device){
    if (cap->isOpened()) cap->release();
    cap->open(device);
}

void OpenCVWorker::receiveSetup(const int device){

    checkIfDeviceAlreadyOpen(device);
    if(!cap->isOpened()){
        status = false;
        return;
    }
    status = true;
}

void OpenCVWorker::receiveToggleStream(){
    toggleStream = !toggleStream;
}

void OpenCVWorker::receiveEnableBinaryThreshold(){
    binaryThresholdEnable = !binaryThresholdEnable;
}

void OpenCVWorker::receiveBinaryThreshold(int threshold){
    binaryThreshhold = threshold;
}

void OpenCVWorker::receiveEndProgram(){
    cap->release();
    _frameOriginal.release ();
    _frameProcessed.release();
}

void OpenCVWorker::receiveLogoToggle(){
    toggleDisplayLogo = !toggleDisplayLogo;
    if(toggleDisplayLogo == true){
        displayLogo();
    }
}

void OpenCVWorker::reseiveLogoPose1(){
    yOrigin = _frameProcessed.rows - _frameProcessed.rows;
    xOrigin  = _frameProcessed.cols - 100;
}

void OpenCVWorker::reseiveLogoPose2(){
    yOrigin = _frameProcessed.rows -100;
    xOrigin  = _frameProcessed.cols - 100;
}

void OpenCVWorker::reseiveLogoPose3(){
    yOrigin = _frameProcessed.rows -100;
    xOrigin  = _frameProcessed.cols -  _frameProcessed.cols;
}

void OpenCVWorker::reseiveLogoPose4(){

    yOrigin = _frameProcessed.rows -_frameProcessed.rows;
    xOrigin  = _frameProcessed.cols -  _frameProcessed.cols;

}
void OpenCVWorker::receiveLogoYPose(int ypose){
    if(ypose > _frameProcessed.cols) return;
    yOrigin = ypose;
}

void OpenCVWorker::receiveLogoXPose(int xpose){
    if(xpose > _frameProcessed.rows) return;
    xOrigin = xpose;
}

void OpenCVWorker::receiveSaltandPepperTog(){
    toggleSaltPepper = !toggleSaltPepper;
}

void OpenCVWorker::receiveHSV (){
    toggleHSV = !toggleHSV;
}

void OpenCVWorker::receiveLAB (){
    toggleLAB = !toggleLAB;
}

void OpenCVWorker::receiveRGB (){
    toggleRGB = !toggleRGB;
}

void OpenCVWorker::receiveYCrCb (){
    toggleYCrCb = !toggleYCrCb;
}

void OpenCVWorker::receiveGRAY (){
    toggleGRAY = !toggleGRAY;
}


void OpenCVWorker::receiceHistogram (){
    toggleHistogram = !toggleHistogram;
}

void OpenCVWorker::receiveMorphology(){
    toggledMorphology = !toggledMorphology;
}

void OpenCVWorker::receiveRectangle(){
    structElemt = 0;
}
void OpenCVWorker::receiveCross(){
    structElemt = 1;
}
void OpenCVWorker::receiveEllips(){
    structElemt = 2;
}
void OpenCVWorker::receiveOpening(){
    operationType = 0;
}
void OpenCVWorker::receiveClosing(){
    operationType = 1;
}
void OpenCVWorker::receiveErode(){
    operationType = 2;
}
void OpenCVWorker::receiveDilate(){
    operationType = 3;
}
void OpenCVWorker::receiveHistEqual(){
    toggleHistEqual = !toggleHistEqual;
}
void OpenCVWorker::receiveBlure(){
    toggleBlur = !toggleBlur;
}

void OpenCVWorker::receiveGaussian(){
    blurOperation = 1;
}
void OpenCVWorker::receiveHomogeneous(){
    blurOperation = 0;
}
void OpenCVWorker::receiveMedian(){
    blurOperation = 2;
}
void OpenCVWorker::receiveBilateral(){
    blurOperation = 3;
}
void OpenCVWorker::receiveSobel (){
    toggleSobel = !toggleSobel;
}

void OpenCVWorker::receiveLaplacian (){
    toggleLaplacian = !toggleLaplacian;
}

void OpenCVWorker::receiveCanny (){
    toggleCanny = !toggleCanny;
}

void OpenCVWorker::receiveHoughCircles (){
    toggleHoughCircles = !toggleHoughCircles;
}

void OpenCVWorker::receiveHoughLines (){
    toggleHoughLines = !toggleHoughLines;
}

void OpenCVWorker::receiveContour (){
    toggleContour = !toggleContour;
}

void OpenCVWorker::receiveShapeDescriptor (){
    toggleShapeDescriptor = !toggleShapeDescriptor;
}

void OpenCVWorker::receiveHarris (){
    toggleHarrisCorner = !toggleHarrisCorner;
}

void OpenCVWorker::receiveCalibCapture(){
    toggleCalibCapture = !toggleCalibCapture;
}
void OpenCVWorker::receiveShowundistort(){
    toggleShowUdistort = !toggleShowUdistort;
}



void OpenCVWorker::receiveCameraCalibration(){
    chdir("../QtOpencvTutorial");
    toggleCalibration = true;

    cap->release();
    _frameOriginal.release ();
    _frameProcessed.release();

    if (toggleCam){
        toggleCam = false;
    }

    if(toggleImageSource){
        toggleImageSource = false;
    }
    if (toggleVideostrem){
        toggleVideostrem = false;
    }
    if(toggleMatch){
        toggleMatch = false;
    }

std::vector<cv::Mat> data;
    _inputImage = QFileDialog::getOpenFileName(
                this,
                tr("Distorted Image"),
                "../QtOpencvTutorial/imageFolder",
                tr("Image Files (*.png  *.jpg  *.bmp *.tif)")
                );

    cv::String path("imageFolder/*.jpg"); //select only jpg
    std::vector<cv::String> fn;
    cv::glob(path,fn,true); // recurse

    cv::Mat image = cv::imread(fn[1]);
    cv::Size boardSize(6,4);
    data = camCalibrnt.addChessboardPoints (fn,boardSize);
    for(auto i : data){
         _frameProcessed = i;
        cv::waitKey(100);
    }
    camCalibrnt.calibrate(image.size());
    //_frameOriginal = cv::imread(fn[9]);
    _frameProcessed = camCalibrnt.remap(_frameOriginal);    //undistorted image
    cameraMatrix = camCalibrnt.getCameraMatrix();

    std::ostringstream oss;
    oss << std::endl << std::setw(12) <<cameraMatrix.at<double>(0,0) << " " << std::setw(12) <<cameraMatrix.at<double>(0,1) << " " << std::setw(12) <<cameraMatrix.at<double>(0,2) << std::endl <<
           std::setw(12) << cameraMatrix.at<double>(1,0) << " " << std::setw(12) << cameraMatrix.at<double>(1,1) << " " << std::setw(12) << cameraMatrix.at<double>(1,2) << std::endl <<
           std::setw(12) << cameraMatrix.at<double>(2,0) << " " << std::setw(12) << cameraMatrix.at<double>(2,1) << " " << std::setw(12) << cameraMatrix.at<double>(2,2) << std::endl;
        QString strFmatrix(oss.str().c_str());

        emit sendFMatrix (strFmatrix);

}


void OpenCVWorker::receive7Point(){
    toggle7point = !toggle7point;
    getsecondImage();
    image1 = _frameOriginal;
    fMat.getFundamental7PointAndDrawEpilines(image1, image2,match,fMat7point);

    if(fMat7point.empty ()) return;
    std::ostringstream oss;
        oss << std::endl << std::setw(12) <<fMat7point.at<double>(0,0) << " " << std::setw(12) <<fMat7point.at<double>(0,1) << " " << std::setw(12) <<fMat7point.at<double>(0,2) << std::endl <<
               std::setw(12) << fMat7point.at<double>(1,0) << " " << std::setw(12) << fMat7point.at<double>(1,1) << " " << std::setw(12) << fMat7point.at<double>(1,2) << std::endl <<
               std::setw(12) << fMat7point.at<double>(2,0) << " " << std::setw(12) << fMat7point.at<double>(2,1) << " " << std::setw(12) << fMat7point.at<double>(2,2) << std::endl;
            QString strFmatrix(oss.str().c_str());
            emit sendFMatrix (strFmatrix);
}
void OpenCVWorker::receive8Point(){
    toggle8point = !toggle8point;
    getsecondImage();
    image1 = _frameOriginal;
    fMat.getFundamental8PointAndDrawEpilines(image1, image2,match,fMat8point);

    if(fMat8point.empty ()) return;
    std::ostringstream oss;
     oss << std::endl << std::setw(12) <<fMat8point.at<double>(0,0) << " " << std::setw(12) <<fMat8point.at<double>(0,1) << " " << std::setw(12) <<fMat8point.at<double>(0,2) << std::endl <<
           std::setw(12) << fMat8point.at<double>(1,0) << " " << std::setw(12) << fMat8point.at<double>(1,1) << " " << std::setw(12) << fMat8point.at<double>(1,2) << std::endl <<
           std::setw(12) << fMat8point.at<double>(2,0) << " " << std::setw(12) << fMat8point.at<double>(2,1) << " " << std::setw(12) << fMat8point.at<double>(2,2) << std::endl;
        QString strFmatrix(oss.str().c_str());

        emit sendFMatrix (strFmatrix);
}
void OpenCVWorker::receiveRANSAC(){

    toggleRANSAC = !toggleRANSAC;
    getsecondImage();
    image1 = _frameOriginal;
    fMat.getFundamentalRANSACAndDrawEpilines(image1, image2,match,fMatRANSAC);

    if(fMatRANSAC.empty ()) return;
    std::ostringstream oss;
    oss << std::endl << std::setw(12) <<fMatRANSAC.at<double>(0,0) << " " << std::setw(12) <<fMatRANSAC.at<double>(0,1) << " " << std::setw(12) <<fMatRANSAC.at<double>(0,2) << std::endl <<
           std::setw(12) << fMatRANSAC.at<double>(1,0) << " " << std::setw(12) << fMatRANSAC.at<double>(1,1) << " " << std::setw(12) << fMatRANSAC.at<double>(1,2) << std::endl <<
           std::setw(12) << fMatRANSAC.at<double>(2,0) << " " << std::setw(12) << fMatRANSAC.at<double>(2,1) << " " << std::setw(12) << fMatRANSAC.at<double>(2,2) << std::endl;
        QString strFmatrix(oss.str().c_str());

        emit sendFMatrix (strFmatrix);
}

void OpenCVWorker::receiveHomography(){
    toggleHomogrphy = !toggleHomogrphy;
    getsecondImage();
    image1 = _frameOriginal;
    fMat.getHomography (image1, image2,match,fMatHomography);

    if(fMatHomography.empty ()) return;
    std::ostringstream oss;
    oss << std::endl << std::setw(12) <<fMatHomography.at<double>(0,0) << " " << std::setw(12) <<fMatHomography.at<double>(0,1) << " " << std::setw(12) <<fMatHomography.at<double>(0,2) << std::endl <<
           std::setw(12) << fMatHomography.at<double>(1,0) << " " << std::setw(12) << fMatHomography.at<double>(1,1) << " " << std::setw(12) << fMatHomography.at<double>(1,2) << std::endl <<
           std::setw(12) << fMatHomography.at<double>(2,0) << " " << std::setw(12) << fMatHomography.at<double>(2,1) << " " << std::setw(12) << fMatHomography.at<double>(2,2) << std::endl;
        QString strFmatrix(oss.str().c_str());

        emit sendFMatrix (strFmatrix);
}

void OpenCVWorker::receiveImageStitch(){
    toggleStitch = !toggleStitch;

    std::vector<cv::Mat>images;
    getsecondImage();
    image1 = _frameOriginal;
    images.push_back (image1);
    images.push_back (image2);

    cv::Stitcher::Mode mode = cv::Stitcher::PANORAMA;

    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(mode);
    cv::Stitcher::Status status = stitcher->stitch(images, match);
}


