#ifndef OPENCVWORKER_H
#define OPENCVWORKER_H

#include <QObject>
#include <QImage>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/stitching.hpp"
#include "compute_fmatrix.h"
#include "find_matches.h"
#include "camera_calibration.h"
#include "extract_features.h"


#include <QWidget>
#include <QPixmap>
#include <string>

class OpenCVWorker : public QWidget
{
    Q_OBJECT

private:
    cv::Mat _frameOriginal;
    cv::Mat fMat7point;
    cv::Mat fMat8point;
    cv::Mat fMatRANSAC;
    cv::Mat fMatHomography;

    cv::Mat imgLogo;
    cv::Mat _frameProcessed;
    cv::VideoCapture *cap;
    QImage _outputImage;
    QString _inputImage;
    QString _inputVideo;
    cv::Mat cameraMatrix;
    cv::Mat image2;
    cv::Mat image1;
    cv::Mat match;
    cv::Mat saltpepper_noise;
    bool status;
    bool toggleImageSource;
    bool toggleStream;
    bool toggleCam;
    bool toggleVideostrem;
    bool binaryThresholdEnable;
    bool toggleDisplayLogo;
    bool toggleSaltPepper;
    bool toggleHSV;
    bool toggleRGB;
    bool toggleLAB;
    bool toggleYCrCb;
    bool toggleGRAY;
    bool toggleHistogram;
    bool toggledMorphology;
    bool toggleHistEqual;
    bool toggleBlur;
    bool toggleSobel;
    bool toggleLaplacian;
    bool toggleCanny;
    bool toggleHoughLines;
    bool toggleHoughCircles;
    bool toggleContour;
    bool toggleShapeDescriptor;
    bool toggleHarrisCorner;
    bool toggleCalibration;
    bool toggleCalibCapture;
    bool toggleShowUdistort;
    bool toggleSURF;
    bool toggleFAST;
    bool toggleSIFT;
    bool toggleMatch;
    bool toggleFmatrix;
    bool toggleHomogrphy;
    bool toggle7point;
    bool toggle8point;
    bool toggleRANSAC;
    bool toggleStitch;
    bool toggleFacedetection;
    bool toggleFacecascade;
    bool toggleFRwithPCA;
    bool toggleFRwithFisherF;
    bool toggleFRwithLBP;
    int blurOperation;
    int binaryThreshhold;
    int structElemt;
    int operationType;
    int xOrigin;
    int yOrigin;
    int fmMethod;



    void checkIfDeviceAlreadyOpen(const int dev);

public:
    explicit OpenCVWorker(QWidget *parent = nullptr);
    ~OpenCVWorker();
    void displayLogo();
    void saltAndPepperNoise(cv::Mat img1);
    void convertToHSV(cv::Mat img2);
    void convertToLAB(cv::Mat img3);
    void convertToRGB(cv::Mat img4);
    void convertToYCrCb(cv::Mat img5);
    void getsecondImage();



signals:
    void sendProcessedFrame(QImage processedFrame);
    void sendStatus(QString msg, int cod);
    void sendFirstFrame(QImage img);
    void sendFMatrix(QString);
    void sendLabel(int);


public slots:
    void receiveProcessedGrabFrame();
    void receiveSetup(const int device);
    void receiveToggleStream();
    void receiveEnableBinaryThreshold();
    void receiveBinaryThreshold(int threshold);
    void receiveGrabFrame();
    void receiveImage();
    void receiveVideo();
    void receiveTriggerToOpenWebCam();
    void receiveEndProgram();
    void receiveLogoToggle();
    void reseiveLogoPose1();
    void reseiveLogoPose2();
    void reseiveLogoPose3();
    void reseiveLogoPose4();
    void receiveLogoXPose(int XPose);
    void receiveLogoYPose(int yPose);
    void receiveSaltandPepperTog();
    void receiveHSV();
    void receiveRGB();
    void receiveYCrCb();
    void receiveLAB();
    void receiveGRAY();
    void receiceHistogram();
    void receiveMorphology();
    void receiveRectangle();
    void receiveCross();
    void receiveEllips();
    void receiveOpening();
    void receiveClosing();
    void receiveErode();
    void receiveDilate();
    void receiveHistEqual();
    void receiveBlure();
    void receiveGaussian();
    void receiveHomogeneous();
    void receiveMedian();
    void receiveBilateral();
    void receiveSobel();
    void receiveLaplacian();
    void receiveCanny();
    void receiveHoughLines();
    void receiveHoughCircles();
    void receiveContour();
    void receiveShapeDescriptor();
    void receiveHarris();
    void receiveCameraCalibration();
    void receiveCalibCapture();
    void receiveShowundistort();
    void receive7Point();
    void receive8Point();
    void receiveRANSAC();
    void receiveSIFT();
    void receiveFAST();
    void receiveSURF();
    void receiveMatch();
    void receiveHomography();
    void receiveImageStitch();
    void receiveFaceDetection();
    void receiveFaceCascade();
    void receiveFaceRecgWithPCA();
    void receiveFaceRecgWithFisherF();
    void receiveFaceRecgWithLBP();

};

#endif // OPENCVWORKER_H
