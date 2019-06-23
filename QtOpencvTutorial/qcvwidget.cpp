#include "qcvwidget.h"
#include "ui_qcvwidget.h"
#include "opencvworker.h"
#include <QTimer>
#include <iostream>
#include <bits/stdc++.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <time.h>


QCVWidget::QCVWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::QCVWidget)
{
    toggleCapture = false;
    ui->setupUi(this);
    ui->labelView->setScaledContents(true);
    ui->labelView2->setScaledContents(true);
    ui->groupBoxMorphology->setEnabled (false);
    ui->groupBox_4->setEnabled (false);
    ui->groupBoxBlur->setEnabled (false);
    ui->textEditMsgBox->setText ("Messsage box...");

    ui->comboBox->addItem (tr("Browse"));
    ui->comboBox->addItem (tr("Video"));
    ui->comboBox->addItem (tr("Image"));
    ui->comboBox->addItem (tr("Cam"));

    ui->tools->addItem (tr("Image Processing"));
    ui->tools->addItem (tr("Show Logo"));
    ui->tools->addItem (tr("Salt & Pepper Noise"));
    ui->tools->addItem (tr("Histogram"));
    ui->tools->addItem (tr("Hist-Equalization"));
    ui->tools->addItem (tr("Morphology"));
    ui->tools->addItem (tr("Blure"));
    ui->tools->addItem (tr("Sobel"));
    ui->tools->addItem (tr("Laplacian"));
    ui->tools->addItem (tr("Canny"));
    ui->tools->addItem (tr("Hough Circles"));
    ui->tools->addItem (tr("Hough Lines"));
    ui->tools->addItem (tr("Contour"));
    ui->tools->addItem (tr("Shape Descriptor"));
    ui->tools->addItem (tr("Harris Corner"));



    ui->ComboBoxColorSpace->addItem (tr("Color Space"));
    ui->ComboBoxColorSpace->addItem (tr("LAB"));
    ui->ComboBoxColorSpace->addItem (tr("YCrCb"));
    ui->ComboBoxColorSpace->addItem (tr("HSV"));
    ui->ComboBoxColorSpace->addItem (tr("RGB"));
    ui->ComboBoxColorSpace->addItem (tr("GRAY"));
    ui->ComboBoxColorSpace->addItem (tr("BW"));

    ui->ComboBoxCmpVision->addItem (tr("Computer Vision"));
    ui->ComboBoxCmpVision->addItem (tr("FAST"));
    ui->ComboBoxCmpVision->addItem (tr("SURF"));
    ui->ComboBoxCmpVision->addItem (tr("SIFT"));
    ui->ComboBoxCmpVision->addItem (tr("Image Matching"));
    ui->ComboBoxCmpVision->addItem (tr("Camere Calibration"));
    ui->ComboBoxCmpVision->addItem (tr("7point FMatrix"));
    ui->ComboBoxCmpVision->addItem (tr("8point FMatrix"));
    ui->ComboBoxCmpVision->addItem (tr("RANSAC"));
    ui->ComboBoxCmpVision->addItem (tr("Homography"));
    ui->ComboBoxCmpVision->addItem (tr("Stitch Image"));
    ui->ComboBoxCmpVision->addItem (tr("Face Detection (DNN)"));
    ui->ComboBoxCmpVision->addItem (tr("FaceD with CClassifier"));
    ui->ComboBoxCmpVision->addItem (tr("FaceRecg with PCA"));
    ui->ComboBoxCmpVision->addItem (tr("FaceRecg with FisherF"));
    ui->ComboBoxCmpVision->addItem (tr("FaceD with LBP"));


    setup();
}

QCVWidget::~QCVWidget()
{
    thread->quit();
    while(thread->isFinished());
    delete thread;
    delete ui;
    delete event;
}



void QCVWidget::setup(){
    thread = new QThread();
    OpenCVWorker *Worker = new  OpenCVWorker();
    QTimer *workerTrigger  = new QTimer();
    workerTrigger->setInterval(100);
    workerTrigger->start();

    connect(workerTrigger, SIGNAL(timeout()), Worker, SLOT(receiveGrabFrame()));
    connect(workerTrigger, SIGNAL(timeout()), Worker, SLOT(receiveProcessedGrabFrame()));

    connect(this, SIGNAL(sendSetup(int)), Worker, SLOT(receiveSetup(int)));
    connect(this, SIGNAL(sendToggleStream()), Worker, SLOT(receiveToggleStream()));

    connect(ui->pushButtonPlay, SIGNAL(clicked(bool)), this, SLOT(receiveToggleStream()));
    connect(ui->spinBoxBinarThreashold, SIGNAL(valueChanged(int)), Worker, SLOT(receiveBinaryThreshold(int)));
    connect (ui->radioButton, SIGNAL (clicked()), Worker, SLOT (reseiveLogoPose1()));
    connect (ui->radioButton_2, SIGNAL (clicked()), Worker, SLOT (reseiveLogoPose2()));
    connect (ui->radioButton_3, SIGNAL (clicked()), Worker, SLOT (reseiveLogoPose3()));
    connect (ui->radioButton_4, SIGNAL (clicked()), Worker, SLOT (reseiveLogoPose4()));
    connect (ui->spinBoxXlogoPose,SIGNAL (valueChanged(int)), Worker, SLOT (receiveLogoXPose(int)));
    connect (ui->spinBoxYlogoPose,SIGNAL (valueChanged(int)), Worker, SLOT (receiveLogoYPose(int)));
    connect (ui->radioButtonRct,SIGNAL (clicked()), Worker, SLOT (receiveRectangle()));
    connect (ui->radioButtonCross,SIGNAL (clicked()),Worker,SLOT (receiveCross()));
    connect (ui->radioButtonEllipse,SIGNAL (clicked()),Worker,SLOT (receiveEllips()));
    connect (ui->radioButtonOpen,SIGNAL (clicked()),Worker,SLOT (receiveOpening()));
    connect (ui->radioButtonClose,SIGNAL (clicked()),Worker,SLOT (receiveClosing()));
    connect (ui->radioButtonErode,SIGNAL (clicked()),Worker,SLOT (receiveErode()));
    connect (ui->radioButtonDilate,SIGNAL (clicked()),Worker,SLOT (receiveDilate()));
    connect(ui->radioButtonHomogeneous,SIGNAL(clicked()), Worker,SLOT(receiveHomogeneous()));
    connect(ui->radioButtonGaussian,SIGNAL(clicked()), Worker,SLOT(receiveGaussian()));
    connect(ui->radioButtonMedian,SIGNAL(clicked()), Worker,SLOT(receiveMedian()));
    connect(ui->radioButtonBilateral,SIGNAL(clicked()), Worker,SLOT(receiveBilateral()));
    connect (ui->radioButtonPnt7, SIGNAL (clicked()), Worker,SLOT (receive7Point()));
    connect (ui->radioButtonPnt8, SIGNAL (clicked()), Worker,SLOT (receive8Point()));
    connect (ui->radioButtonRNSK, SIGNAL (clicked()), Worker,SLOT (receiveRANSAC()));

    connect(Worker, SIGNAL(sendProcessedFrame(QImage)), this, SLOT(receiveFrame(QImage)));
    connect(Worker, SIGNAL(sendFirstFrame(QImage)), this, SLOT(receiveOriginalFrame(QImage)));
    connect (Worker,SIGNAL (sendFMatrix(QString)), this, SLOT (receiveFmatrix(QString)));
    connect (Worker,SIGNAL (sendLabel(int)), this, SLOT (receiveLabel(int)));


    connect (ui->comboBox, SIGNAL(activated(int)), this,SLOT (receiveVideoTrigger()));
    connect (this, SIGNAL(sendVideoFile()), Worker, SLOT (receiveVideo()));
    connect (this, SIGNAL (sendImageFile()), Worker, SLOT (receiveImage()));
    connect (this, SIGNAL (sendTriggerCam()), Worker, SLOT (receiveTriggerToOpenWebCam()));
    connect (this, SIGNAL (sendEnd()), Worker, SLOT (receiveEndProgram()));

    connect (ui->pushButtonQuiteUI, SIGNAL(clicked(bool)),this, SLOT(closeInterface()));

    connect (ui->pushButtonCapture, SIGNAL (clicked(bool)),this, SLOT (saveImage()));
    connect (ui->pushButtonUndistort, SIGNAL (clicked(bool)), Worker,SLOT (receiveShowundistort()));
    connect (ui->checkBoxFmatrix, SIGNAL (clicked(bool)), this, SLOT (receiveString()));
    connect (ui->checkBoxFaceRecg, SIGNAL (clicked(bool)), this, SLOT (receivedLabel()));


    connect (ui->ComboBoxCmpVision, SIGNAL(activated(int)), this,SLOT (receiveCmpVision()));
    connect(this, SIGNAL(sendCalibration()), Worker, SLOT(receiveCameraCalibration()));
    connect(this, SIGNAL(sendSURF()), Worker, SLOT(receiveSURF()));
    connect(this, SIGNAL(sendSIFT()), Worker, SLOT(receiveSIFT()));
    connect(this, SIGNAL(sendFAST()), Worker, SLOT(receiveFAST()));
    connect(this, SIGNAL(sendMatch()), Worker, SLOT(receiveMatch()));
    connect(this, SIGNAL(send7point()), Worker, SLOT(receive7Point()));
    connect(this, SIGNAL(send8point()), Worker, SLOT(receive8Point()));
    connect(this, SIGNAL(sendRANSAC()), Worker, SLOT(receiveRANSAC()));
    connect(this, SIGNAL(sendHomography()), Worker, SLOT(receiveHomography()));
    connect(this, SIGNAL(sendStitch()), Worker, SLOT(receiveImageStitch()));
    connect(this, SIGNAL(sendFaceDetection()), Worker, SLOT(receiveFaceDetection()));
    connect(this, SIGNAL(sendFDwithCascadeCl()), Worker, SLOT(receiveFaceCascade()));
    connect(this, SIGNAL(sendPCA()), Worker, SLOT(receiveFaceRecgWithPCA()));
    connect(this, SIGNAL(sendFishers()), Worker, SLOT(receiveFaceRecgWithFisherF()));
    connect(this, SIGNAL(sendLBP()), Worker, SLOT(receiveFaceRecgWithLBP()));



    connect (ui->tools, SIGNAL(activated(int)), this,SLOT (receiveToolOptions()));
    connect (this, SIGNAL (sendLogoToggle()), Worker, SLOT (receiveLogoToggle()));
    connect (this, SIGNAL (sendSaltPepper()), Worker, SLOT (receiveSaltandPepperTog()));
    connect (this, SIGNAL (sendHistogram()), Worker, SLOT (receiceHistogram ()));
    connect (this, SIGNAL (sendHistEqual()), Worker, SLOT (receiveHistEqual()));
    connect (this, SIGNAL (sendMorphology()), Worker, SLOT (receiveMorphology()));
    connect(this, SIGNAL(sendBlure()), Worker, SLOT(receiveBlure()));
    connect(this, SIGNAL(sendSobel()), Worker, SLOT(receiveSobel()));
    connect(this, SIGNAL(sendLaplacian()), Worker, SLOT(receiveLaplacian()));
    connect(this, SIGNAL(sendCanny()), Worker, SLOT(receiveCanny()));
    connect(this, SIGNAL(sendHoughCircles()), Worker, SLOT(receiveHoughCircles()));
    connect(this, SIGNAL(sendHoughLines()), Worker, SLOT(receiveHoughLines()));
    connect(this, SIGNAL(sendContour()), Worker, SLOT(receiveContour()));
    connect(this, SIGNAL(sendShapeDescriptor()), Worker, SLOT(receiveShapeDescriptor()));
    connect(this, SIGNAL(sendHarris()), Worker, SLOT(receiveHarris()));



    connect (ui->ComboBoxColorSpace, SIGNAL(activated(int)), this,SLOT (receiveColorSpace()));
    connect (this, SIGNAL (sendHSV()), Worker, SLOT (receiveHSV()));
    connect (this, SIGNAL (sendRGB()), Worker, SLOT (receiveRGB()));
    connect (this, SIGNAL (sendLAB()), Worker, SLOT (receiveLAB()));
    connect (this, SIGNAL (sendYCrCb()), Worker, SLOT (receiveYCrCb()));
    connect (this, SIGNAL (sendGRAY()), Worker, SLOT (receiveGRAY()));
    connect (this, SIGNAL (sendBWImage()), Worker, SLOT (receiveEnableBinaryThreshold()));


    Worker->moveToThread(thread);
    workerTrigger->moveToThread(thread);
    thread->start();
    emit sendSetup(0);

}


void QCVWidget::receiveFrame(QImage frame){
    if(toggleCapture){
        using namespace std::this_thread;
        using namespace std::chrono;
        sleep_for(nanoseconds(1));
        sleep_until(system_clock::now() + seconds(5));
    }
    toggleCapture = false;

    ui->labelView->setPixmap(QPixmap::fromImage(frame));

}

void QCVWidget::receiveLabel (int msg){
    auto temp = std::to_string(msg);
    std::string label = std::string("Face Label is: ") + temp;
    QString msgNlabel = QString::fromStdString(label);
    lab = msgNlabel;
}

void QCVWidget::receivedLabel(){
    if(ui->checkBoxFaceRecg->isChecked ())
        ui->textEditMsgBox->setText (lab);
}


void QCVWidget::receiveOriginalFrame(QImage firstFrame){

    ui->labelView2->setPixmap(QPixmap::fromImage(firstFrame));
}

void QCVWidget::receiveFmatrix (QString fMat){
    fMatrix = fMat;
}

void QCVWidget::receiveString(){
    if(ui->checkBoxFmatrix->isChecked ())
        ui->textEditMsgBox->append (fMatrix);
}

void QCVWidget::receiveToggleStream(){
    if(!ui->pushButtonPlay->text().compare("Start Preview \n>"))
        ui->pushButtonPlay->setText("Stop Preview \n||");
    else ui->pushButtonPlay->setText("Start Preview \n>");

    emit sendToggleStream();
}

void QCVWidget::receiveVideoTrigger(){
    if(ui->comboBox->currentIndex() == 1) {
       emit sendVideoFile();
    }
    else if (ui->comboBox->currentIndex() == 2) {
        emit sendImageFile();
    }
    else if (ui->comboBox->currentIndex() == 3) {
        emit sendTriggerCam();
    }
}

void QCVWidget::closeInterface(){
    emit sendEnd();
    thread->~QThread ();
    event->accept();
}

void QCVWidget::receiveCmpVision (){
    if(ui->ComboBoxCmpVision->currentIndex () == 1){
        emit sendFAST();
    }
    if(ui->ComboBoxCmpVision->currentIndex () == 2){
        emit sendSURF();
    }
    if(ui->ComboBoxCmpVision->currentIndex () == 3){
        emit sendSIFT();
    }
    if(ui->ComboBoxCmpVision->currentIndex () == 4){
        emit sendMatch();
    }
    if(ui->ComboBoxCmpVision->currentIndex () == 5){
        ui->textEditMsgBox->setText ("Select an input distorted image\n - display calibraton pattern on each image\n -wait for to vew undistorted image ");
        if(!ui->pushButtonPlay->text().compare("Start Preview \n>")){
            ui->pushButtonPlay->setText("Stop Preview \n||");
        }
        emit sendToggleStream();
        emit sendCalibration();
        ui->textEditMsgBox->setText ("Camera Intrinsic: ");
    }
    if(ui->ComboBoxCmpVision->currentIndex () == 6){
        emit send7point();
        ui->textEditMsgBox->setText ("7point Fundamental Matrix: ");
    }
    if(ui->ComboBoxCmpVision->currentIndex () == 7){
        emit send8point ();
        ui->textEditMsgBox->setText ("8point Fundamental Matrix: ");
    }
    if(ui->ComboBoxCmpVision->currentIndex () == 8){
        emit sendRANSAC ();
        ui->textEditMsgBox->setText ("RANSAC Fundamental Matrix:");
    }
    if(ui->ComboBoxCmpVision->currentIndex () == 9){
        emit sendHomography ();
        ui->textEditMsgBox->setText ("Homography: ");
    }
    if(ui->ComboBoxCmpVision->currentIndex () == 10){
        emit sendStitch ();
    }
    if(ui->ComboBoxCmpVision->currentIndex () == 11){
        emit sendFaceDetection ();
       ui->textEditMsgBox->setText (" ");
    }
    if(ui->ComboBoxCmpVision->currentIndex () == 12){
        emit sendFDwithCascadeCl ();
    }
    if(ui->ComboBoxCmpVision->currentIndex () == 13){
        emit sendPCA ();
    }
    if(ui->ComboBoxCmpVision->currentIndex () == 14){
        emit sendFishers ();
    }
    if(ui->ComboBoxCmpVision->currentIndex () == 15){
        emit sendLBP ();
    }
}


void QCVWidget::receiveToolOptions(){


    if (ui->tools->currentIndex() == 1){
        if(ui->groupBox_4->isEnabled () == 0){
            ui->groupBox_4->setEnabled (true);
        }
        else {
            ui->groupBox_4->setEnabled (false);
        }
        emit sendLogoToggle();
    }

    if(ui->tools->currentIndex () == 2)
        emit sendSaltPepper();
    if(ui->tools->currentIndex () == 3)
        emit sendHistogram();
    if (ui->tools->currentIndex () == 4)
        emit sendHistEqual ();
    if(ui->tools->currentIndex () == 5){
        if(ui->groupBoxMorphology->isEnabled () == 0){
            ui->groupBoxMorphology->setEnabled (true);
        }
        else {
            ui->groupBoxMorphology->setEnabled (false);
        }

        emit sendMorphology();
    }
    if(ui->tools->currentIndex() == 6){
        if(ui->groupBoxBlur->isEnabled () == 0){
            ui->groupBoxBlur->setEnabled (true);
        }
        else {
            ui->groupBoxBlur->setEnabled (false);
        }
        emit sendBlure();
    }
    if(ui->tools->currentIndex () == 7){
        if(ui->groupBoxThresh->isEnabled () == 0){
            ui->groupBoxThresh->tr ("set Scale");
        }
        emit sendSobel ();
    }
    if(ui->tools->currentIndex () == 8){
        if(ui->groupBoxThresh->isEnabled () == 0){
            ui->groupBoxThresh->tr ("set Scale");
        }
        emit sendLaplacian ();
    }
    if(ui->tools->currentIndex () == 9){
        if(ui->groupBoxThresh->isEnabled () == 0){
            ui->groupBoxThresh->tr ("set Threshold");
        }
        emit sendCanny ();
    }
    if (ui->tools->currentIndex () == 10)
        emit sendHoughCircles ();
    if (ui->tools->currentIndex () == 11)
        emit sendHoughLines ();
    if(ui->tools->currentIndex () == 12)
        emit sendContour ();
    if(ui->tools->currentIndex () == 13)
        emit sendShapeDescriptor ();
    if(ui->tools->currentIndex () == 14)
        emit sendHarris ();


}

void QCVWidget::receiveColorSpace (){
    if(ui->ComboBoxColorSpace->currentIndex () == 1)
        emit sendLAB ();
    if(ui->ComboBoxColorSpace->currentIndex () == 2)
        emit sendYCrCb();
    if(ui->ComboBoxColorSpace->currentIndex () == 3)
        emit sendHSV ();
    if(ui->ComboBoxColorSpace->currentIndex () == 4)
        emit sendRGB ();
    if(ui->ComboBoxColorSpace->currentIndex () == 5)
        emit sendGRAY ();
    if(ui->ComboBoxColorSpace->currentIndex () == 6){
        emit sendBWImage ();
    }
}



void QCVWidget::saveImage(){

    toggleCapture = true;
    chdir("../QtOpencvTutorial");
    if (mkdir("imageFolder", 0777) == -1) {
        ui->textEditMsgBox->setText(strerror(errno));
    }

    DIR *dp;
    int i = 0;
    struct dirent *ep;
    dp = opendir("imageFolder");

    if (dp != NULL) {
        while (ep = (readdir (dp))){
            i++;
        }
    }
    (void) closedir (dp);
    auto temp = std::to_string(i-1);

    std::string filename = std::string("imageFolder/") + std::string("Image") + temp + std::string(".jpg");
    QString file_name = QString::fromStdString(filename);
    ui->labelView2->grab().save(file_name);
    std::string txt = std::string("Image has been successfully saved \n in ") + filename;
    QString dipTxt = QString::fromStdString(txt);
    ui->textEditMsgBox->setText(dipTxt);

}


