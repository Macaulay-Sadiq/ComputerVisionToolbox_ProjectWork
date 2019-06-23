#ifndef QCVWIDGET_H
#define QCVWIDGET_H

#include <QWidget>
#include <QThread>
#include <QCloseEvent>
#include <opencv2/core/core.hpp>



namespace Ui {
class QCVWidget;
}

class QCVWidget : public QWidget
{
    Q_OBJECT

private:
    Ui::QCVWidget *ui;
    QThread *thread;
    QCloseEvent *event;
    QString lab;
    void setup();

public:
    explicit QCVWidget(QWidget *parent = nullptr);
    ~QCVWidget();
    QString fMatrix;
    bool toggleCapture;


signals:
    void sendSetup(int device);
    void sendToggleStream();
    void sendVideoFile();
    void sendImageFile();
    void sendTriggerCam();
    void sendEnd();
    void sendLogoToggle();
    void sendSaltPepper();
    void sendHSV();
    void sendRGB();
    void sendYCrCb();
    void sendLAB();
    void sendGRAY();
    void sendHistogram();
    void sendMorphology();
    void sendBWImage();
    void sendHistEqual();
    void sendBlure();
    void sendSobel();
    void sendLaplacian();
    void sendCanny();
    void sendHoughLines();
    void sendHoughCircles();
    void sendContour();
    void sendShapeDescriptor();
    void sendHarris();
    void sendCalibration();
    void sendFAST();
    void sendSIFT();
    void sendSURF();
    void sendMatch();
    void send7point();
    void send8point();
    void sendRANSAC();
    void sendHomography();
    void sendStitch();
    void sendFaceDetection();
    void sendFDwithCascadeCl();
    void sendPCA();
    void sendFishers();
    void sendLBP();




private slots:
    void receiveOriginalFrame(QImage frame1);
    void receiveFrame(QImage frame);
    void receiveFmatrix(QString strFmatrix);
    void receiveLabel(int);
    void receiveToggleStream();
    void receiveVideoTrigger();
    void closeInterface();
    void receiveToolOptions();
    void receiveColorSpace();
    void saveImage();
    void receiveCmpVision();
    void receiveString();
    void receivedLabel();

};

#endif // QCVWIDGET_H
