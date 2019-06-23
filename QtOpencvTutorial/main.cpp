#include "qcvwidget.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QCVWidget w;
    w.show();

    return a.exec();
}
