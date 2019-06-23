#include "toolsbox.h"
#include <stdio.h>

toolsBox::toolsBox()
{
}

toolsBox::~toolsBox(){}


void toolsBox::saltAndPepper(cv::Mat Image){

}


cv::Mat toolsBox::histogramOfImage(cv::Mat src){
    if (src.type()==CV_8UC1) {

        // Initialize parameters
        int histSize = 256;    // bin size
        float range[] = { 0, 255 };
        const float *ranges[] = { range };

        // Calculate histogram
        cv:: MatND hist;
        calcHist( &src, 1, 0, cv::Mat(), hist, 1, &histSize, ranges, true, false );

        // Show the calculated histogram in command window
        double total;
        total = src.rows * src.cols;

        // Plot the histogram
        int hist_w = 512; int hist_h = 400;
        int bin_w = cvRound( (double) hist_w/histSize );

        cv::Mat histImage( hist_h, hist_w, CV_8UC1, cv::Scalar( 0,0,0) );
        normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
        for( int i = 1; i < histSize; i++ )
        {
            line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                  cv::Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                  cv::Scalar( 255, 0, 0), 2, 8, 0  );
        }
        return  histImage;
    }
    else {
        cv::Mat dst;



        /// Separate the image in 3 places ( B, G and R )
        std::vector<cv::Mat> bgr_planes;
        split( src, bgr_planes );

        /// Establish the number of bins
        int histSize = 256;

        /// Set the ranges ( for B,G,R) )
        float range[] = { 0, 256 } ;
        const float* histRange = { range };

        bool uniform = true; bool accumulate = false;

        cv::Mat b_hist, g_hist, r_hist;

        /// Compute the histograms:
        calcHist( &bgr_planes[0], 1, 0,cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

        // Draw the histograms for B, G and R
        int hist_w = 512; int hist_h = 400;
        int bin_w = cvRound( (double) hist_w/histSize );

        cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );

        /// Normalize the result to [ 0, histImage.rows ]
        normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
        normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
        normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

        /// Draw for each channel
        for( int i = 1; i < histSize; i++ )
        {
            line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                  cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                  cv::Scalar( 255, 0, 0), 2, 8, 0  );
            line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                  cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                  cv::Scalar( 0, 255, 0), 2, 8, 0  );
            line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                  cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                  cv::Scalar( 0, 0, 255), 2, 8, 0  );
        }
        return  histImage;
    }
}


cv::Mat toolsBox::morphologyOperation(cv::Mat src, int struct_elemt, int size,int op){

    cv::Mat dst;
    cv::Mat element = getStructuringElement( struct_elemt, cv::Size( 2*size + 1, 2*size+1 ), cv::Point( size, size ) );


    if( struct_elemt == 0 ){
        struct_elemt = cv::MORPH_RECT;
    }
    else if( struct_elemt == 1 ){
        struct_elemt = cv::MORPH_CROSS;
    }
    else if( struct_elemt == 2) {
        struct_elemt = cv::MORPH_ELLIPSE;
    }

    if( op == 0 ){
        op = cv::MORPH_OPEN;
        cv::morphologyEx( src, dst, op, element );
    }
    else if( op == 1 ){
        op = cv::MORPH_CLOSE;
        cv::morphologyEx( src, dst, op, element );
    }
    else if( op == 2) {
        erode( src, dst, element );
    }
    else if( op == 3) {
        dilate( src, dst, element );
    }

    return dst;
}

cv::Mat toolsBox::histEqualization(cv::Mat src){
    cv::Mat dst;

    if (src.type() != CV_8UC1){
        cvtColor(src, src, cv::COLOR_RGB2YCrCb);

        std::vector<cv::Mat> vec_channels;
        split(src, vec_channels);

        equalizeHist(vec_channels[0], vec_channels[0]);
        merge(vec_channels, src);

        cvtColor(src, dst,  cv::COLOR_YCrCb2RGB);

    }
    else {
        equalizeHist( src, dst );
    }

    return dst;
}





cv::Mat toolsBox::imageBlur(cv::Mat src, int op, int KERNEL){

    cv::Mat dst = src.clone();

    /// Applying Homogeneous blur
    if( op == 0 ) {
        for ( int i = 1; i < KERNEL; i = i + 2 ) {
            blur( src, dst, cv::Size( i, i ), cv::Point(-1,-1) );
        }
    }

    /// Applying Gaussian blugroupBoxr
    if( op == 1 ) {

        for ( int i = 1; i < KERNEL; i = i + 2 ) {
            GaussianBlur( src, dst, cv::Size( i, i ), 0, 0 );
        }
    }

    /// Applying Median blur
    if( op == 2) {

        for ( int i = 1; i < KERNEL; i = i + 2 ) {
            medianBlur ( src, dst, i );
        }
    }

    /// Applying Bilateral Filter
    if( op == 3) {

        for ( int i = 1; i < KERNEL; i = i + 2 )
        { bilateralFilter ( src, dst, i, i*2, i/2 );
        }
    }
    return dst;
}


cv::Mat toolsBox::sobelOperation(cv::Mat src, int scale){
    cv::Mat src_gray = src.clone();
    cv::Mat dst;

    int delta = 0;
    int ddepth = CV_16S;

    GaussianBlur( src, src, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );

    if (src.type() != CV_8UC1){
        /// Convert it to gray
        cvtColor( src, src_gray, CV_BGR2GRAY );
    }

    /// Generate grad_x and grad_y
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    cv::Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    cv::Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst );

    return dst;
}


cv::Mat toolsBox::laplacianOperation(cv::Mat src, int scale){

    cv::Mat src_gray = src.clone();
    cv::Mat dst;
    int kernel_size = 3;
    int delta = 0;
    int ddepth = CV_16S;

    /// Remove noise by blurring with a Gaussian filter
    GaussianBlur( src, src, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );

    if (src.type() != CV_8UC1){
        /// Convert the image to grayscale
        cvtColor( src, src_gray, CV_BGR2GRAY );
    }

    /// Apply Laplace function
    cv::Mat abs_dst;

    Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT );
    convertScaleAbs( dst, abs_dst );

    return abs_dst;
}

cv::Mat toolsBox::cannyEdgeDetection(cv::Mat src, int threshold ){

    cv::Mat dst;
    cv::Mat src_gray = src.clone();

    int KERNEL = 3;
    int ratio = 3;

    if (src.type() != CV_8UC1){
        /// Convert the image to grayscale
        cvtColor( src, src_gray, CV_BGR2GRAY );
    }

    for ( int i = 1; i < KERNEL; i = i + 2 ) {
        blur( src_gray, dst, cv::Size( i, i ), cv::Point(-1,-1) );
    }

    Canny( dst, dst, threshold, threshold*ratio, KERNEL );

    return dst;
}


cv::Mat toolsBox::HoughLines(cv::Mat src, int n){
    cv::Mat dst,cdst;
    cv::Mat src_gray = src.clone();

    if (src.type() != CV_8UC1){
        /// Convert the image to grayscale
        cvtColor( src, src_gray, CV_BGR2GRAY );
    }

    Canny(src_gray, src_gray, 50, 200, 3);

    std::vector<cv::Vec4i> lines;
    HoughLinesP(src_gray, lines, 1, CV_PI/180, 50, 50, n );

    for( size_t i = 0; i < lines.size(); i++ ) {
        cv::Vec4i l = lines[i];
        line( src, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 3, CV_AA);
    }

    return src;
}

cv::Mat toolsBox::HoughCircles(cv::Mat src, int n){
    cv::Mat dst;
    cv::Mat src_gray = src.clone();

    if (src.type() != CV_8UC1){
        /// Convert the image to grayscale
        cvtColor( src, src_gray, CV_BGR2GRAY );
    }

    medianBlur(src_gray, src_gray, 5);

    std::vector<cv::Vec3f>  circles;

    cv::HoughCircles(src_gray, circles, cv::HOUGH_GRADIENT, 1,
                     src_gray.rows/16,
                     100, 30, 1, n
                     );
    for( size_t i = 0; i < circles.size(); i++ )
    {
        cv:: Vec3i c = circles[i];
        circle( src, cv::Point(c[0], c[1]), c[2], cv::Scalar(0,0,255), 3, cv::LINE_AA);
        circle( src, cv::Point(c[0], c[1]), 2, cv::Scalar(0,255,0), 3, cv::LINE_AA);
    }

    return src;

}


cv::Mat toolsBox::imageContours(cv::Mat src, int thresh){

    cv::Mat src_gray = src.clone();
    cv::RNG rng(12345);

    if (src.type() != CV_8UC1){
        /// Convert the image to grayscale
        cvtColor( src, src_gray, CV_BGR2GRAY );
    }

    blur( src_gray, src_gray, cv::Size(3,3) );
    cv::Mat canny_output;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    /// Detect edges using canny
    Canny( src_gray, canny_output, thresh, thresh*2, 3 );
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
    /// Draw contours
   // cv::Mat drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
        cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( src, contours, i, color, 2, 8, hierarchy, 0, cv::Point() );
    }

    return src ;
}

cv::Mat toolsBox::shapeDescriptor(cv::Mat src, int threash){
    cv::Mat src_gray = src.clone();
    cv::RNG rng(12345);

    if (src.type() != CV_8UC1){
        /// Convert the image to grayscale
        cvtColor( src, src_gray, CV_BGR2GRAY );
    }

    blur( src_gray, src_gray, cv::Size(3,3) );
    cv::Mat canny_output;
    Canny( src_gray, canny_output, threash, threash*2 );

    std::vector<std::vector<cv::Point> > contours;
    findContours( canny_output, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );

    std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
    std::vector<cv::Rect> boundRect( contours.size() );
    std::vector<cv::Point2f>centers( contours.size() );
    std::vector<float>radius( contours.size() );

    for( size_t i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( contours[i], contours_poly[i], 3, true );
        boundRect[i] = boundingRect( contours_poly[i] );
        minEnclosingCircle( contours_poly[i], centers[i], radius[i] );
    }


    for( size_t i = 0; i< contours.size(); i++ )
    {
        cv::Scalar color = cv::Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( src, contours_poly, (int)i, color );
        rectangle( src, boundRect[i].tl(), boundRect[i].br(), color, 2 );
        circle( src, centers[i], (int)radius[i], color, 2 );
    }
    return src;
}


cv::Mat toolsBox::harrisCornerDetector(cv::Mat src, int thresh){
    cv::Mat src_gray = src.clone();
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros( src.size(), CV_32FC1 );
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;


    if (src.type() != CV_8UC1){
        /// Convert the image to grayscale
        cvtColor( src, src_gray, CV_BGR2GRAY );
    }

    /// Detecting corners
      cornerHarris( src_gray, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT );

      /// Normalizing
        normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
        convertScaleAbs( dst_norm, dst_norm_scaled );

        /// Drawing a circle around corners
          for( int j = 0; j < dst_norm.rows ; j++ ) {
              for( int i = 0; i < dst_norm.cols; i++ ) {
                    if( (int) dst_norm.at<float>(j,i) > thresh+70 ) {
                       circle( src, cv::Point( i, j ), 10,  cv::Scalar(0), 2, 8, 0 );
                      }
                  }
             }
          return src;
}

cv::Mat toolsBox::extractSURF(cv::Mat src){

    cv::Mat src_gray = src.clone();

    if (src.type() != CV_8UC1){
        /// Convert the image to grayscale
        cvtColor( src, src_gray, CV_BGR2GRAY );
    }
    //-- Step 1: Detect the keypoints using SURF Detector
      int minHessian = 400;

      //cv::Ptr<cv::SURF> detector = cv::SURF::create( minHessian );



}




