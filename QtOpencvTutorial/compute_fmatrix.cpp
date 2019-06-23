#include "compute_fmatrix.h"


FMatrix::FMatrix()
{

}

cv::Mat FMatrix::displayImages(std::vector<cv::Mat>outImage){
    int size;
    int i = 1;
    int m, n;
    int x, y;

    // w - Maximum number of images in a row
    // h - Maximum number of images in a column
    int w, h;

    // scale - How much we have to resize the image
    float scale;
    int max;

    w = 2; h = 2;
    size = 400;
    m = 20;
    n = 20;

    // Create a new 3 channel image
    cv::Mat DispImage = cv::Mat::zeros(cv::Size(100 + size*w, 60 + size*h), CV_8UC3);



//    // Loop for nArgs number of arguments
    for (auto img :outImage) {

        m += (20 + size);

        // Find the width and height of the image
        x = img.cols;
        y = img.rows;

        // Find whether height or width is greater in order to resize the image
        max = (x > y)? x: y;

        // Find the scaling factor to resize the image
        scale = (float) ( (float) max / size );

        // Used to Align the images
        if( i % w == 0 && m!= 20) {
            m = 20;
            n+= 20 + size;
        }

        // Set the image ROI to display the current image
        // Resize the input image and copy the it to the Single Big Image
        cv::Rect ROI(m, n, (int)( x/scale ), (int)( y/scale ));
        cv::Mat temp;
        resize(img,temp, cv::Size(ROI.width, ROI.height));
        temp.copyTo(DispImage(ROI));
        i = i+1;
    }

 return DispImage;
}


void FMatrix::getFundamental7PointAndDrawEpilines(cv::Mat image, cv::Mat image2, cv::Mat &img, cv::Mat &fundamental){

    //compute keypoints for image
    surfDetector = cv::xfeatures2d::SURF::create(2500);
    surfDetector->detect(image, keypoints1);

    //compute keypoints for image2
    surfDetector->detect(image2, keypoints2);

    //compute descriptors for firsta and second images
    surfDetector->compute(image, keypoints1, descriptors);
    surfDetector->compute(image2, keypoints2, descriptors2);

    //detect matches
    flannBasedMatcher.match(descriptors, descriptors2, matches);

    /* between church01 and church03 */

    std::nth_element(matches.begin(), matches.begin() + 15 - 1, matches.end());
    matches.erase(matches.begin() + 15, matches.end());

    //to draw matches
    cv::drawMatches(image, keypoints1, image2, keypoints2, matches, imageFeatures, cv::Scalar(0, 0, 255));

    for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
         it!= matches.end(); ++it) {
             // Get the indexes of the selected matched keypoints
             pointIndexes1.push_back(it->queryIdx);
             pointIndexes2.push_back(it->trainIdx);
    }

    // Convert keypoints into Point2f
    cv::KeyPoint::convert(keypoints1,selPoints1,pointIndexes1);
    cv::KeyPoint::convert(keypoints2,selPoints2,pointIndexes2);

    // check by drawing the points
    std::vector<cv::Point2f>::const_iterator it= selPoints1.begin();
    while (it!=selPoints1.end()) {

        // draw a circle at each corner location
        cv::circle(image,*it,3,cv::Scalar(0,255,0),2);
        ++it;
    }

    it= selPoints2.begin();
    while (it!=selPoints2.end()) {
        // draw a circle at each corner location
        cv::circle(image2,*it,3,cv::Scalar(0,255,0),2);
        ++it;
    }

    // Compute F matrix from 7 matches
    fundamental= cv::findFundamentalMat(
        cv::Mat(selPoints1), // points in first image
        cv::Mat(selPoints2), // points in second image
        CV_FM_7POINT);       // 7-point method


    // draw the left points corresponding epipolar lines in right image
    cv::computeCorrespondEpilines(
        cv::Mat(selPoints1), // image points
        1,                   // in image 1 (can also be 2)
        fundamental, // F matrix
        lines1);     // vector of epipolar lines

    // for all epipolar lines
    for (std::vector<cv::Vec3f>::const_iterator it= lines1.begin();
         it!=lines1.end(); ++it) {

             // draw the epipolar line between first and last column
             cv::line(image2,cv::Point(0,-(*it)[2]/(*it)[1]),
                             cv::Point(image2.cols,-((*it)[2]+(*it)[0]*image2.cols)/(*it)[1]),
                             cv::Scalar(255,255,255));
    }

    // draw the left points corresponding epipolar lines in left image
    cv::computeCorrespondEpilines(cv::Mat(selPoints2),2,fundamental,lines2);
    for (std::vector<cv::Vec3f>::const_iterator it= lines2.begin();
         it!=lines2.end(); ++it) {

             // draw the epipolar line between first and last column
             cv::line(image,cv::Point(0,-(*it)[2]/(*it)[1]),
                             cv::Point(image.cols,-((*it)[2]+(*it)[0]*image.cols)/(*it)[1]),
                             cv::Scalar(255,255,255));
    }
    qDebug() << "after epilines";


    cv::Size size(image.cols,image.rows);
    cv::resize(imageFeatures, imageFeatures, size);
    std::vector<cv::Mat>gridImg;

    gridImg.push_back (image);
    gridImg.push_back (image2);
    gridImg.push_back (imageFeatures);


    img = displayImages(gridImg);
}

// function to get fundamental matrix from 8 point algo
void FMatrix::getFundamental8PointAndDrawEpilines(cv::Mat image, cv::Mat image2, cv::Mat &img, cv::Mat &fundamental){

    //compute keypoints for image
    surfDetector = cv::xfeatures2d::SURF::create(2500);
    surfDetector->detect(image, keypoints1);

    //compute keypoints for image2
    surfDetector->detect(image2, keypoints2);

    //compute descriptors for firsta and second images
    surfDetector->compute(image, keypoints1, descriptors);
    surfDetector->compute(image2, keypoints2, descriptors2);

    //detect matches
    flannBasedMatcher.match(descriptors, descriptors2, matches);

    std::nth_element(matches.begin(), matches.begin() + 15 - 1, matches.end());
    matches.erase(matches.begin() + 15, matches.end());

    //to draw matches
    cv::drawMatches(image, keypoints1, image2, keypoints2, matches, imageFeatures, cv::Scalar(0, 0, 255));


    // Convert 1 vector of keypoints into
    // 2 vectors of Point2f
    for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
         it!= matches.end(); ++it) {

             // Get the indexes of the selected matched keypoints
             pointIndexes1.push_back(it->queryIdx);
             pointIndexes2.push_back(it->trainIdx);
    }

    // Convert keypoints into Point2f
    cv::KeyPoint::convert(keypoints1,selPoints1,pointIndexes1);
    cv::KeyPoint::convert(keypoints2,selPoints2,pointIndexes2);

    // check by drawing the points
    std::vector<cv::Point2f>::const_iterator it= selPoints1.begin();
    while (it!=selPoints1.end()) {

        // draw a circle at each corner location
        cv::circle(image,*it,3,cv::Scalar(255,0,0),2);
        ++it;
    }

    it= selPoints2.begin();
    while (it!=selPoints2.end()) {

        // draw a circle at each corner location
        cv::circle(image2,*it,3,cv::Scalar(255,0,0),2);
        ++it;
    }

    // Compute F matrix from 7 matches
    fundamental = cv::findFundamentalMat(
                cv::Mat(selPoints1), // points in first image
                cv::Mat(selPoints2), // points in second image
                CV_FM_8POINT);       // 7-point method


    // draw the left points corresponding epipolar lines in right image
    cv::computeCorrespondEpilines(
        cv::Mat(selPoints1), // image points
        1,                   // in image 1 (can also be 2)
        fundamental, // F matrix
        lines1);     // vector of epipolar lines

    // for all epipolar lines
    for (std::vector<cv::Vec3f>::const_iterator it= lines1.begin();
         it!=lines1.end(); ++it) {

             // draw the epipolar line between first and last column
             cv::line(image2,cv::Point(0,-(*it)[2]/(*it)[1]),
                             cv::Point(image2.cols,-((*it)[2]+(*it)[0]*image2.cols)/(*it)[1]),
                             cv::Scalar(255,255,255));
    }

    // draw the left points corresponding epipolar lines in left image
    cv::computeCorrespondEpilines(cv::Mat(selPoints2),2,fundamental,lines2);
    for (std::vector<cv::Vec3f>::const_iterator it= lines2.begin();
         it!=lines2.end(); ++it) {

             // draw the epipolar line between first and last column
             cv::line(image,cv::Point(0,-(*it)[2]/(*it)[1]),
                             cv::Point(image.cols,-((*it)[2]+(*it)[0]*image.cols)/(*it)[1]),
                             cv::Scalar(255,255,255));
    }


    cv::Size size(image.cols,image.rows);
    cv::resize(imageFeatures, imageFeatures, size);
    std::vector<cv::Mat>gridImg;

    gridImg.push_back (image);
    gridImg.push_back (image2);
    gridImg.push_back (imageFeatures);


    img = displayImages(gridImg);
}

// function to get fundamental matrix from 7 point algo
void FMatrix::getFundamentalRANSACAndDrawEpilines(cv::Mat image, cv::Mat image2, cv::Mat &img, cv::Mat &fundamental){

    //compute keypoints for image
    surfDetector = cv::xfeatures2d::SURF::create(2500);
    surfDetector->detect(image, keypoints1);

    //compute keypoints for image2
    surfDetector->detect(image2, keypoints2);

    //compute descriptors for firsta and second images
    surfDetector->compute(image, keypoints1, descriptors);
    surfDetector->compute(image2, keypoints2, descriptors2);

    //detect matches
    flannBasedMatcher.match(descriptors, descriptors2, matches);

    //to draw matches
    cv::drawMatches(image, keypoints1, image2, keypoints2, matches, imageFeatures, cv::Scalar(255, 0, 0));

    // Convert keypoints into Point2f
    for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
         it!= matches.end(); ++it) {

             // Get the position of left keypoints
             float x= keypoints1[it->queryIdx].pt.x;
             float y= keypoints1[it->queryIdx].pt.y;
             points1.push_back(cv::Point2f(x,y));
             // Get the position of right keypoints
             x= keypoints2[it->trainIdx].pt.x;
             y= keypoints2[it->trainIdx].pt.y;
             points2.push_back(cv::Point2f(x,y));
    }

    // Compute F matrix using RANSAC
    std::vector<uchar> inliers(points1.size(),0);
    fundamental= cv::findFundamentalMat(
        cv::Mat(points1),cv::Mat(points2), // matching points
        inliers,      // match status (inlier ou outlier)
        CV_FM_RANSAC, // RANSAC method
        1,            // distance to epipolar line
        0.98);        // confidence probability

    std::nth_element(matches.begin(), matches.begin() + 15 - 1, matches.end());
    matches.erase(matches.begin() + 15, matches.end());

    //get selected points
    for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
         it!= matches.end(); ++it) {

             // Get the indexes of the selected matched keypoints
             pointIndexes1.push_back(it->queryIdx);
             pointIndexes2.push_back(it->trainIdx);
    }

    // Convert keypoints into Point2f
    std::vector<cv::Point2f> selPoints1, selPoints2;
    cv::KeyPoint::convert(keypoints1,selPoints1,pointIndexes1);
    cv::KeyPoint::convert(keypoints2,selPoints2,pointIndexes2);


    // Draw the epipolar line of few points
    cv::computeCorrespondEpilines(cv::Mat(selPoints1),1,fundamental,lines1);
    for (std::vector<cv::Vec3f>::const_iterator it= lines1.begin();
         it!=lines1.end(); ++it) {

             cv::line(image2,cv::Point(0,-(*it)[2]/(*it)[1]),
                             cv::Point(image2.cols,-((*it)[2]+(*it)[0]*image2.cols)/(*it)[1]),
                             cv::Scalar(255,255,255));
    }

    cv::computeCorrespondEpilines(cv::Mat(selPoints2),2,fundamental,lines2);
    for (std::vector<cv::Vec3f>::const_iterator it= lines2.begin();
         it!=lines2.end(); ++it) {

             cv::line(image,cv::Point(0,-(*it)[2]/(*it)[1]),
                             cv::Point(image.cols,-((*it)[2]+(*it)[0]*image.cols)/(*it)[1]),
                             cv::Scalar(255,255,255));
    }

    // Draw the inlier points
    std::vector<cv::Point2f>::const_iterator itPts= points1.begin();
    std::vector<uchar>::const_iterator itIn= inliers.begin();
    while (itPts!=points1.end()) {

        // draw a circle at each inlier location
        if (*itIn) {
            cv::circle(image,*itPts,3,cv::Scalar(0,0,255),2);
            points1In.push_back(*itPts);
        }
        ++itPts;
        ++itIn;
    }

    itPts= points2.begin();
    itIn= inliers.begin();
    while (itPts!=points2.end()) {

        // draw a circle at each inlier location
        if (*itIn) {
            cv::circle(image2,*itPts,3,cv::Scalar(0,0,255),2);
            points2In.push_back(*itPts);
        }
        ++itPts;
        ++itIn;
    }

    // Display the images with points
    cv::Size size(image.cols,image.rows);
    cv::resize(imageFeatures, imageFeatures, size);
    std::vector<cv::Mat>gridImg;

    gridImg.push_back (image);
    gridImg.push_back (image2);
    gridImg.push_back (imageFeatures);

    img = displayImages(gridImg);

}

//function to get homography
void FMatrix::getHomography(cv::Mat image1, cv::Mat image2, cv::Mat &img_matches, cv::Mat &homography){

    cv::Mat src_gray1,src_gray2;
    src_gray1 = image1.clone();
    src_gray2 = image2.clone();

    if (image1.type() != CV_8UC1){
        /// Convert it to gray
        cvtColor( image1, src_gray1, CV_BGR2GRAY );
    }

    if (image2.type() != CV_8UC1){
        /// Convert it to gray
        cvtColor( image2, src_gray2, CV_BGR2GRAY );
    }
    int minHessian = 400;
      cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );
      std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;
      cv::Mat descriptors_object, descriptors_scene;
      detector->detectAndCompute( image1, cv::Mat(), keypoints_object, descriptors_object );
      detector->detectAndCompute( image2, cv::Mat(), keypoints_scene, descriptors_scene );
      //-- Step 2: Matching descriptor vectors using FLANN matcher
      cv::FlannBasedMatcher matcher;
      std::vector< cv::DMatch > matches;
      matcher.match( descriptors_object, descriptors_scene, matches );
      double max_dist = 0; double min_dist = 100;
      //-- Quick calculation of max and min distances between keypoints
      for( int i = 0; i < descriptors_object.rows; i++ )
      { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
      }
      printf("-- Max dist : %f \n", max_dist );
      printf("-- Min dist : %f \n", min_dist );
      //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
      std::vector< cv::DMatch > good_matches;
      for( int i = 0; i < descriptors_object.rows; i++ )
      { if( matches[i].distance < 3*min_dist )
         { good_matches.push_back( matches[i]); }
      }
      drawMatches( image1, keypoints_object, image2, keypoints_scene,
                   good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
      //-- Localize the object
      std::vector<cv::Point2f> obj;
      std::vector<cv::Point2f> scene;
      for( size_t i = 0; i < good_matches.size(); i++ )
      {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
      }
       homography = findHomography( obj, scene, cv::RANSAC );
      //-- Get the corners from the image_1 ( the object to be "detected" )
      std::vector<cv::Point2f> obj_corners(4);
      obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( image1.cols, 0 );
      obj_corners[2] = cvPoint( image1.cols, image1.rows ); obj_corners[3] = cvPoint( 0, image1.rows );
      std::vector<cv::Point2f> scene_corners(4);
      perspectiveTransform( obj_corners, scene_corners, homography);
      //-- Draw lines between the corners (the mapped object in the scene - image_2 )
      line( img_matches, scene_corners[0] + cv::Point2f( image1.cols, 0), scene_corners[1] + cv::Point2f( image1.cols, 0), cv::Scalar(0, 0, 255), 4 );
      line( img_matches, scene_corners[1] + cv::Point2f( image1.cols, 0), scene_corners[2] + cv::Point2f( image1.cols, 0), cv::Scalar( 0, 0, 255), 4 );
      line( img_matches, scene_corners[2] + cv::Point2f( image1.cols, 0), scene_corners[3] + cv::Point2f( image1.cols, 0), cv::Scalar( 0, 0, 255), 4 );
      line( img_matches, scene_corners[3] + cv::Point2f( image1.cols, 0), scene_corners[0] + cv::Point2f( image1.cols, 0), cv::Scalar( 0, 0, 255), 4 );

}

