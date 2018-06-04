#include <iostream>
#include <string>
#include <cstdlib>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;
bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i < j );
}
int main(int argc, const char** argv)
{

	

	VideoCapture cap;
	cap.open(0);
	Mat src,dst;
	Mat grayScale;
	gpu::GpuMat d_frame;
	gpu::GpuMat gpuGrad_x, gpuGrad_y;

	gpu::GpuMat gpuGradient, gpuBlur;

	gpu::GpuMat gpuErode,gpuDilate;

	gpu::GpuMat gpuThres,gpuMorphologyEx ;

	gpu::GpuMat close;

	int pic_seq = 0; 
	// Mat element = getStructuringElement(MORPH_RECT, Size(21, 7));

	Mat element = getStructuringElement(MORPH_RECT, Size(21, 7));
	
	while(1){

		cap >> src;
		cvtColor( src, grayScale, COLOR_RGB2GRAY );
		d_frame.upload(grayScale);

		gpu::Sobel(d_frame,gpuGrad_x , CV_32F, 1, 0,-1);
		gpu::Sobel(d_frame,gpuGrad_y , CV_32F, 0, 1,-1);

		//###################### version3


		// gpu::subtract(gpuGrad_x, gpuGrad_y, gpuGradient);

		// gpu::exp(gpuGradient,gpuGradient);
		// gpu::log(gpuGradient,gpuGradient);
		// gpu::sqr(gpuGradient,gpuGradient);

		// gpuGradient.convertTo(gpuGradient, CV_8UC4);

		// gpu::blur(gpuGradient,gpuBlur, Size(11,11));
		// // 
		// gpu::erode(gpuBlur,gpuErode,Mat(),Point(-1,-1),4);
		// // gpu::abs(gpuErode,gpuErode);
		// gpu::threshold(gpuErode,gpuThres, 200,255, THRESH_BINARY);
		// gpu::morphologyEx(gpuThres, gpuMorphologyEx, MORPH_CLOSE, element);
		// gpu::dilate(gpuMorphologyEx,gpuDilate,Mat(),Point(-1,-1),15);

		// gpu::threshold(gpuBlur,gpuThres, 110,255, THRESH_BINARY);
		
		// // 
		



		//###################### version2

		// gpu::abs(gpuGrad_x,gpuGrad_x);
		// gpu::abs(gpuGrad_y,gpuGrad_y);
		// gpu::subtract(gpuGrad_x, gpuGrad_y, gpuGradient);

		// gpu::exp(gpuGradient,gpuGradient);
		// gpu::log(gpuGradient,gpuGradient);

		// gpuGradient.convertTo(gpuGradient, CV_8UC4);
		// gpu::blur(gpuGradient,gpuBlur, Size(11,11));
		// gpu::erode(gpuBlur,gpuErode,Mat(),Point(-1,-1),2);

		// gpu::threshold(gpuErode,gpuThres, 110,255, THRESH_BINARY);
		// gpu::morphologyEx(gpuThres, gpuMorphologyEx, MORPH_CLOSE, element);
		// gpu::dilate(gpuMorphologyEx,gpuDilate,Mat(),Point(-1,-1),4);



		//###################### version1
		gpu::subtract(gpuGrad_x, gpuGrad_y, gpuGradient);

		gpu::abs(gpuGradient,gpuGradient);

		gpuGradient.convertTo(gpuGradient, CV_8UC4);

		gpu::blur(gpuGradient,gpuBlur, Size(9,9));

		gpu::threshold(gpuBlur,gpuThres, 225,255, THRESH_BINARY);

		gpu::morphologyEx(gpuThres, gpuMorphologyEx, MORPH_CLOSE, element);

		gpu::erode(gpuMorphologyEx,gpuErode,Mat(),Point(-1,-1),4);

		gpu::dilate(gpuErode,gpuDilate,Mat(),Point(-1,-1),10);


		gpuDilate.download(dst);

		imshow("frame1", dst);

		gpuDilate.download(dst);

		vector<vector<Point> > contours;
        findContours(dst,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        // std::cout << "find : " << contours.size() << std::endl;

        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );
        vector<Point2f>center( contours.size() );
        vector<float>radius( contours.size() );

  //       std::sort(contours.begin(), contours.end(), compareContourAreas);
  //       std::vector<cv::Point> biggestContour = contours[contours.size()-1];
		// std::vector<cv::Point> smallestContour = contours[0];

		// std::cout << "biggestContour : " << biggestContour << std::endl;
		// std::cout << "smallestContour : " << smallestContour << std::endl;
		
        for( int i = 0; i < contours.size(); i++ )
            { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
               boundRect[i] = boundingRect( Mat(contours_poly[i]) );
               minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
            }
         for( int i = 0; i< contours.size(); i++ )
           {
           	
             if(contourArea(contours[i]) > 5500 ){
             	// std::cout << "find : " << contours.size() << std::endl;
                stringstream seq;
                seq << pic_seq;
                Rect r = boundRect[i];
                cv::Mat croppedImage = src(r);
                cv::imwrite( "./image/Gray_Image"+seq.str()+".jpg", croppedImage );
                pic_seq++;
                // pic_seq++;
                cv::rectangle(src, boundRect[i].tl(), boundRect[i].br(),  Scalar(255,0,0), 2, 8, 0 );
                // circle( frame, center[i], (int)radius[i], Scalar(255,255,0), 2, 8, 0 );
                // circle( frame, mc[i], 20, Scalar(255,0,0), 2, 8, 0 );
                }
             // circle( frame, mc[i], 20, Scalar(255,0,0), -1, 8, 0 );
               
           }

       //      std::system("zbarimg ./image/Gray_Image.jpg>test.txt"); // execute the UNIX command "ls -l >test.txt"
    			// std::cout << std::ifstream("test.txt").rdbuf();
        
		imshow("frame2", src);
		// imshow("frame", dst);
		int key = waitKey(30);
		if (key == 27)
            break;
	}
}
// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/highgui/highgui.hpp"
// #include "highgui.h"
// #include <stdlib.h>
// #include <stdio.h>

// using namespace cv;

// /// Global variables
// Mat src, erosion_dst, dilation_dst;

// int erosion_elem = 0;
// int erosion_size = 0;
// int dilation_elem = 0;
// int dilation_size = 0;
// int const max_elem = 2;
// int const max_kernel_size = 21;

// /** Function Headers */
// void Erosion( int, void* );
// void Dilation( int, void* );

// /** @function main */
// int main( int argc, char** argv )
// {
//   /// Load an image
// 	VideoCapture cap;
// 	cap.open(0);
//   	while(1){
// 		cap >> src;

//   /// Create windows
// 		  namedWindow( "Erosion Demo", CV_WINDOW_AUTOSIZE );
// 		  namedWindow( "Dilation Demo", CV_WINDOW_AUTOSIZE );
// 		  cvMoveWindow( "Dilation Demo", src.cols, 0 );

// 		  /// Create Erosion Trackbar
// 		  createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Erosion Demo",
// 		                  &erosion_elem, max_elem,
// 		                  Erosion );

// 		  createTrackbar( "Kernel size:\n 2n +1", "Erosion Demo",
// 		                  &erosion_size, max_kernel_size,
// 		                  Erosion );

// 		  /// Create Dilation Trackbar
// 		  createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo",
// 		                  &dilation_elem, max_elem,
// 		                  Dilation );

// 		  createTrackbar( "Kernel size:\n 2n +1", "Dilation Demo",
// 		                  &dilation_size, max_kernel_size,
// 		                  Dilation );

// 		  /// Default start
// 		  Erosion( 0, 0 );
// 		  Dilation( 0, 0 );
// 		int key = waitKey(30);
// 		if (key == 27)
//             break;
// 		 }
//   return 0;
// }

// /**  @function Erosion  */
// void Erosion( int, void* )
// {
//   int erosion_type;
//   if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
//   else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
//   else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

//   Mat element = getStructuringElement( erosion_type,
//                                        Size( 2*erosion_size + 1, 2*erosion_size+1 ),
//                                        Point( erosion_size, erosion_size ) );

//   /// Apply the erosion operation
//   erode( src, erosion_dst, element );
//   imshow( "Erosion Demo", erosion_dst );
// }

// /** @function Dilation */
// void Dilation( int, void* )
// {
//   int dilation_type;
//   if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
//   else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
//   else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

//   Mat element = getStructuringElement( dilation_type,
//                                        Size( 2*dilation_size + 1, 2*dilation_size+1 ),
//                                        Point( dilation_size, dilation_size ) );
//   /// Apply the dilation operation
//   dilate( src, dilation_dst, element );
//   imshow( "Dilation Demo", dilation_dst );
// }