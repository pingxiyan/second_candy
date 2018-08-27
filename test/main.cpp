#include <iostream>

#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

void detectAndDraw( Mat& img, int waitValue = 0 );

std::vector<std::string> getImgList(std::string inputName)
{
	std::vector<std::string> vecFn;
	FILE* f = fopen(inputName.c_str(), "rt");
	if (f) {
		char buf[1000 + 1];
		while (fgets(buf, 1000, f)) {
			int len = (int) strlen(buf);
			while (len > 0 && isspace(buf[len - 1]))
				len--;
			buf[len] = '\0';
			cout << "file " << buf << endl;
			vecFn.push_back(buf);
		}
		fclose(f);
	}
	return vecFn;
}

void process_one_image(cv::Mat src)
{
	Mat frame1 = src.clone();
	detectAndDraw(frame1, 1);
}

void process_images(std::vector<std::string>& vecFn)
{
	for (auto fn : vecFn) {
		cv::Mat image = imread(fn, 1);
		if (!image.empty()) {
			process_one_image(image);
			char c = (char) waitKey(0);
			if (c == 27 || c == 'q' || c == 'Q') {
				break;
			}else if(c == 's' || c == 'S'){
				static int save_idx = 0;
				std::cout << "save image " << save_idx++ << std::endl;
				cv::imwrite(std::to_string(save_idx) + ".jpg", image);
			}
		} else {
			std::cerr << "Aw snap, couldn't read image " << fn << endl;
		}
	}
}

void process_camera(VideoCapture& capture)
{
	Mat frame;
	int wait_tm = 10;
	for (;;) {
		capture >> frame;
		if (frame.empty())
			break;

		process_one_image(frame);

		char c = (char) waitKey(wait_tm);
		if (c == 'w' || c == 'W') {
			wait_tm = wait_tm == 0 ? 10 : 0;
		} else if (c == 27 || c == 'q' || c == 'Q') {
			break;
		} else if (c == 's' || c == 'S') {
			static int save_idx = 0;
			std::cout << "save image " << save_idx++ << std::endl;
			cv::imwrite(std::to_string(save_idx) + ".jpg", frame);
		}
	}
}

int main(int argc, char** argv)
{
	VideoCapture capture;
	Mat frame, image;
	string inputName;

	if(argc != 2){
		std::cout << "testapp camera_id or image_name" << std::endl;
		return 1;
	}
	else{
		inputName = argv[1];
	}

	/**
	 * open camera or image files
	 */
	if (inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1)) {
		int camera = inputName.empty() ? 0 : inputName[0] - '0';
		if (!capture.open(camera))
			cout << "Capture from camera #" << camera << " didn't work" << endl;
	} else if (inputName.size()) {
		image = imread(inputName, 1);
		if (image.empty()) {
			if (!capture.open(inputName))
				cout << "Could not read " << inputName << endl;
		}
	}

	if (capture.isOpened()) {
		cout << "Video capturing has been started ..." << endl;
		process_camera(capture);
	} else {
		cout << "Image name: " << inputName << endl;
		if (!image.empty()) {
			process_one_image(image);
			waitKey(0);
		} else if (!inputName.empty()) {
			/* assume it is a text file containing the
			 list of the image filenames to be processed - one per line */
			std::vector<std::string> vecFn = getImgList(inputName);
			process_images(vecFn);
		}
	}

	return 0;
}

void detectAndDraw( Mat& img, int waitValue )
{
    const static Scalar colors[] =
    {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };

    cv::imshow( "video", img );
    cv::waitKey(waitValue);
}
