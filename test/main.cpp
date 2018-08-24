#include <iostream>

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void detectAndDraw( Mat& img, int waitValue = 0 );

int main(int argc, char** argv)
{
	VideoCapture capture;
	Mat frame, image;
	string inputName;
	bool tryflip;
	double scale;

	if(argc != 2){
		std::cout << "testapp camera_id or image_name" << std::endl;
		return 1;
	}
	else{
		inputName = argv[1];
	}

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

		for (;;) {
			capture >> frame;
			if (frame.empty())
				break;

			Mat frame1 = frame.clone();
			detectAndDraw(frame1, 1);

			char c = (char) waitKey(10);
			if (c == 27 || c == 'q' || c == 'Q')
				break;
		}
	} else {
		cout << "Image name: " << inputName << endl;
		if (!image.empty()) {
			detectAndDraw(image);
			waitKey(0);
		} else if (!inputName.empty()) {
			/* assume it is a text file containing the
			 list of the image filenames to be processed - one per line */
			FILE* f = fopen(inputName.c_str(), "rt");
			if (f) {
				char buf[1000 + 1];
				while (fgets(buf, 1000, f)) {
					int len = (int) strlen(buf);
					while (len > 0 && isspace(buf[len - 1]))
						len--;
					buf[len] = '\0';
					cout << "file " << buf << endl;
					image = imread(buf, 1);
					if (!image.empty()) {
						detectAndDraw(image);
						char c = (char) waitKey(0);
						if (c == 27 || c == 'q' || c == 'Q')
							break;
					} else {
						cerr << "Aw snap, couldn't read image " << buf << endl;
					}
				}
				fclose(f);
			}
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
