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
			cout << buf << endl;
			vecFn.push_back(buf);
		}
		fclose(f);
	}
	return vecFn;
}

int main(int argc, char** argv)
{
	std::vector<std::string> vecFn = getImgList("..//filename.set");
	
	for(auto fn : vecFn) {
		cv::Mat src = cv::imread(fn, 1);
		if (src.empty()) {
			std::cout << "can't imread: " << fn << std::endl;
			continue;
		}
		
		cv::imshow("xx", src);
		cv::waitKey(0);
	}
	
	return 0;
}