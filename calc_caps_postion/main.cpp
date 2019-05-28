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
#include <string>
#include <sstream>
#include <vector>
std::vector<std::string> split(const std::string &s, char delim) {
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> elems;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
    // elems.push_back(std::move(item)); // if C++11 (based on comment from @mchiasson)
  }
  return elems;
}

std::vector<cv::Rect> parse_roi(std::string fn) {
	std::vector<cv::Rect> vecRoi;
	std::string tmp = fn;

	for(;;) {
		int p1 = tmp.find("[");
		int p2 = tmp.find("]");
		if(p1 >= 0) {
			std::string sss = tmp.substr(p1 + 1, p2-p1 - 1);
			tmp = tmp.substr(p2+1, tmp.length()-p2);

			std::vector<std::string> vtmp = split(sss, ',');
			int x1=std::atoi(vtmp[0].c_str());
			int x2=std::atoi(vtmp[1].c_str());
			int x3=std::atoi(vtmp[2].c_str());
			int x4=std::atoi(vtmp[3].c_str());

			vecRoi.push_back(cv::Rect(x1,x2, x3-x1+1, x4-x2+1));
			// vecRoi.push_back(cv::Rect(x1, x2, x3, x4));
		}
		else{
			break;
		}
	}
	
	return vecRoi;
}

cv::Mat get_roi(cv::Mat src, cv::Rect rt, cv::Point& offpt) {
	cv::Rect rtROI = cv::Rect(rt.x - rt.width/2, rt. y, rt.width + rt.width, rt.height * 3 / 5);
	cv::Rect realROI = rtROI & cv::Rect(0, 0, src.cols, src.rows);
	offpt.x = realROI.x;
	offpt.y = realROI.y;
	return cv::Mat(src, realROI);
}

bool is_peak(std::vector<float> wave, int pos) {
	int p1 = std::max(0, pos - 10);
	int p2 = std::min((int)wave.size() - 1, pos + 10);

	auto peak = std::max_element(wave.begin()+p1, wave.begin()+p2);
	int peakpos = peak - wave.begin();

	bool isPeak = true;
	if(*peak >= 50) {
		int idx = 0;
		while(idx++ < 15) {
			int b = peakpos + idx;
			int t = peakpos - idx;
			if(b < (int)wave.size() - 1 && t > 0) {
				if( wave[b - 1] > wave[b] && wave[t + 1] > wave[t]){
					continue;
				}
			}else{
				isPeak = false;
				break;
			}
		}
	}
	return isPeak;
}

int find_beast_pos(std::vector<float> wave, int pos) {
	for( int i = pos; i < (int)wave.size() - 1; i++){
		if(wave[i+1] > wave[i]){
			return i;
		}
	}
	return (int)wave.size() - 1;
}

int find_match_wave_peak(std::vector<float> left, std::vector<float> right) {
	auto peakleft = std::max_element(left.begin(), left.end());
	auto peakright = std::max_element(right.begin(), right.end());

	int posleft = peakleft - left.begin();
	int posright = peakright - right.begin();

	if(std::abs(posleft - posright) < 15){
		return posleft > posright ? find_beast_pos(left, posleft) : find_beast_pos(right, posright);
	}
	
	if(*peakleft >= *peakright) {
		if(is_peak(right, posleft)){
			return find_beast_pos(left, posleft);
		}
	}
	else {
		if(is_peak(left, posright)){
			return find_beast_pos(right, posright);
		}
	}

	return 0;
}

cv::Rect get_caps(cv::Mat roi) {
	cv::Rect rt;
	cv::Mat gray;
	cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
	cv::Mat edge = cv::Mat::zeros(gray.size(), CV_8UC1);

	int height = gray.rows;
	std::vector<float> left(height);
	std::vector<float> right(height);
	std::vector<float> left2(height);
	std::vector<float> right2(height);

	int stride = gray.cols;
	for(int h = 4; h < gray.rows-4; h++){
		uint8_t *psrc = gray.data + h * gray.cols;
		uint8_t *pdst = edge.data + h * edge.cols;
		for(int w = 4; w < gray.cols-4; w++) {
			// int diff = psrc[w-2] + psrc[w-3] - psrc[w+2] - psrc[w+3];
			int diff = std::min(0, psrc[w-2*stride] - psrc[w+2*stride]) 
				+ std::min(0, psrc[w-3*stride] - psrc[w+3*stride]);
			pdst[w] = std::abs(diff)/2;
		}
	}

	int mid  = gray.cols/2;
	for(int h = 4; h < gray.rows-20; h++){
		uint8_t *pdst = edge.data + h * edge.cols;
		int offv = 0;
		for(int w = 0; w < mid; w++) {
			left[h] = left[h] + pdst[w + offv*stride];
			if(w%3==1){
				offv ++;
				if(offv >= 19){
					break;
				}
			}
			// pdst[w + offv*stride] = 255;
		}

		// cv::imshow("ddd", edge);
		// cv::waitKey(0);

		offv = 0;
		int idx = 0;
		for(int w = stride - 1; w > mid; w--, idx++) {
			right[h] = right[h] + pdst[w + offv*stride];
			if(idx%3==1){
				offv ++;
				if(offv >= 19){
					break;
				}
			}
		}
	}

	for(int h = 5; h < height-5; h++) {
		left2[h] = std::accumulate(left.begin()+h-5, left.begin()+h+4, 0);
		right2[h] = std::accumulate(right.begin()+h-5, right.begin()+h+4, 0);
	}

	int maxleft = *std::max_element(left2.begin(), left2.end());
	int maxright = *std::max_element(right2.begin(), right2.end());
	int maxv = std::max(maxleft, maxright);

	for(int h = 5; h < height-5; h++) {
		left2[h] = left2[h]*100.f/maxv;
		right2[h] = right2[h]*100.f/maxv;
	}

	int peak = find_match_wave_peak(left2, right2);

	cv::Mat leftshow=cv::Mat::zeros(height, 100, CV_8UC1);
	cv::Mat rightshow=cv::Mat::zeros(height, 100, CV_8UC1);
	
	for(int h = 5; h < height-5; h++) {
		cv::line(leftshow, cv::Point((int)left2[h], h), cv::Point((int)left2[h+1], h+1), cv::Scalar(180,180,180), 1);
		cv::line(rightshow, cv::Point((int)right2[h], h), cv::Point((int)right2[h+1], h+1), cv::Scalar(180,180,180), 1);
	}
	cv::line(leftshow, cv::Point(0, peak), cv::Point(leftshow.cols-1, peak), cv::Scalar(180,180,180), 1);

	cv::imshow("leftshow", leftshow);
	cv::imshow("rightshow", rightshow);

	cv::imshow("gray", gray);
	cv::imshow("edge", edge);
	cv::waitKey(0);

	return rt;
}

int main(int argc, char** argv) {
	std::vector<std::string> vecFn = getImgList("../../imagelist.set");
	
	for(auto fn : vecFn) {
		cv::Mat src = cv::imread(fn, 1);
		if (src.empty()) {
			std::cout << "can't imread: " << fn << std::endl;
			continue;
		}

		std::vector<cv::Rect> vecRoi = parse_roi(fn);
		for(auto rt:vecRoi){
			cv::Point offpt;
			cv::Mat roi = get_roi(src, rt, offpt);
			cv::Rect rtcaps = get_caps(roi);			

			cv::Rect newRt = rt;
			newRt.y = rtcaps.y + offpt.y;
			newRt.height = rt.height - (newRt.y - rt.y);
			// std::cout << rt.x << ", "<< rt.y << ", "<< rt.width << ", "<< rt.height<< std::endl;
			cv::rectangle(src, rt, cv::Scalar(0,255,0), 1);
			cv::rectangle(src, newRt, cv::Scalar(0,0,255), 1);
		}
		// cv::Mat roi = get_roi(src, , )
		
		// cv::imshow("xx", src);
		// cv::waitKey(0);
	}
	
	return 0;
}