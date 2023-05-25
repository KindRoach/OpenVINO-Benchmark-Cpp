#include "opencv2/opencv.hpp"
#include <chrono>
#include <iostream>
#include <future>

using namespace cv;
using namespace std;
using namespace std::chrono;

double decode() {
    VideoCapture cap("outputs/video.mp4");

    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    Mat frame;
    uint64_t frame_count = 0;
    auto finish = system_clock::now() + 1min;
    while (system_clock::now() < finish) {
        bool success = cap.read(frame);
        if (not success) {
            cap.set(CAP_PROP_POS_FRAMES, 0);
        }
        frame_count++;
    }

    cap.release();

    return static_cast<double>(frame_count) / 60.0;
}

int main() {
    vector<future<double>> futures;
    for (int i = 0; i < 16; ++i) {
        futures.push_back(async(decode));
    }

    double total_fps = 0;
    for (auto &result: futures) {
        double fps = result.get();
        if (fps > 0) {
            total_fps += fps;
        }
    }

    cout << "fps: " << total_fps << endl;
}

