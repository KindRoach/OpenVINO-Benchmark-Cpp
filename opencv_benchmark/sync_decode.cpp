#include "opencv2/opencv.hpp"
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace std::chrono;

int main() {
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

    cout << "fps: " << static_cast<double>(frame_count) / 60.0 << endl;

    cap.release();
}
