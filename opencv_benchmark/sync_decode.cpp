#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <argparse/argparse.hpp>

using namespace cv;
using namespace std;

void sync_decode(int sec) {
    VideoCapture cap("outputs/video.mp4");

    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return;
    }

    Mat frame;
    uint64_t frame_count = 0;
    auto finish = chrono::system_clock::now() + chrono::seconds(sec);
    while (chrono::system_clock::now() < finish) {
        bool success = cap.read(frame);
        if (not success) {
            cap.set(CAP_PROP_POS_FRAMES, 0);
        }
        frame_count++;
    }

    cout << "fps: " << static_cast<double>(frame_count) / 60.0 << endl;

    cap.release();
}

int main(int argc, char *argv[]) {
    argparse::ArgumentParser program("sync_decode");
    program.add_argument("-t", "--time")
            .help("time in seconds for benchmark")
            .default_value(60)
            .scan<'i', int>();
    program.parse_args(argc, argv);
    int sec = program.get<int>("time");
    sync_decode(sec);
}
