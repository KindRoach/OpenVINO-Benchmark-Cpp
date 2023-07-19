#include <thread>
#include <future>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <argparse/argparse.hpp>

using namespace cv;
using namespace std;

uint64_t decode(int sec) {
    VideoCapture cap("output/video.mp4");

    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    Mat frame;
    uint64_t frame_count = 0;
    auto finish = chrono::system_clock::now() + chrono::seconds(sec);
    while (chrono::system_clock::now() < finish) {
        bool success = cap.read(frame);
        if (not success) {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        frame_count++;
    }

    cap.release();

    return frame_count;
}

void async_decode(uint n_stream, uint sec) {
    vector<future<uint64_t>> futures;

    auto start = chrono::system_clock::now();

    for (int i = 0; i < n_stream; ++i) {
        futures.push_back(async(decode, sec));
    }

    uint64_t total_fps = 0;
    for (auto &result: futures) {
        uint64_t fps = result.get();
        if (fps > 0) {
            total_fps += fps;
        }
    }

    auto end = chrono::system_clock::now();
    auto diff = chrono::duration_cast<chrono::seconds>(end - start).count();
    double fps = static_cast<double>(total_fps) / static_cast<double>(diff);

    cout << "fps: " << fps << endl;
}

int main(int argc, char *argv[]) {
    argparse::ArgumentParser program("sync_decode");
    program.add_argument("-t", "--time")
            .help("time in seconds for benchmark")
            .default_value(60)
            .scan<'i', int>();
    program.add_argument("-n", "--n_stream")
            .help("number of decode stream")
            .default_value(16)
            .scan<'i', int>();
    program.parse_args(argc, argv);
    int sec = program.get<int>("time");
    int n_stream = program.get<int>("n_stream");
    async_decode(n_stream, sec);
}

