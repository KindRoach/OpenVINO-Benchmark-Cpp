#include <argparse/argparse.hpp>
#include <chrono>
#include <future>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <set>
#include <spdlog/spdlog.h>

argparse::ArgumentParser parseArg(int argc, char *const *argv);
using namespace cv;
using namespace std;

int decode(int sec) {
    VideoCapture cap("output/video.mp4");

    if (!cap.isOpened()) {
        spdlog::error("Error opening video stream or file");
        exit(1);
    }

    Mat frame;
    int frame_count = 0;
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

void sync_decode(int sec) {
    spdlog::info("sync encoding with in {} seconds...", sec);
    auto start = chrono::system_clock::now();
    auto frames = decode(sec);
    auto end = chrono::system_clock::now();
    auto diff = chrono::duration_cast<chrono::seconds>(end - start).count();
    double fps = static_cast<double>(frames) / static_cast<double>(diff);
    cout << "fps: " << fps << endl;
}

void multi_decode(int n_stream, int sec) {
    spdlog::info("async encoding with {} threads in {} seconds...", n_stream, sec);
    vector<future<int>> futures;

    auto start = chrono::system_clock::now();

    for (int i = 0; i < n_stream; ++i) {
        futures.push_back(async(decode, sec));
    }

    int total_frames = 0;
    for (auto &result: futures) {
        int frames = result.get();
        if (frames > 0) {
            total_frames += frames;
        }
    }

    auto end = chrono::system_clock::now();
    auto diff = chrono::duration_cast<chrono::seconds>(end - start).count();
    double fps = static_cast<double>(total_frames) / static_cast<double>(diff);

    cout << "fps: " << fps << endl;
}

argparse::ArgumentParser parseArg(int argc, char *const *argv) {
    argparse::ArgumentParser program("sync_decode");
    program.add_argument("-t", "--time")
            .help("time in seconds for benchmark")
            .default_value(60)
            .scan<'i', int>();
    program.add_argument("-rm", "--run_mode")
            .help("run mode: sync or multi")
            .default_value(string{"sync"})
            .action([](const std::string &value) {
                static const std::set<std::string> choices = {"sync", "multi"};
                if (choices.contains(value)) {
                    return value;
                }
                spdlog::error("illegal input for run mode: {}", value);
                exit(1);
            });
    program.add_argument("-n", "--n_stream")
            .help("number of decode stream")
            .default_value(16)
            .scan<'i', int>();
    program.parse_args(argc, argv);
    return program;
}

int main(int argc, char *argv[]) {
    argparse::ArgumentParser program = parseArg(argc, argv);

    int sec = program.get<int>("time");
    string run_mode = program.get<string>("run_mode");
    int n_stream = program.get<int>("n_stream");

    if (run_mode == "sync") {
        sync_decode(sec);
    } else if (run_mode == "multi") {
        multi_decode(n_stream, sec);
    }
}
