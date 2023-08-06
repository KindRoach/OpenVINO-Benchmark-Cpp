#pragma once

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>

template<typename T>
class ConcurrentQueue : public std::queue<T> {
public:
    explicit ConcurrentQueue(int capacity) : capacity(capacity), closed(false) {}

    void push(const T &item) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return !is_full(); });
        std::queue<T>::push(item);
        lock.unlock();
        cv.notify_all();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return !std::queue<T>::empty(); });
        T item = std::queue<T>::front();
        std::queue<T>::pop();
        lock.unlock();
        cv.notify_all();
        return item;
    }

    bool is_full() {
        return std::queue<T>::size() == capacity;
    }

    void close() {
        std::unique_lock<std::mutex> lock(mutex);
        closed = true;
        lock.unlock();
        cv.notify_all();
    }

    bool is_closed() {
        std::unique_lock<std::mutex> lock(mutex);
        return closed;
    }


private:
    int capacity;
    bool closed;
    std::mutex mutex;
    std::condition_variable cv;
};