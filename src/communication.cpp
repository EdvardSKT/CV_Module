#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <utility>

template <typename T>
class BoundedChannel {
public:
    explicit BoundedChannel(std::size_t capacity) : capacity_(capacity) {
        if (capacity_ == 0) {
            throw std::invalid_argument("BoundedChannel capacity must be > 0");
        }
    }

    BoundedChannel(const BoundedChannel&) = delete;
    BoundedChannel& operator=(const BoundedChannel&) = delete;
    BoundedChannel(BoundedChannel&&) = delete;
    BoundedChannel& operator=(BoundedChannel&&) = delete;

    ~BoundedChannel() = default;

    [[nodiscard]] bool is_closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

    [[nodiscard]] std::size_t capacity() const noexcept {
        return capacity_;
    }

    [[nodiscard]] std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        closed_ = true;
        not_empty_cv_.notify_all();
        not_full_cv_.notify_all();
    }

    template <typename U>
    [[nodiscard]] bool send(U&& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_cv_.wait(lock, [this] {
            return closed_ || queue_.size() < capacity_;
        });

        if (closed_) {
            return false;
        }

        queue_.emplace(std::forward<U>(value));
        not_empty_cv_.notify_one();
        return true;
    }

    template <typename... Args>
    [[nodiscard]] bool emplace(Args&&... args) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_cv_.wait(lock, [this] {
            return closed_ || queue_.size() < capacity_;
        });

        if (closed_) {
            return false;
        }

        queue_.emplace(std::forward<Args>(args)...);
        not_empty_cv_.notify_one();
        return true;
    }

    template <typename U>
    [[nodiscard]] bool try_send(U&& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_ || queue_.size() >= capacity_) {
            return false;
        }

        queue_.emplace(std::forward<U>(value));
        not_empty_cv_.notify_one();
        return true;
    }

    template <typename Rep, typename Period, typename U>
    [[nodiscard]] bool send_for(
        const std::chrono::duration<Rep, Period>& timeout,
        U&& value
    ) {
        std::unique_lock<std::mutex> lock(mutex_);
        const bool ready = not_full_cv_.wait_for(lock, timeout, [this] {
            return closed_ || queue_.size() < capacity_;
        });

        if (!ready || closed_) {
            return false;
        }

        queue_.emplace(std::forward<U>(value));
        not_empty_cv_.notify_one();
        return true;
    }

    [[nodiscard]] std::optional<T> recv() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_cv_.wait(lock, [this] {
            return closed_ || !queue_.empty();
        });

        if (queue_.empty()) {
            return std::nullopt;
        }

        T value = std::move(queue_.front());
        queue_.pop();
        not_full_cv_.notify_one();
        return value;
    }

    [[nodiscard]] std::optional<T> try_recv() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }

        T value = std::move(queue_.front());
        queue_.pop();
        not_full_cv_.notify_one();
        return value;
    }

    template <typename Rep, typename Period>
    [[nodiscard]] std::optional<T> recv_for(
        const std::chrono::duration<Rep, Period>& timeout
    ) {
        std::unique_lock<std::mutex> lock(mutex_);
        const bool ready = not_empty_cv_.wait_for(lock, timeout, [this] {
            return closed_ || !queue_.empty();
        });

        if (!ready || queue_.empty()) {
            return std::nullopt;
        }

        T value = std::move(queue_.front());
        queue_.pop();
        not_full_cv_.notify_one();
        return value;
    }

private:
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_cv_;
    std::condition_variable not_full_cv_;
    std::queue<T> queue_;
    bool closed_ = false;
};