#pragma once

template <typename T>
BoundedChannel<T>::BoundedChannel(std::size_t capacity) : capacity_(capacity) {
    if (capacity_ == 0) {
        throw std::invalid_argument("BoundedChannel capacity must be > 0");
    }
}

template <typename T>
bool BoundedChannel<T>::is_closed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
}

template <typename T>
std::size_t BoundedChannel<T>::capacity() const noexcept {
    return capacity_;
}

template <typename T>
std::size_t BoundedChannel<T>::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

template <typename T>
void BoundedChannel<T>::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
        return;
    }
    closed_ = true;
    not_empty_cv_.notify_all();
    not_full_cv_.notify_all();
}

template <typename T>
template <typename U>
bool BoundedChannel<T>::send(U&& value) {
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

template <typename T>
template <typename... Args>
bool BoundedChannel<T>::emplace(Args&&... args) {
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

template <typename T>
template <typename U>
bool BoundedChannel<T>::try_send(U&& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || queue_.size() >= capacity_) {
        return false;
    }

    queue_.emplace(std::forward<U>(value));
    not_empty_cv_.notify_one();
    return true;
}

template <typename T>
template <typename Rep, typename Period, typename U>
bool BoundedChannel<T>::send_for(
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

template <typename T>
std::optional<T> BoundedChannel<T>::recv() {
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

template <typename T>
std::optional<T> BoundedChannel<T>::try_recv() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
        return std::nullopt;
    }

    T value = std::move(queue_.front());
    queue_.pop();
    not_full_cv_.notify_one();
    return value;
}

template <typename T>
template <typename Rep, typename Period>
std::optional<T> BoundedChannel<T>::recv_for(
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