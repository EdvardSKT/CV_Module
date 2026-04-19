#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <utility>

template <typename T>
class BoundedChannel {
public:
    explicit BoundedChannel(std::size_t capacity);

    BoundedChannel(const BoundedChannel&) = delete;
    BoundedChannel& operator=(const BoundedChannel&) = delete;
    BoundedChannel(BoundedChannel&&) = delete;
    BoundedChannel& operator=(BoundedChannel&&) = delete;

    ~BoundedChannel() = default;

    [[nodiscard]] bool is_closed() const;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t size() const;

    void close();

    template <typename U>
    [[nodiscard]] bool send(U&& value);

    template <typename... Args>
    [[nodiscard]] bool emplace(Args&&... args);

    template <typename U>
    [[nodiscard]] bool try_send(U&& value);

    template <typename Rep, typename Period, typename U>
    [[nodiscard]] bool send_for(
        const std::chrono::duration<Rep, Period>& timeout,
        U&& value
    );

    [[nodiscard]] std::optional<T> recv();
    [[nodiscard]] std::optional<T> try_recv();

    template <typename Rep, typename Period>
    [[nodiscard]] std::optional<T> recv_for(
        const std::chrono::duration<Rep, Period>& timeout
    );

private:
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_cv_;
    std::condition_variable not_full_cv_;
    std::queue<T> queue_;
    bool closed_ = false;
};

#include "BoundedChannel.tpp"