#include "onyx/storage/stream_ingestor.hpp"
#include <sstream>

namespace onyx::storage {

StreamIngestor::StreamIngestor(std::shared_ptr<Table> target_table, size_t ring_buffer_bytes)
    : target_table_(std::move(target_table)), ring_buffer_(ring_buffer_bytes) {
    start_time_ = std::chrono::high_resolution_clock::now();
}

StreamIngestor::~StreamIngestor() {
    stop();
}

void StreamIngestor::start() {
    if (running_.exchange(true)) return;
    start_time_ = std::chrono::high_resolution_clock::now();
    worker_thread_ = std::thread(&StreamIngestor::worker_loop, this);
}

void StreamIngestor::stop() {
    if (!running_.exchange(false)) return;
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

bool StreamIngestor::ingest_record(const void* data, size_t bytes) {
    if (!running_.load(std::memory_order_relaxed)) return false;
    bool ok = ring_buffer_.push(data, bytes);
    if (ok) {
        total_records_.fetch_add(1, std::memory_order_relaxed);
        total_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }
    return ok;
}

double StreamIngestor::records_per_second() const noexcept {
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_).count();
    if (elapsed_us <= 0) return 0.0;
    return static_cast<double>(total_records_.load()) / (static_cast<double>(elapsed_us) / 1000000.0);
}

void StreamIngestor::worker_loop() {
    constexpr size_t BATCH_SIZE = 4096;
    auto schema = target_table_->schema();

    while (running_.load(std::memory_order_relaxed) || !ring_buffer_.empty()) {
        auto slice = ring_buffer_.peek_available();
        if (slice.bytes == 0 || !slice.ptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Consume available bytes from ring buffer
        ring_buffer_.consume(slice.bytes);
    }
}

} // namespace onyx::storage
