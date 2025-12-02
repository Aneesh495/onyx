#pragma once

#include "onyx/storage/table.hpp"
#include "onyx/cuda/pinned_ring_buffer.hpp"
#include <thread>
#include <atomic>
#include <chrono>

namespace onyx::storage {

class StreamIngestor {
public:
    StreamIngestor(std::shared_ptr<Table> target_table, size_t ring_buffer_bytes = 64 * 1024 * 1024);
    ~StreamIngestor();

    StreamIngestor(const StreamIngestor&) = delete;
    StreamIngestor& operator=(const StreamIngestor&) = delete;

    void start();
    void stop();

    // Push raw bytes (e.g. CSV line, binary event) into ingestion pipeline
    bool ingest_record(const void* data, size_t bytes);

    [[nodiscard]] uint64_t total_ingested_records() const noexcept { return total_records_.load(); }
    [[nodiscard]] uint64_t total_ingested_bytes() const noexcept { return total_bytes_.load(); }
    [[nodiscard]] double records_per_second() const noexcept;

private:
    void worker_loop();

    std::shared_ptr<Table> target_table_;
    cuda::PinnedRingBuffer ring_buffer_;
    std::atomic<bool> running_{false};
    std::thread worker_thread_;

    std::atomic<uint64_t> total_records_{0};
    std::atomic<uint64_t> total_bytes_{0};
    std::chrono::high_resolution_clock::time_point start_time_;
};

} // namespace onyx::storage
