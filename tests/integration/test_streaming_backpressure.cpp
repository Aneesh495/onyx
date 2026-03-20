#include "test_framework.hpp"
#include "onyx/storage/stream_ingestor.hpp"
#include <thread>
#include <atomic>
#include <vector>

ONYX_TEST(StreamingBackpressureAndConcurrentProducers) {
    auto schema = std::make_shared<onyx::storage::Schema>(std::vector<onyx::storage::Field>{
        {"sensor_id", onyx::DataType(onyx::TypeId::INT32)},
        {"val", onyx::DataType(onyx::TypeId::FLOAT64)}
    });

    auto table = std::make_shared<onyx::storage::Table>("pressure_stream", schema);
    // 256KB ring buffer to test saturation and consumer drain
    onyx::storage::StreamIngestor ingestor(table, 256 * 1024);
    ingestor.start();

    constexpr int NUM_PRODUCERS = 4;
    constexpr int RECORDS_PER_PRODUCER = 5000;
    std::atomic<uint64_t> success_count{0};

    std::vector<std::thread> producers;
    producers.reserve(NUM_PRODUCERS);

    for (int p = 0; p < NUM_PRODUCERS; ++p) {
        producers.emplace_back([&ingestor, &success_count, p]() {
            std::string payload = std::to_string(100 + p) + ",42.0\n";
            for (int i = 0; i < RECORDS_PER_PRODUCER; ++i) {
                while (!ingestor.ingest_record(payload.data(), payload.size())) {
                    std::this_thread::yield();
                }
                success_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& t : producers) {
        t.join();
    }

    ingestor.stop();

    ONYX_ASSERT_EQ(success_count.load(), NUM_PRODUCERS * RECORDS_PER_PRODUCER);
}
