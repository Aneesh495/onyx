#include "onyx/storage/stream_ingestor.hpp"
#include <stdexcept>
#include <chrono>

extern void register_test(std::string name, std::function<void()> func);

#define ONYX_TEST(name) \
    static void test_##name(); \
    static int dummy_##name = (register_test(#name, test_##name), 0); \
    static void test_##name()

#define ONYX_ASSERT(cond) do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); } while (0)
#define ONYX_ASSERT_EQ(a, b) do { if ((a) != (b)) throw std::runtime_error("Assertion failed: " #a " == " #b); } while (0)

ONYX_TEST(StreamingIngestorThroughput) {
    auto schema = std::make_shared<onyx::storage::Schema>(std::vector<onyx::storage::Field>{
        {"sensor_id", onyx::DataType(onyx::TypeId::INT32)},
        {"val", onyx::DataType(onyx::TypeId::FLOAT64)}
    });

    auto table = std::make_shared<onyx::storage::Table>("sensor_stream", schema);
    onyx::storage::StreamIngestor ingestor(table, 1024 * 1024);
    ingestor.start();

    // Stream 10,000 micro-batch records
    std::string payload = "101,42.5\n";
    for (int i = 0; i < 10000; ++i) {
        bool ok = ingestor.ingest_record(payload.data(), payload.size());
        ONYX_ASSERT(ok);
    }

    ONYX_ASSERT_EQ(ingestor.total_ingested_records(), 10000);
    ingestor.stop();
}
