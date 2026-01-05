#include "test_framework.hpp"
#include "onyx/storage/stream_ingestor.hpp"
#include <stdexcept>
#include <chrono>


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
