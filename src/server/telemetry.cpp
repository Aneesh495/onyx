#include "onyx/server/http_server.hpp"
#include "onyx/cuda/gpu_memory_pool.hpp"
#include <sstream>

namespace onyx::server {

std::string HttpServer::handle_get_metrics() {
    auto mem_stats = cuda::GpuMemoryPool::global().get_stats();

    std::ostringstream json;
    json << "{\n"
         << "  \"gpu_memory\": {\n"
         << "    \"reserved_bytes\": " << mem_stats.total_reserved_bytes << ",\n"
         << "    \"active_bytes\": " << mem_stats.active_allocated_bytes << ",\n"
         << "    \"peak_bytes\": " << mem_stats.peak_allocated_bytes << ",\n"
         << "    \"total_allocations\": " << mem_stats.total_allocations << ",\n"
         << "    \"total_deallocations\": " << mem_stats.total_deallocations << ",\n"
         << "    \"num_arenas\": " << mem_stats.num_arenas << ",\n"
         << "    \"fragmentation_ratio\": " << mem_stats.fragmentation_ratio() << "\n"
         << "  },\n"
         << "  \"engine\": {\n"
         << "    \"cuda_enabled\": "
#if defined(ONYX_HAS_CUDA) && ONYX_HAS_CUDA == 1
         << "true"
#else
         << "false"
#endif
         << ",\n"
         << "    \"simt_emulation\": "
#if !defined(ONYX_HAS_CUDA) || ONYX_HAS_CUDA == 0
         << "true"
#else
         << "false"
#endif
         << "\n  }\n"
         << "}";
    return json.str();
}

std::string HttpServer::handle_get_tables() {
    auto table_names = engine_->catalog().list_tables();
    std::ostringstream json;
    json << "{\n  \"tables\": [\n";
    for (size_t i = 0; i < table_names.size(); ++i) {
        auto tbl = engine_->catalog().get_table(table_names[i]);
        if (i > 0) json << ",\n";
        json << "    {\n"
             << "      \"name\": \"" << tbl->name() << "\",\n"
             << "      \"rows\": " << tbl->num_rows() << ",\n"
             << "      \"batches\": " << tbl->num_batches() << ",\n"
             << "      \"schema\": \"" << tbl->schema()->to_string() << "\"\n"
             << "    }";
    }
    json << "\n  ]\n}";
    return json.str();
}

} // namespace onyx::server
