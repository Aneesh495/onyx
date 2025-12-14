#include "onyx/server/http_server.hpp"
#include "onyx/storage/catalog.hpp"
#include "onyx/common/logger.hpp"
#include <iostream>
#include <csignal>

static std::atomic<bool> g_shutdown{false};

void signal_handler(int) {
    g_shutdown = true;
}

int main(int argc, char** argv) {
    uint16_t port = 8080;
    if (argc > 1) {
        port = static_cast<uint16_t>(std::stoi(argv[1]));
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Pre-populate sample telemetry dataset
    std::string csv_data =
        "device_id,latency_us,temperature,status\n"
        "101,120.5,45.2,1\n"
        "102,1540.2,68.1,2\n"
        "103,450.8,52.3,1\n"
        "104,2100.0,74.5,3\n"
        "105,89.1,41.0,1\n";

    auto table = onyx::storage::Table::from_csv("telemetry", csv_data);
    onyx::storage::Catalog::instance().register_table(table);

    onyx::server::HttpServer server(port);
    server.start();

    std::cout << "Onyx Telemetry Server running on port " << port << ". Press Ctrl+C to stop.\n";

    while (!g_shutdown.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "Stopping Onyx server...\n";
    server.stop();
    return 0;
}
