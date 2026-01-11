#pragma once

#include "onyx/execution/query_engine.hpp"
#include <string>
#include <thread>
#include <atomic>
#include <functional>

namespace onyx::server {

class HttpServer {
public:
    explicit HttpServer(uint16_t port = 8080, execution::QueryEngine* engine = &execution::QueryEngine::instance());
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    void start();
    void stop();

    [[nodiscard]] uint16_t port() const noexcept { return port_; }
    [[nodiscard]] bool is_running() const noexcept { return running_.load(); }

private:
    void listen_loop();
    void handle_connection(int client_socket);

    std::string handle_get_metrics();
    std::string handle_get_tables();
    std::string handle_post_query(const std::string& body);
    std::string handle_post_stream(const std::string& body);
    std::string handle_static_file(const std::string& path);

    uint16_t port_;
    execution::QueryEngine* engine_;
    int server_socket_{-1};
    std::atomic<bool> running_{false};
    std::thread listen_thread_;
};

} // namespace onyx::server
