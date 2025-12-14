#include "onyx/server/http_server.hpp"
#include "onyx/common/logger.hpp"

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <fstream>

namespace onyx::server {

HttpServer::HttpServer(uint16_t port, execution::QueryEngine* engine)
    : port_(port), engine_(engine) {}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::start() {
    if (running_.exchange(true)) return;

    server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket_ < 0) {
        ONYX_LOG_ERROR("Failed to create server socket");
        running_ = false;
        return;
    }

    int opt = 1;
    setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);

    if (bind(server_socket_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ONYX_LOG_ERROR("Failed to bind socket to port " << port_);
        close(server_socket_);
        running_ = false;
        return;
    }

    if (listen(server_socket_, 64) < 0) {
        ONYX_LOG_ERROR("Failed to listen on socket");
        close(server_socket_);
        running_ = false;
        return;
    }

    ONYX_LOG_INFO("Onyx HTTP Telemetry Server listening on http://127.0.0.1:" << port_);
    listen_thread_ = std::thread(&HttpServer::listen_loop, this);
}

void HttpServer::stop() {
    if (!running_.exchange(false)) return;
    if (server_socket_ >= 0) {
        close(server_socket_);
        server_socket_ = -1;
    }
    if (listen_thread_.joinable()) {
        listen_thread_.join();
    }
}

void HttpServer::listen_loop() {
    while (running_.load()) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_sock = accept(server_socket_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_sock < 0) {
            if (!running_.load()) break;
            continue;
        }

        std::thread([this, client_sock]() {
            handle_connection(client_sock);
        }).detach();
    }
}

void HttpServer::handle_connection(int client_socket) {
    char buffer[4096];
    ssize_t bytes_read = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
    if (bytes_read <= 0) {
        close(client_socket);
        return;
    }
    buffer[bytes_read] = '\0';

    std::string request(buffer);
    std::istringstream req_stream(request);
    std::string method, path, proto;
    req_stream >> method >> path >> proto;

    std::string body;
    size_t body_pos = request.find("\r\n\r\n");
    if (body_pos != std::string::npos) {
        body = request.substr(body_pos + 4);
    }

    std::string response_body;
    std::string content_type = "application/json";
    int status_code = 200;

    if (method == "GET" && path == "/metrics") {
        response_body = handle_get_metrics();
    } else if (method == "GET" && path == "/tables") {
        response_body = handle_get_tables();
    } else if (method == "POST" && path == "/query") {
        response_body = handle_post_query(body);
    } else if (method == "POST" && path == "/stream") {
        response_body = handle_post_stream(body);
    } else {
        response_body = handle_static_file(path);
        if (path.ends_with(".html") || path == "/") content_type = "text/html";
        else if (path.ends_with(".css")) content_type = "text/css";
        else if (path.ends_with(".js")) content_type = "application/javascript";
    }

    std::ostringstream response;
    response << "HTTP/1.1 " << status_code << " OK\r\n"
             << "Content-Type: " << content_type << "\r\n"
             << "Content-Length: " << response_body.size() << "\r\n"
             << "Access-Control-Allow-Origin: *\r\n"
             << "Connection: close\r\n\r\n"
             << response_body;

    std::string res_str = response.str();
    send(client_socket, res_str.c_str(), res_str.size(), 0);
    close(client_socket);
}

std::string HttpServer::handle_post_query(const std::string& body) {
    // Parse query string from body
    std::string sql = body;
    size_t qpos = body.find("\"query\":");
    if (qpos != std::string::npos) {
        size_t first = body.find('\"', qpos + 8);
        size_t last = body.find('\"', first + 1);
        if (first != std::string::npos && last != std::string::npos) {
            sql = body.substr(first + 1, last - first - 1);
        }
    }

    auto res = engine_->execute_query(sql);

    std::ostringstream json;
    json << "{\n"
         << "  \"status\": \"" << (res.status.is_ok() ? "OK" : res.status.message()) << "\",\n"
         << "  \"latency_ms\": " << res.execution_time_ms << ",\n"
         << "  \"kernel_us\": " << res.kernel_time_us << ",\n"
         << "  \"rows\": " << (res.table ? res.table->num_rows() : 0) << "\n"
         << "}";
    return json.str();
}

std::string HttpServer::handle_post_stream([[maybe_unused]] const std::string& body) {
    return "{\"status\": \"OK\", \"bytes_ingested\": " + std::to_string(body.size()) + "}";
}

std::string HttpServer::handle_static_file(const std::string& path) {
    std::string file_path = "web" + (path == "/" ? "/index.html" : path);
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return "{\"error\": \"File not found\"}";
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

} // namespace onyx::server
