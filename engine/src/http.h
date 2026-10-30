// Minimal HTTP/1.1 + SSE plumbing for the Bruce local server.
#pragma once

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "json.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#if defined(_MSC_VER)
#pragma comment(lib, "ws2_32.lib")
#endif
using socket_t = SOCKET;
static const socket_t k_invalid_socket = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
using socket_t = int;
static const socket_t k_invalid_socket = -1;
#endif

struct HttpRequest {
    std::string method;
    std::string path;
    std::string version;
    std::map<std::string, std::string> headers;
    std::string body;
    int error_status = 0;
    std::string error_message;
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;
    std::map<std::string, std::string> headers;
};

static inline std::string ascii_lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static inline std::string trim_str(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

static inline std::string http_reason(int status) {
    switch (status) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        default: return "Error";
    }
}

static inline void close_socket(socket_t s) {
    if (s == k_invalid_socket) return;
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
}

static inline bool send_all(socket_t s, const void* data, size_t n) {
    const char* p = (const char*)data;
    while (n > 0) {
#ifdef _WIN32
        int sent = ::send(s, p, (int)std::min<size_t>(n, 1 << 20), 0);
        if (sent == SOCKET_ERROR) {
            if (WSAGetLastError() == WSAEINTR) continue;
            return false;
        }
#else
        ssize_t sent = ::send(s, p, std::min<size_t>(n, 1 << 20), MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) continue;
            return false;
        }
#endif
        if (sent == 0) return false;
        p += sent;
        n -= (size_t)sent;
    }
    return true;
}

static inline void set_socket_timeout(socket_t s, int seconds) {
#ifdef _WIN32
    DWORD ms = (DWORD)seconds * 1000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&ms, sizeof(ms));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char*)&ms, sizeof(ms));
#else
    timeval tv;
    tv.tv_sec = seconds;
    tv.tv_usec = 0;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

static inline bool recv_chunk(socket_t s, std::string& buf) {
    char tmp[8192];
    int n = ::recv(s, tmp, (int)sizeof(tmp), 0);
    if (n <= 0) return false;
    buf.append(tmp, (size_t)n);
    return true;
}

// Reads one request. Returns false on socket/parse error; req.error_status is set for HTTP errors.
static inline bool read_http_request(socket_t s, HttpRequest& req, size_t max_body = 64u << 20) {
    std::string buf;
    size_t header_end = std::string::npos;
    while ((header_end = buf.find("\r\n\r\n")) == std::string::npos) {
        if (buf.size() > (256u << 10)) { req.error_status = 400; req.error_message = "request headers too large"; return false; }
        if (!recv_chunk(s, buf)) { req.error_status = 400; req.error_message = "connection closed"; return false; }
    }

    std::string head = buf.substr(0, header_end);
    std::string body = buf.substr(header_end + 4);
    std::istringstream hs(head);
    std::string line;
    if (!std::getline(hs, line)) { req.error_status = 400; req.error_message = "empty request"; return false; }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    {
        std::istringstream ls(line);
        if (!(ls >> req.method >> req.path >> req.version)) {
            req.error_status = 400; req.error_message = "bad request line"; return false;
        }
    }
    while (std::getline(hs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = ascii_lower(trim_str(line.substr(0, colon)));
        std::string val = trim_str(line.substr(colon + 1));
        req.headers[key] = val;
    }

    auto it_len = req.headers.find("content-length");
    size_t content_length = 0;
    if (it_len != req.headers.end()) {
        char* endp = nullptr;
        unsigned long long v = std::strtoull(it_len->second.c_str(), &endp, 10);
        if (!endp || *endp != '\0') { req.error_status = 400; req.error_message = "bad Content-Length"; return false; }
        content_length = (size_t)v;
    }
    if (content_length > max_body) { req.error_status = 413; req.error_message = "request body too large"; return false; }
    if (req.headers.find("transfer-encoding") != req.headers.end() &&
        ascii_lower(req.headers["transfer-encoding"]).find("chunked") != std::string::npos) {
        req.error_status = 411; req.error_message = "chunked request bodies are not supported";
        return false;
    }

    auto it_exp = req.headers.find("expect");
    if (it_exp != req.headers.end() && ascii_lower(it_exp->second).find("100-continue") != std::string::npos) {
        static const char* cont = "HTTP/1.1 100 Continue\r\n\r\n";
        if (!send_all(s, cont, strlen(cont))) return false;
    }

    while (body.size() < content_length) {
        if (!recv_chunk(s, body)) { req.error_status = 400; req.error_message = "incomplete request body"; return false; }
    }
    body.resize(content_length);
    req.body = std::move(body);
    return true;
}

static inline bool send_http_response(socket_t s, const HttpResponse& r) {
    std::ostringstream out;
    out << "HTTP/1.1 " << r.status << " " << http_reason(r.status) << "\r\n";
    out << "Content-Type: " << r.content_type << "\r\n";
    out << "Content-Length: " << r.body.size() << "\r\n";
    out << "Connection: close\r\n";
    out << "Access-Control-Allow-Origin: *\r\n";
    out << "Access-Control-Allow-Headers: *\r\n";
    out << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
    out << "Server: bruce\r\n";
    for (const auto& kv : r.headers) out << kv.first << ": " << kv.second << "\r\n";
    out << "\r\n";
    std::string head = out.str();
    return send_all(s, head.data(), head.size()) && send_all(s, r.body.data(), r.body.size());
}

static inline bool send_json(socket_t s, int status, const mj::Value& v,
                             const std::map<std::string, std::string>& extra = {}) {
    HttpResponse r;
    r.status = status;
    r.content_type = "application/json; charset=utf-8";
    r.body = mj::dump(v);
    r.headers = extra;
    return send_http_response(s, r);
}

static inline mj::Value json_error(const std::string& message, const std::string& type = "invalid_request_error") {
    mj::Value err = mj::Value::obj();
    mj::Value body = mj::Value::obj();
    body["message"] = message;
    body["type"] = type;
    body["param"] = mj::Value();
    body["code"] = mj::Value();
    err["error"] = std::move(body);
    return err;
}

struct SseWriter {
    socket_t sock = k_invalid_socket;
    bool ok = true;

    bool begin(socket_t s, int status = 200) {
        sock = s;
        std::ostringstream out;
        out << "HTTP/1.1 " << status << " " << http_reason(status) << "\r\n";
        out << "Content-Type: text/event-stream; charset=utf-8\r\n";
        out << "Cache-Control: no-cache\r\n";
        out << "Connection: close\r\n";
        out << "Transfer-Encoding: chunked\r\n";
        out << "X-Accel-Buffering: no\r\n";
        out << "Access-Control-Allow-Origin: *\r\n";
        out << "Access-Control-Allow-Headers: *\r\n";
        out << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
        out << "Server: bruce\r\n\r\n";
        std::string head = out.str();
        ok = send_all(sock, head.data(), head.size());
        return ok;
    }

    bool chunk(const std::string& payload) {
        if (!ok) return false;
        char hex[32];
        std::snprintf(hex, sizeof(hex), "%zx", payload.size());
        std::string frame;
        frame.reserve(payload.size() + 16);
        frame += hex;
        frame += "\r\n";
        frame += payload;
        frame += "\r\n";
        ok = send_all(sock, frame.data(), frame.size());
        return ok;
    }

    bool event(const std::string& json) {
        std::string payload = "data: ";
        payload += json;
        payload += "\n\n";
        return chunk(payload);
    }

    bool comment(const std::string& text) {
        std::string payload = ": ";
        payload += text;
        payload += "\n\n";
        return chunk(payload);
    }

    bool finish() {
        if (!ok) return false;
        static const char* end = "0\r\n\r\n";
        ok = send_all(sock, end, strlen(end));
        return ok;
    }
};

// Holds back an incomplete UTF-8 sequence between token pieces.
struct Utf8Carry {
    std::string buf;

    static size_t complete_prefix(const std::string& s) {
        size_t i = 0;
        while (i < s.size()) {
            unsigned char c = (unsigned char)s[i];
            size_t n = 0;
            if (c < 0x80) n = 1;
            else if ((c & 0xE0) == 0xC0) n = 2;
            else if ((c & 0xF0) == 0xE0) n = 3;
            else if ((c & 0xF8) == 0xF0) n = 4;
            else { i++; continue; }
            if (i + n > s.size()) break;
            bool valid = true;
            for (size_t k = 1; k < n; ++k) {
                if (((unsigned char)s[i + k] & 0xC0) != 0x80) { valid = false; break; }
            }
            if (!valid) { i++; continue; }
            i += n;
        }
        return i;
    }

    void feed(const std::string& in, std::string& out) {
        buf += in;
        size_t n = complete_prefix(buf);
        out.append(buf.data(), n);
        buf.erase(0, n);
    }

    void flush(std::string& out) {
        out += buf;
        buf.clear();
    }
};

// Buffers text so a multi-token stop string is never emitted past its match.
struct StopFilter {
    std::vector<std::string> stops;
    std::string pending;
    size_t max_len = 0;
    bool hit = false;

    void set(std::vector<std::string> s) {
        stops.clear();
        for (auto& x : s) if (!x.empty()) stops.push_back(std::move(x));
        max_len = 0;
        for (const auto& x : stops) max_len = std::max(max_len, x.size());
    }

    bool feed(const std::string& in, std::string& out) {
        if (hit) return true;
        pending += in;
        size_t best = std::string::npos;
        for (const auto& s : stops) {
            size_t p = pending.find(s);
            if (p != std::string::npos && p < best) best = p;
        }
        if (best != std::string::npos) {
            out.append(pending.data(), best);
            pending.clear();
            hit = true;
            return true;
        }
        size_t keep = max_len > 0 ? max_len - 1 : 0;
        if (pending.size() > keep) {
            size_t emit = pending.size() - keep;
            out.append(pending.data(), emit);
            pending.erase(0, emit);
        }
        return false;
    }

    void flush(std::string& out) {
        if (!hit) {
            out += pending;
            pending.clear();
        }
    }
};
