// Tiny dependency-free JSON value / parser / writer used by the Bruce HTTP server.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mj {

class Value {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::vector<std::pair<std::string, Value>> object;

    Value() = default;
    Value(std::nullptr_t) {}
    Value(bool v) : type(Type::Bool), boolean(v) {}
    Value(int v) : type(Type::Number), number((double)v) {}
    Value(int64_t v) : type(Type::Number), number((double)v) {}
    Value(uint32_t v) : type(Type::Number), number((double)v) {}
    Value(uint64_t v) : type(Type::Number), number((double)v) {}
    Value(float v) : type(Type::Number), number((double)v) {}
    Value(double v) : type(Type::Number), number(v) {}
    Value(const char* v) : type(Type::String), string(v ? v : "") {}
    Value(const std::string& v) : type(Type::String), string(v) {}
    Value(std::string&& v) : type(Type::String), string(std::move(v)) {}

    static Value arr() { Value v; v.type = Type::Array; return v; }
    static Value obj() { Value v; v.type = Type::Object; return v; }

    bool is_null() const { return type == Type::Null; }
    bool is_bool() const { return type == Type::Bool; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }

    size_t size() const {
        if (type == Type::Array) return array.size();
        if (type == Type::Object) return object.size();
        if (type == Type::String) return string.size();
        return 0;
    }

    bool has(const std::string& key) const {
        if (type != Type::Object) return false;
        for (const auto& kv : object) if (kv.first == key) return true;
        return false;
    }

    Value& operator[](const std::string& key) {
        if (type != Type::Object) { type = Type::Object; object.clear(); }
        for (auto& kv : object) if (kv.first == key) return kv.second;
        object.emplace_back(key, Value());
        return object.back().second;
    }

    const Value& at(const std::string& key) const {
        static const Value null_value;
        if (type != Type::Object) return null_value;
        for (const auto& kv : object) if (kv.first == key) return kv.second;
        return null_value;
    }

    const Value& operator[](const std::string& key) const { return at(key); }

    Value& operator[](size_t index) {
        if (type != Type::Array) { type = Type::Array; array.clear(); }
        if (index >= array.size()) array.resize(index + 1);
        return array[index];
    }

    const Value& at(size_t index) const {
        static const Value null_value;
        if (type != Type::Array || index >= array.size()) return null_value;
        return array[index];
    }

    void push(Value v) {
        if (type != Type::Array) { type = Type::Array; array.clear(); }
        array.push_back(std::move(v));
    }

    std::string as_string(const std::string& def = "") const {
        if (type == Type::String) return string;
        return def;
    }

    double as_number(double def = 0.0) const {
        if (type == Type::Number) return number;
        if (type == Type::Bool) return boolean ? 1.0 : 0.0;
        return def;
    }

    int as_int(int def = 0) const {
        double d = as_number((double)def);
        if (!std::isfinite(d)) return def;
        if (d > 2147483647.0) return 2147483647;
        if (d < -2147483648.0) return -2147483648;
        return (int)d;
    }

    int64_t as_i64(int64_t def = 0) const {
        double d = as_number((double)def);
        if (!std::isfinite(d)) return def;
        if (d > 9.22e18) return (int64_t)9223372036854775807LL;
        if (d < -9.22e18) return (int64_t)(-9223372036854775807LL - 1);
        return (int64_t)d;
    }

    float as_float(float def = 0.0f) const {
        return (float)as_number((double)def);
    }

    bool as_bool(bool def = false) const {
        if (type == Type::Bool) return boolean;
        if (type == Type::Number) return number != 0.0;
        if (type == Type::String) return !string.empty() && string != "false" && string != "0";
        return def;
    }
};

namespace detail {
inline void append_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back((char)cp);
    } else if (cp <= 0x7FF) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

inline int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
} // namespace detail

class Parser {
public:
    explicit Parser(const std::string& text) : begin_(text.data()), p_(text.data()), end_(text.data() + text.size()) {}

    Value parse() {
        skip_ws();
        Value v = parse_value(0);
        skip_ws();
        if (p_ != end_) fail("trailing characters after JSON value");
        return v;
    }

private:
    const char* begin_;
    const char* p_;
    const char* end_;
    std::string error_;

    [[noreturn]] void fail(const std::string& msg) const {
        size_t off = (size_t)(p_ - begin_);
        throw std::runtime_error("JSON error at byte " + std::to_string(off) + ": " + msg);
    }

    void skip_ws() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\r' || *p_ == '\n')) ++p_;
    }

    char peek() const {
        if (p_ >= end_) fail("unexpected end of input");
        return *p_;
    }

    void expect(char c) {
        if (p_ >= end_ || *p_ != c) fail(std::string("expected '") + c + "'");
        ++p_;
    }

    bool consume_literal(const char* lit) {
        size_t n = std::strlen(lit);
        if ((size_t)(end_ - p_) < n) return false;
        if (std::memcmp(p_, lit, n) != 0) return false;
        p_ += n;
        return true;
    }

    Value parse_value(int depth) {
        if (depth > 128) fail("nesting too deep");
        skip_ws();
        char c = peek();
        if (c == 'n') {
            if (!consume_literal("null")) fail("invalid literal");
            return Value();
        }
        if (c == 't') {
            if (!consume_literal("true")) fail("invalid literal");
            return Value(true);
        }
        if (c == 'f') {
            if (!consume_literal("false")) fail("invalid literal");
            return Value(false);
        }
        if (c == '"') return Value(parse_string());
        if (c == '[') return parse_array(depth + 1);
        if (c == '{') return parse_object(depth + 1);
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        fail("unexpected character");
    }

    Value parse_array(int depth) {
        Value v = Value::arr();
        expect('[');
        skip_ws();
        if (p_ < end_ && *p_ == ']') { ++p_; return v; }
        while (true) {
            v.array.push_back(parse_value(depth));
            skip_ws();
            if (p_ < end_ && *p_ == ',') { ++p_; skip_ws(); continue; }
            expect(']');
            break;
        }
        return v;
    }

    Value parse_object(int depth) {
        Value v = Value::obj();
        expect('{');
        skip_ws();
        if (p_ < end_ && *p_ == '}') { ++p_; return v; }
        while (true) {
            skip_ws();
            if (peek() != '"') fail("object key must be a string");
            std::string key = parse_string();
            skip_ws();
            expect(':');
            Value child = parse_value(depth);
            bool replaced = false;
            for (auto& kv : v.object) {
                if (kv.first == key) { kv.second = std::move(child); replaced = true; break; }
            }
            if (!replaced) v.object.emplace_back(std::move(key), std::move(child));
            skip_ws();
            if (p_ < end_ && *p_ == ',') { ++p_; continue; }
            expect('}');
            break;
        }
        return v;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (p_ < end_) {
            unsigned char c = (unsigned char)*p_++;
            if (c == '"') return out;
            if (c == '\\') {
                if (p_ >= end_) fail("unterminated escape");
                char e = *p_++;
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        uint32_t cp = parse_hex4();
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            if (p_ + 1 < end_ && p_[0] == '\\' && p_[1] == 'u') {
                                p_ += 2;
                                uint32_t lo = parse_hex4();
                                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                } else {
                                    detail::append_utf8(out, 0xFFFD);
                                    cp = lo;
                                }
                            } else {
                                cp = 0xFFFD;
                            }
                        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                            cp = 0xFFFD;
                        }
                        detail::append_utf8(out, cp);
                        break;
                    }
                    default: fail("invalid escape");
                }
            } else {
                if (c < 0x20) fail("control character in string");
                out.push_back((char)c);
            }
        }
        fail("unterminated string");
    }

    uint32_t parse_hex4() {
        if (end_ - p_ < 4) fail("short \\u escape");
        uint32_t cp = 0;
        for (int i = 0; i < 4; ++i) {
            int h = detail::hex_value(*p_++);
            if (h < 0) fail("invalid hex digit");
            cp = (cp << 4) | (uint32_t)h;
        }
        return cp;
    }

    Value parse_number() {
        const char* start = p_;
        if (*p_ == '-') ++p_;
        if (p_ >= end_) fail("short number");
        if (*p_ == '0') {
            ++p_;
        } else if (*p_ >= '1' && *p_ <= '9') {
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        } else {
            fail("invalid number");
        }
        if (p_ < end_ && *p_ == '.') {
            ++p_;
            if (p_ >= end_ || *p_ < '0' || *p_ > '9') fail("invalid fraction");
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        }
        if (p_ < end_ && (*p_ == 'e' || *p_ == 'E')) {
            ++p_;
            if (p_ < end_ && (*p_ == '+' || *p_ == '-')) ++p_;
            if (p_ >= end_ || *p_ < '0' || *p_ > '9') fail("invalid exponent");
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        }
        std::string tmp(start, (size_t)(p_ - start));
        char* endp = nullptr;
        double d = std::strtod(tmp.c_str(), &endp);
        if (!endp || *endp != '\0' || !std::isfinite(d)) fail("number out of range");
        return Value(d);
    }
};

inline Value parse(const std::string& text) {
    return Parser(text).parse();
}

inline void dump_string(std::string& out, const std::string& s) {
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
                    out += buf;
                } else {
                    out.push_back((char)c);
                }
        }
    }
    out.push_back('"');
}

inline void dump_value(std::string& out, const Value& v) {
    switch (v.type) {
        case Value::Type::Null: out += "null"; break;
        case Value::Type::Bool: out += v.boolean ? "true" : "false"; break;
        case Value::Type::Number: {
            double d = v.number;
            if (!std::isfinite(d)) { out += "null"; break; }
            double id = std::floor(d);
            if (d == id && std::fabs(d) < 9.0e15) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%lld", (long long)d);
                out += buf;
            } else {
                char buf[40];
                std::snprintf(buf, sizeof(buf), "%.17g", d);
                out += buf;
            }
            break;
        }
        case Value::Type::String: dump_string(out, v.string); break;
        case Value::Type::Array: {
            out.push_back('[');
            for (size_t i = 0; i < v.array.size(); ++i) {
                if (i) out.push_back(',');
                dump_value(out, v.array[i]);
            }
            out.push_back(']');
            break;
        }
        case Value::Type::Object: {
            out.push_back('{');
            for (size_t i = 0; i < v.object.size(); ++i) {
                if (i) out.push_back(',');
                dump_string(out, v.object[i].first);
                out.push_back(':');
                dump_value(out, v.object[i].second);
            }
            out.push_back('}');
            break;
        }
    }
}

inline std::string dump(const Value& v) {
    std::string out;
    out.reserve(256);
    dump_value(out, v);
    return out;
}

} // namespace mj
