// Minimal GGUF v3 reader (metadata + tensor table).  The header and metadata are
// read into a prefix buffer and the tensor payload is streamed through read_at()
// (ReadFile on Windows, pread on POSIX) instead of being memory-mapped; see the
// loader in bruce.cu.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

enum ggml_type_t : uint32_t {
    GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1, GGML_TYPE_Q4_0 = 2, GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6, GGML_TYPE_Q5_1 = 7, GGML_TYPE_Q8_0 = 8, GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10, GGML_TYPE_Q3_K = 11, GGML_TYPE_Q4_K = 12, GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14, GGML_TYPE_Q8_K = 15, GGML_TYPE_IQ2_XXS = 16, GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ3_XXS = 18, GGML_TYPE_IQ1_S = 19, GGML_TYPE_IQ4_NL = 20, GGML_TYPE_IQ3_S = 21,
    GGML_TYPE_IQ2_S = 22, GGML_TYPE_IQ4_XS = 23, GGML_TYPE_I8 = 24, GGML_TYPE_I16 = 25,
    GGML_TYPE_I32 = 26, GGML_TYPE_I64 = 27, GGML_TYPE_F64 = 28, GGML_TYPE_IQ1_M = 29,
    GGML_TYPE_BF16 = 30, GGML_TYPE_Q4_0_4_4 = 31, GGML_TYPE_Q4_0_4_8 = 32, GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0 = 34, GGML_TYPE_TQ2_0 = 35, GGML_TYPE_MXFP4 = 39,
};

inline size_t ggml_type_size(uint32_t t) {
    switch (t) {
        case GGML_TYPE_F32: return 4;
        case GGML_TYPE_F16: return 2;
        case GGML_TYPE_BF16: return 2;
        case GGML_TYPE_Q4_0: return 18;   // 32 vals
        case GGML_TYPE_Q4_1: return 20;
        case GGML_TYPE_Q5_0: return 22;
        case GGML_TYPE_Q5_1: return 24;
        case GGML_TYPE_Q8_0: return 34;
        case GGML_TYPE_Q8_1: return 36;
        case GGML_TYPE_Q2_K: return 84;   // 256 vals
        case GGML_TYPE_Q3_K: return 110;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q5_K: return 176;
        case GGML_TYPE_Q6_K: return 210;
        case GGML_TYPE_IQ4_NL: return 18;
        case GGML_TYPE_MXFP4: return 17;
        default: return 0;
    }
}
inline int ggml_blck_size(uint32_t t) {
    switch (t) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16: return 1;
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q8_1: case GGML_TYPE_IQ4_NL: case GGML_TYPE_MXFP4: return 32;
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: return 256;
        default: return 1;
    }
}

struct GgufTensor {
    std::string name;
    uint32_t type = 0;
    uint32_t ndim = 0;
    uint64_t ne[4] = {1,1,1,1};
    uint64_t offset = 0;   // relative to data section
    uint64_t nbytes = 0;
    int block = -1;        // blk.N if name matches
    std::string suffix;
};

struct GgufValue {
    uint32_t type = 0;     // 8 = string, 9 = array, ...
    uint32_t arr_type = 0;
    std::string s;
    std::vector<std::string> sa;
    std::vector<int64_t> ia;
    std::vector<float> fa;
    double num = 0;
};

struct GgufFile {
    size_t map_size = 0;                 // file size
    const uint8_t* base = nullptr;       // metadata bytes; null when weights are streamed
    std::vector<uint8_t> meta;           // header + metadata + tensor table
#ifdef _WIN32
    void* read_h = nullptr;    // FILE_FLAG_SEQUENTIAL_SCAN handle for the streamed weight loader
#else
    int read_fd = -1;          // O_RDONLY descriptor used by pread() in read_at()
#endif
    uint64_t n_tensors = 0;
    uint64_t data_offset = 0;
    std::vector<GgufTensor> tensors;
    std::unordered_map<std::string, GgufValue> kv;
    std::unordered_map<std::string, int> by_name;

    void open(const char* path);
    void close();
    void parse_meta(size_t n);
    bool read_at(void* dst, size_t n, uint64_t off) const;
    const GgufTensor* find(const std::string& name) const {
        auto it = by_name.find(name);
        return it == by_name.end() ? nullptr : &tensors[it->second];
    }
    std::string str(const std::string& k, const std::string& def = "") const {
        auto it = kv.find(k); return it == kv.end() ? def : it->second.s;
    }
    int64_t i(const std::string& k, int64_t def = 0) const {
        auto it = kv.find(k);
        if (it == kv.end()) return def;
        return (int64_t)it->second.num;
    }
    float f(const std::string& k, float def = 0) const {
        auto it = kv.find(k); return it == kv.end() ? def : (float)it->second.num;
    }
    const std::vector<float>& fa(const std::string& k) const {
        static std::vector<float> empty;
        auto it = kv.find(k); return it == kv.end() ? empty : it->second.fa;
    }
};

struct GgufReader {
    const uint8_t* p = nullptr;
    const uint8_t* end = nullptr;
    template <typename T> T rd() {
        if (p + sizeof(T) > end) throw std::runtime_error("gguf: unexpected eof");
        T v; memcpy(&v, p, sizeof(T)); p += sizeof(T); return v;
    }
    std::string rstr() {
        uint64_t n = rd<uint64_t>();
        if (p + n > end) throw std::runtime_error("gguf: bad string");
        std::string s((const char*)p, (size_t)n); p += n; return s;
    }
    void skip(size_t n) { p += n; }
};

inline void GgufFile::open(const char* path) {
#ifdef _WIN32
    // No memory mapping: a mapped file makes every later ReadFile on it run at
    // raw disk speed (measured 392 MB/s vs 7400 MB/s from the OS cache), which
    // costs more than the mapping saves.  The header, metadata and tensor table
    // are read into a prefix buffer; the weights are streamed through read_at().
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error(std::string("cannot open ") + path);
    read_h = (void*)h;
    LARGE_INTEGER sz; GetFileSizeEx(h, &sz);
    map_size = (size_t)sz.QuadPart;
    size_t want = (size_t)std::min<uint64_t>((uint64_t)map_size, 32ull << 20);
    for (;;) {
        meta.resize(want);
        if (!read_at(meta.data(), want, 0)) throw std::runtime_error("gguf: read failed");
        try {
            parse_meta(want);
            break;
        } catch (const std::runtime_error& e) {
            const char* m = e.what();
            const bool eof = std::strstr(m, "unexpected eof") || std::strstr(m, "bad string");
            if (!eof || want >= map_size) throw;
            want = (size_t)std::min<uint64_t>((uint64_t)map_size, (uint64_t)want * 2);
        }
    }
    base = nullptr;   // tensor_data() is not usable; every byte comes from read_at()
#else
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) throw std::runtime_error(std::string("cannot open ") + path);
    read_fd = fd;
    struct stat st;
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        read_fd = -1;
        throw std::runtime_error("gguf: fstat failed");
    }
    map_size = (size_t)st.st_size;
    size_t want = (size_t)std::min<uint64_t>((uint64_t)map_size, 32ull << 20);
    for (;;) {
        meta.resize(want);
        if (!read_at(meta.data(), want, 0)) throw std::runtime_error("gguf: read failed");
        try {
            parse_meta(want);
            break;
        } catch (const std::runtime_error& e) {
            const char* m = e.what();
            const bool eof = std::strstr(m, "unexpected eof") || std::strstr(m, "bad string");
            if (!eof || want >= map_size) throw;
            want = (size_t)std::min<uint64_t>((uint64_t)map_size, (uint64_t)want * 2);
        }
    }
    base = nullptr;   // read_at() streams the weights from the descriptor
#endif
}

inline void GgufFile::parse_meta(size_t n) {
    base = meta.data();
    GgufReader r{base, base + n};
    tensors.clear(); by_name.clear(); kv.clear();
    uint32_t magic = r.rd<uint32_t>();
    if (magic != 0x46554747u) throw std::runtime_error("not a gguf file");
    uint32_t version = r.rd<uint32_t>();
    if (version != 3) throw std::runtime_error("unsupported gguf version");
    n_tensors = r.rd<uint64_t>();
    uint64_t n_kv = r.rd<uint64_t>();

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = r.rstr();
        uint32_t t = r.rd<uint32_t>();
        GgufValue v; v.type = t;
        switch (t) {
            case 0: v.num = r.rd<uint8_t>(); break;
            case 1: v.num = r.rd<int8_t>(); break;
            case 2: v.num = r.rd<uint16_t>(); break;
            case 3: v.num = r.rd<int16_t>(); break;
            case 4: v.num = r.rd<uint32_t>(); break;
            case 5: v.num = r.rd<int32_t>(); break;
            case 6: v.num = r.rd<float>(); break;
            case 7: v.num = r.rd<uint8_t>() ? 1 : 0; break;
            case 8: v.s = r.rstr(); break;
            case 10: v.num = (double)r.rd<uint64_t>(); break;
            case 11: v.num = (double)r.rd<int64_t>(); break;
            case 12: v.num = r.rd<double>(); break;
            case 9: {
                v.arr_type = r.rd<uint32_t>();
                uint64_t n = r.rd<uint64_t>();
                switch (v.arr_type) {
                    case 8: for (uint64_t k = 0; k < n; ++k) v.sa.push_back(r.rstr()); break;
                    case 6: for (uint64_t k = 0; k < n; ++k) v.fa.push_back(r.rd<float>()); break;
                    case 4: for (uint64_t k = 0; k < n; ++k) v.ia.push_back(r.rd<uint32_t>()); break;
                    case 5: for (uint64_t k = 0; k < n; ++k) v.ia.push_back(r.rd<int32_t>()); break;
                    case 10: for (uint64_t k = 0; k < n; ++k) v.ia.push_back((int64_t)r.rd<uint64_t>()); break;
                    case 11: for (uint64_t k = 0; k < n; ++k) v.ia.push_back(r.rd<int64_t>()); break;
                    case 0: for (uint64_t k = 0; k < n; ++k) v.ia.push_back(r.rd<uint8_t>()); break;
                    case 1: for (uint64_t k = 0; k < n; ++k) v.ia.push_back(r.rd<int8_t>()); break;
                    case 2: for (uint64_t k = 0; k < n; ++k) v.ia.push_back(r.rd<uint16_t>()); break;
                    case 3: for (uint64_t k = 0; k < n; ++k) v.ia.push_back(r.rd<int16_t>()); break;
                    case 7: for (uint64_t k = 0; k < n; ++k) v.ia.push_back(r.rd<uint8_t>()); break;
                    case 12: for (uint64_t k = 0; k < n; ++k) v.fa.push_back((float)r.rd<double>()); break;
                    default: throw std::runtime_error("gguf: unsupported array type");
                }
                break;
            }
            default: throw std::runtime_error("gguf: unsupported kv type");
        }
        kv[key] = std::move(v);
    }

    uint64_t align = 32;
    {
        auto it = kv.find("general.alignment");
        if (it != kv.end()) align = (uint64_t)it->second.num;
    }

    tensors.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensor t;
        t.name = r.rstr();
        t.ndim = r.rd<uint32_t>();
        if (t.ndim > 4) throw std::runtime_error("gguf: too many dims");
        for (uint32_t d = 0; d < t.ndim; ++d) t.ne[d] = r.rd<uint64_t>();
        t.type = r.rd<uint32_t>();
        t.offset = r.rd<uint64_t>();
        size_t ts = ggml_type_size(t.type);
        int bs = ggml_blck_size(t.type);
        if (!ts) throw std::runtime_error("gguf: unsupported tensor type " + std::to_string(t.type) + " for " + t.name);
        uint64_t n = 1;
        for (uint32_t d = 0; d < t.ndim; ++d) n *= t.ne[d];
        t.nbytes = n / bs * ts;
        // parse blk.N.suffix
        if (t.name.rfind("blk.", 0) == 0) {
            size_t dot = t.name.find('.', 4);
            if (dot != std::string::npos) {
                t.block = atoi(t.name.substr(4, dot - 4).c_str());
                t.suffix = t.name.substr(dot + 1);
            }
        }
        by_name[t.name] = (int)tensors.size();
        tensors.push_back(std::move(t));
    }
    uint64_t pos = (uint64_t)(r.p - base);
    data_offset = (pos + align - 1) / align * align;
}

inline void GgufFile::close() {
#ifdef _WIN32
    if (read_h) CloseHandle((HANDLE)read_h);
    read_h = nullptr;
#else
    if (read_fd >= 0) ::close(read_fd);
    read_fd = -1;
#endif
    base = nullptr;
}

// Read n bytes at an absolute file offset.  Used by the streamed weight loader;
// reads come from one thread, so a seek+ReadFile pair is enough (the sequential
// scan hint keeps the OS read-ahead going).
inline bool GgufFile::read_at(void* dst, size_t n, uint64_t off) const {
#ifdef _WIN32
    if (!read_h) {
        if (!base || off + n > map_size) return false;
        memcpy(dst, base + off, n);
        return true;
    }
    HANDLE h = (HANDLE)read_h;
    LARGE_INTEGER li; li.QuadPart = (LONGLONG)off;
    if (!SetFilePointerEx(h, li, nullptr, FILE_BEGIN)) return false;
    uint8_t* p = (uint8_t*)dst;
    size_t left = n;
    while (left) {
        DWORD want = (DWORD)(left > (1u << 30) ? (1u << 30) : left);
        DWORD got = 0;
        if (!ReadFile(h, p, want, &got, nullptr) || got == 0) return false;
        p += got; left -= got;
    }
    return true;
#else
    if (read_fd < 0) {
        if (!base || off + n > map_size) return false;
        memcpy(dst, base + off, n);
        return true;
    }
    uint8_t* p = (uint8_t*)dst;
    size_t left = n;
    uint64_t at = off;
    while (left) {
        ssize_t got = pread(read_fd, p, left, (off_t)at);
        if (got < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (got == 0) return false;
        p += got;
        left -= (size_t)got;
        at += (uint64_t)got;
    }
    return true;
#endif
}
