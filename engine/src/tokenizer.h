// GPT-2 byte-level BPE tokenizer for the Dory GGUF (pre-tokenizer: pixtral/llama3).
// Uses the vendored llama.cpp unicode helpers so tokenization matches llama.cpp.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gguf.h"
#include "unicode.h"

struct Tokenizer {
    std::vector<std::string> tokens;      // raw GGUF token text
    std::vector<int32_t> token_type;      // 1 normal, 2 unknown, 3 control, 4 user-defined, 6 byte
    std::vector<std::string> pieces;      // decoded bytes for output
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> bpe_rank;
    std::vector<int32_t> special_ids;     // sorted by token text length desc

    int n_vocab = 0;
    int bos_id = 1;
    int eos_id = 11;
    int unk_id = 0;
    int pad_id = 11;
    bool add_bos = false;
    bool add_eos = false;
    std::string model_name;
    std::string pre_type;
    std::string chat_template;

    static const char* llama3_regex() {
        return R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    }

    bool is_special(int id) const {
        if (id < 0 || id >= n_vocab) return false;
        int t = token_type[id];
        return t == 2 || t == 3 || t == 4;
    }

    bool is_control(int id) const {
        if (id < 0 || id >= n_vocab) return false;
        int t = token_type[id];
        return t == 2 || t == 3;
    }

    bool is_eog(int id) const { return id == eos_id; }

    std::string decode_token(int id, bool special = true) const {
        if (id < 0 || id >= n_vocab) return std::string();
        if (!special && is_control(id)) return std::string();
        return pieces[(size_t)id];
    }

    std::string decode(const std::vector<int>& ids, bool skip_special = false) const {
        std::string out;
        for (int id : ids) {
            if (skip_special && is_control(id)) continue;
            out += decode_token(id, !skip_special);
        }
        return out;
    }

    void init(const GgufFile& g) {
        auto it_tok = g.kv.find("tokenizer.ggml.tokens");
        auto it_type = g.kv.find("tokenizer.ggml.token_type");
        auto it_merge = g.kv.find("tokenizer.ggml.merges");
        if (it_tok == g.kv.end() || it_tok->second.sa.empty())
            throw std::runtime_error("GGUF has no tokenizer.ggml.tokens");
        if (it_merge == g.kv.end())
            throw std::runtime_error("GGUF has no tokenizer.ggml.merges");

        tokens = it_tok->second.sa;
        n_vocab = (int)tokens.size();

        token_type.assign((size_t)n_vocab, 1);
        if (it_type != g.kv.end() && !it_type->second.ia.empty()) {
            size_t n = std::min((size_t)n_vocab, it_type->second.ia.size());
            for (size_t i = 0; i < n; ++i) token_type[i] = (int32_t)it_type->second.ia[i];
        }

        model_name = g.str("tokenizer.ggml.model", "gpt2");
        pre_type = g.str("tokenizer.ggml.pre", "");
        bos_id = (int)g.i("tokenizer.ggml.bos_token_id", bos_id);
        eos_id = (int)g.i("tokenizer.ggml.eos_token_id", eos_id);
        unk_id = (int)g.i("tokenizer.ggml.unknown_token_id", unk_id);
        pad_id = (int)g.i("tokenizer.ggml.padding_token_id", pad_id);
        add_bos = g.i("tokenizer.ggml.add_bos_token", 0) != 0;
        add_eos = g.i("tokenizer.ggml.add_eos_token", 0) != 0;
        chat_template = g.str("tokenizer.chat_template", "");

        token_to_id.clear();
        token_to_id.reserve((size_t)n_vocab * 2);
        for (int i = 0; i < n_vocab; ++i) token_to_id.emplace(tokens[(size_t)i], (int32_t)i);

        bpe_rank.clear();
        const auto& merges = it_merge->second.sa;
        bpe_rank.reserve(merges.size() * 2);
        for (size_t i = 0; i < merges.size(); ++i) {
            const std::string& word = merges[i];
            size_t pos = word.find(' ', 1);
            if (pos == std::string::npos) continue;
            std::string key = word.substr(0, pos);
            key.push_back('\x01');
            key += word.substr(pos + 1);
            bpe_rank.emplace(std::move(key), (int32_t)i);
        }

        special_ids.clear();
        for (int i = 0; i < n_vocab; ++i) {
            int t = token_type[(size_t)i];
            if (t == 2 || t == 3 || t == 4) special_ids.push_back(i);
        }
        std::stable_sort(special_ids.begin(), special_ids.end(), [&](int a, int b) {
            return tokens[(size_t)a].size() > tokens[(size_t)b].size();
        });

        pieces.resize((size_t)n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            int t = token_type[(size_t)i];
            if (t == 2 || t == 3 || t == 4) {
                pieces[(size_t)i] = tokens[(size_t)i];
            } else {
                pieces[(size_t)i] = decode_byte_text(tokens[(size_t)i]);
            }
        }
    }

    std::vector<int> encode(const std::string& text, bool with_add_special = false,
                            bool parse_special = true) const {
        std::vector<int> out;
        out.reserve(text.size() + 8);
        if (with_add_special && add_bos && bos_id >= 0) out.push_back(bos_id);

        size_t pos = 0;
        while (pos < text.size()) {
            size_t sp_pos = std::string::npos;
            size_t sp_len = 0;
            int sp_id = -1;
            find_special(text, pos, parse_special, sp_pos, sp_len, sp_id);
            if (sp_id >= 0 && sp_pos != std::string::npos) {
                if (sp_pos > pos) encode_raw(text.substr(pos, sp_pos - pos), out);
                out.push_back(sp_id);
                pos = sp_pos + sp_len;
            } else {
                encode_raw(text.substr(pos), out);
                break;
            }
        }

        if (with_add_special && add_eos && eos_id >= 0) out.push_back(eos_id);
        return out;
    }

private:
    struct Sym {
        std::string text;
        int prev = -1;
        int next = -1;
        bool alive = true;
    };

    struct Bigram {
        int left = -1;
        int right = -1;
        int rank = 0;
        std::string text;
    };

    struct BigramCmp {
        bool operator()(const Bigram& a, const Bigram& b) const {
            if (a.rank != b.rank) return a.rank > b.rank;
            return a.left > b.left;
        }
    };

    static std::string decode_byte_text(const std::string& text) {
        std::string out;
        out.reserve(text.size());
        size_t off = 0;
        while (off < text.size()) {
            size_t n = unicode_len_utf8(text[off]);
            if (n == 0 || off + n > text.size()) n = 1;
            std::string cp = text.substr(off, n);
            try {
                out.push_back((char)unicode_utf8_to_byte(cp));
            } catch (...) {
                out += cp;
            }
            off += n;
        }
        return out;
    }

    void find_special(const std::string& text, size_t from, bool parse_special,
                      size_t& best_pos, size_t& best_len, int& best_id) const {
        best_pos = std::string::npos;
        best_len = 0;
        best_id = -1;
        for (int id : special_ids) {
            int t = token_type[(size_t)id];
            if (!parse_special && (t == 2 || t == 3)) continue;
            const std::string& s = tokens[(size_t)id];
            if (s.empty()) continue;
            size_t p = text.find(s, from);
            if (p == std::string::npos) continue;
            if (p == from) {
                // special_ids is sorted longest-first, so this is the longest match here.
                best_pos = p;
                best_len = s.size();
                best_id = id;
                return;
            }
            if (best_pos == std::string::npos || p < best_pos) {
                best_pos = p;
                best_len = s.size();
                best_id = id;
            }
        }
    }

    void encode_raw(const std::string& text, std::vector<int>& out) const {
        if (text.empty()) return;
        std::vector<std::string> words = unicode_regex_split(text, {std::string(llama3_regex())}, true);
        for (const std::string& word : words) {
            if (word.empty()) continue;
            auto direct = token_to_id.find(word);
            if (direct != token_to_id.end()) {
                out.push_back(direct->second);
                continue;
            }
            bpe_word(word, out);
        }
    }

    void add_bigram(std::vector<Sym>& syms,
                    std::priority_queue<Bigram, std::vector<Bigram>, BigramCmp>& q,
                    int left, int right) const {
        if (left < 0 || right < 0) return;
        if (left >= (int)syms.size() || right >= (int)syms.size()) return;
        if (!syms[(size_t)left].alive || !syms[(size_t)right].alive) return;
        if (syms[(size_t)left].next != right) return;
        std::string l = syms[(size_t)left].text;
        std::string r = syms[(size_t)right].text;
        std::string key = l;
        key.push_back('\x01');
        key += r;
        auto it = bpe_rank.find(key);
        if (it == bpe_rank.end()) return;
        Bigram b;
        b.left = left;
        b.right = right;
        b.rank = it->second;
        b.text = l + r;
        q.push(std::move(b));
    }

    void bpe_word(const std::string& word, std::vector<int>& out) const {
        std::vector<Sym> syms;
        syms.reserve(word.size());
        size_t off = 0;
        while (off < word.size()) {
            size_t n = unicode_len_utf8(word[off]);
            if (n == 0 || off + n > word.size()) n = 1;
            Sym s;
            s.text = word.substr(off, n);
            syms.push_back(std::move(s));
            off += n;
        }
        if (syms.empty()) return;
        for (size_t i = 0; i < syms.size(); ++i) {
            syms[i].prev = i ? (int)i - 1 : -1;
            syms[i].next = i + 1 < syms.size() ? (int)i + 1 : -1;
        }

        std::priority_queue<Bigram, std::vector<Bigram>, BigramCmp> q;
        for (size_t i = 1; i < syms.size(); ++i) add_bigram(syms, q, (int)i - 1, (int)i);

        while (!q.empty()) {
            Bigram b = q.top();
            q.pop();
            if (b.left < 0 || b.right < 0) continue;
            if (b.left >= (int)syms.size() || b.right >= (int)syms.size()) continue;
            Sym& left = syms[(size_t)b.left];
            Sym& right = syms[(size_t)b.right];
            if (!left.alive || !right.alive) continue;
            if (left.next != b.right) continue;
            if (left.text + right.text != b.text) continue;

            left.text += right.text;
            left.next = right.next;
            right.alive = false;
            if (left.next >= 0) syms[(size_t)left.next].prev = b.left;
            add_bigram(syms, q, left.prev, b.left);
            add_bigram(syms, q, b.left, left.next);
        }

        int idx = 0;
        while (idx >= 0 && idx < (int)syms.size() && !syms[(size_t)idx].alive) idx = syms[(size_t)idx].next;
        while (idx >= 0 && idx < (int)syms.size()) {
            const std::string& s = syms[(size_t)idx].text;
            auto it = token_to_id.find(s);
            if (it != token_to_id.end()) {
                out.push_back(it->second);
            } else {
                size_t off2 = 0;
                while (off2 < s.size()) {
                    size_t n = unicode_len_utf8(s[off2]);
                    if (n == 0 || off2 + n > s.size()) n = 1;
                    std::string cp = s.substr(off2, n);
                    auto jt = token_to_id.find(cp);
                    if (jt != token_to_id.end()) out.push_back(jt->second);
                    else if (unk_id >= 0) out.push_back(unk_id);
                    off2 += n;
                }
            }
            idx = syms[(size_t)idx].next;
        }
    }
};
