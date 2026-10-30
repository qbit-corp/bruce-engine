// OpenAI-compatible HTTP server for the Bruce engine (single-slot, prefix-cached).
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "chat.h"
#include "http.h"
#include "session.h"

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model_id = "bruce";
    std::string model_path;
    std::string model_name;
    std::string api_key;
    int n_ctx = 262144;
    int n_ctx_slot = 262144;   // per-slot capacity when slots > 1
    int slots = 1;
    int n_predict = 256;
    int n_batch = MBUF;
    int n_loops = 2;
    int n_keep = 0;
    int draft = 0;
    int draft_ngram = 4;
    int draft_nmin = -1;
    bool greedy = false;
    bool no_live = false;
    bool verbose = false;
    bool timings = true;
    bool enable_thinking = true;
    bool medium_effort = false;
    bool truncate_history_thinking = true;
    bool default_cache_prompt = true;
    float default_temp = 1.0f;
    float default_top_p = 0.95f;
    int default_top_k = 40;
    float default_min_p = 0.0f;
    float default_repeat_penalty = 1.0f;
    int default_repeat_last_n = 64;
    float default_frequency_penalty = 0.0f;
    float default_presence_penalty = 0.0f;
    uint64_t default_seed = 0;
    bool seed_set = false;
};

// State of one engine slot, published to /slots and /metrics.  The engine keeps
// one KV partition per slot; with --slots 1 this is the historical single-slot
// server.  A plain struct guarded by state_mu is enough.
struct SlotState {
    long long req_id = -1;
    std::string kind;
    std::string phase = "idle";
    std::string model;
    std::string prompt_preview;
    int prompt_tokens = 0;
    int prompt_processed = 0;
    int prompt_cached = 0;
    int generated = 0;
    int max_tokens = 0;
    long long start_epoch = 0;
    double started_ms = 0.0;
};

struct ServerContext {
    Tokenizer* tok = nullptr;
    ServerConfig cfg;
    std::vector<std::unique_ptr<Session>> sessions;
    BatchCoordinator batcher;
    std::mutex slot_mu;
    std::condition_variable slot_cv;
    std::vector<char> slot_busy;
    std::mutex state_mu;
    std::vector<SlotState> slots;
    std::atomic<bool> stop{false};
    std::atomic<long long> next_req{0};
    std::atomic<int> active{0};
    std::atomic<long long> requests_total{0};
    std::atomic<long long> requests_completed{0};
    std::atomic<long long> requests_failed{0};
    std::atomic<long long> prompt_tokens_total{0};
    std::atomic<long long> completion_tokens_total{0};
    std::atomic<long long> cached_tokens_total{0};
    std::atomic<long long> prefill_us_total{0};
    std::atomic<long long> decode_us_total{0};
    std::chrono::steady_clock::time_point start_time;

    ServerContext(Tokenizer& t, const ServerConfig& c) : tok(&t), cfg(c) {
        start_time = std::chrono::steady_clock::now();
    }
};

static std::atomic<bool> g_srv_stop{false};

static inline long long epoch_now() { return (long long)std::time(nullptr); }

static inline std::string make_id(const char* prefix) {
    static std::atomic<unsigned long long> counter{0};
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s-%llx%llx", prefix,
                  (unsigned long long)epoch_now(), (unsigned long long)counter.fetch_add(1));
    return std::string(buf);
}

static inline std::string path_only(const std::string& p) {
    size_t q = p.find('?');
    return q == std::string::npos ? p : p.substr(0, q);
}

static inline bool srv_should_stop(ServerContext& ctx) {
    return g_srv_stop.load() || ctx.stop.load();
}

static inline uint64_t srv_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static inline uint64_t choice_seed(uint64_t base, int choice) {
    return choice == 0 ? base : srv_splitmix64(base + (uint64_t)choice * 0x9E3779B97F4A7C15ull);
}

static inline uint64_t random_seed() {
    uint64_t r = (uint64_t)std::random_device{}();
    r ^= (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    r ^= (uint64_t)(size_t)&r;
    return srv_splitmix64(r);
}

static inline mj::Value usage_json(int prompt, int completion, int cached = 0) {
    mj::Value u = mj::Value::obj();
    u["prompt_tokens"] = prompt;
    u["completion_tokens"] = completion;
    u["total_tokens"] = prompt + completion;
    mj::Value pd = mj::Value::obj();
    pd["cached_tokens"] = cached;
    u["prompt_tokens_details"] = std::move(pd);
    return u;
}

static inline mj::Value timings_json(int prompt_n, double prompt_ms, int predicted_n,
                                     double predicted_ms, double ttft_ms) {
    mj::Value t = mj::Value::obj();
    if (ttft_ms < 0.0) ttft_ms = prompt_ms;
    t["prompt_n"] = prompt_n;
    t["prompt_ms"] = prompt_ms;
    t["prompt_per_token_ms"] = prompt_n > 0 ? prompt_ms / prompt_n : 0.0;
    t["prompt_per_second"] = prompt_ms > 1e-6 ? prompt_n * 1000.0 / prompt_ms : 0.0;
    t["predicted_n"] = predicted_n;
    t["predicted_ms"] = predicted_ms;
    t["predicted_per_token_ms"] = predicted_n > 0 ? predicted_ms / predicted_n : 0.0;
    t["predicted_per_second"] = predicted_ms > 1e-6 ? predicted_n * 1000.0 / predicted_ms : 0.0;
    t["ttft_ms"] = ttft_ms;
    t["total_ms"] = prompt_ms + predicted_ms;
    return t;
}

static inline mj::Value generation_settings_json(const ServerConfig& cfg, const GenParams* gp = nullptr) {
    mj::Value o = mj::Value::obj();
    o["n_predict"] = gp ? gp->max_tokens : cfg.n_predict;
    o["temperature"] = gp ? gp->temp : cfg.default_temp;
    o["top_p"] = gp ? gp->top_p : cfg.default_top_p;
    o["top_k"] = gp ? gp->top_k : cfg.default_top_k;
    o["min_p"] = gp ? gp->min_p : cfg.default_min_p;
    o["repeat_penalty"] = gp ? gp->repeat_penalty : cfg.default_repeat_penalty;
    o["repeat_last_n"] = gp ? gp->repeat_last_n : cfg.default_repeat_last_n;
    o["frequency_penalty"] = gp ? gp->frequency_penalty : cfg.default_frequency_penalty;
    o["presence_penalty"] = gp ? gp->presence_penalty : cfg.default_presence_penalty;
    o["n_keep"] = gp ? gp->n_keep : cfg.n_keep;
    o["cache_prompt"] = gp ? gp->cache_prompt : cfg.default_cache_prompt;
    o["n_ctx"] = cfg.n_ctx;
    o["n_batch"] = cfg.n_batch;
    o["draft"] = gp ? gp->draft : cfg.draft;
    o["draft_ngram"] = gp ? gp->draft_ngram : cfg.draft_ngram;
    o["draft_nmin"] = gp ? gp->draft_nmin : cfg.draft_nmin;
    o["seed"] = cfg.seed_set ? (int64_t)cfg.default_seed : (int64_t)-1;
    if (gp) o["seed"] = (int64_t)gp->seed;
    return o;
}

static inline mj::Value logprob_entry(const TokenPieceInfo& p) {
    mj::Value o = mj::Value::obj();
    o["token"] = p.text;
    o["logprob"] = p.logprob;
    o["bytes"] = mj::Value();
    mj::Value top = mj::Value::arr();
    for (const auto& t : p.top) {
        mj::Value e = mj::Value::obj();
        e["token"] = t.token;
        e["logprob"] = t.logprob;
        e["bytes"] = mj::Value();
        top.push(std::move(e));
    }
    o["top_logprobs"] = std::move(top);
    return o;
}

static inline void apply_rng(Session& sess, uint64_t seed) {
    sess.set_rng(seed);
}

static inline bool parse_gen_params(const ServerConfig& cfg, const mj::Value& body,
                                    bool chat_request, GenParams& gp, std::string& err) {
    auto get_num = [&](std::initializer_list<const char*> keys, double def) -> double {
        for (const char* k : keys) if (body.has(k)) return body.at(k).as_number(def);
        return def;
    };
    auto get_int = [&](std::initializer_list<const char*> keys, int def) -> int {
        for (const char* k : keys) if (body.has(k)) return body.at(k).as_int(def);
        return def;
    };
    auto get_bool = [&](std::initializer_list<const char*> keys, bool def) -> bool {
        for (const char* k : keys) if (body.has(k)) return body.at(k).as_bool(def);
        return def;
    };

    gp.max_tokens = get_int({"max_tokens", "max_completion_tokens", "n_predict", "max_gen_tokens", "num_predict"},
                            cfg.n_predict);
    gp.temp = (float)get_num({"temperature", "temp"}, cfg.default_temp);
    gp.greedy = cfg.greedy || gp.temp <= 0.0f;
    gp.greedy = get_bool({"greedy"}, gp.greedy);
    gp.top_p = (float)get_num({"top_p", "top-p"}, cfg.default_top_p);
    gp.top_k = get_int({"top_k", "top-k"}, cfg.default_top_k);
    gp.min_p = (float)get_num({"min_p", "min-p"}, cfg.default_min_p);
    gp.repeat_penalty = (float)get_num({"repeat_penalty", "repetition_penalty", "repeat-penalty"},
                                       cfg.default_repeat_penalty);
    gp.repeat_last_n = get_int({"repeat_last_n", "repeat_penalty_last_n", "penalty_last_n", "repeat-last-n"},
                               cfg.default_repeat_last_n);
    gp.frequency_penalty = (float)get_num({"frequency_penalty", "frequency-penalty"},
                                          cfg.default_frequency_penalty);
    gp.presence_penalty = (float)get_num({"presence_penalty", "presence-penalty"},
                                         cfg.default_presence_penalty);
    gp.ignore_eos = get_bool({"ignore_eos", "ignore_eos_token"}, false);
    gp.min_tokens = get_int({"min_tokens"}, 0);
    gp.cache_prompt = get_bool({"cache_prompt", "cache"}, cfg.default_cache_prompt);
    gp.n_keep = get_int({"n_keep", "keep"}, cfg.n_keep);

    if (gp.max_tokens < -1) gp.max_tokens = -1;
    if (gp.greedy) gp.temp = 0.0f;
    if (gp.top_p < 0.0f || gp.top_p > 1.0f) { err = "top_p must be in [0, 1]"; return false; }
    if (gp.min_p < 0.0f || gp.min_p > 1.0f) { err = "min_p must be in [0, 1]"; return false; }
    if (gp.top_k < 0) { err = "top_k must be >= 0"; return false; }
    if (gp.repeat_penalty <= 0.0f) { err = "repeat_penalty must be > 0"; return false; }
    if (gp.frequency_penalty < -2.0f || gp.frequency_penalty > 2.0f) {
        err = "frequency_penalty must be in [-2, 2]"; return false;
    }
    if (gp.presence_penalty < -2.0f || gp.presence_penalty > 2.0f) {
        err = "presence_penalty must be in [-2, 2]"; return false;
    }
    if (gp.min_tokens < 0) gp.min_tokens = 0;
    if (gp.n_keep < 0) gp.n_keep = 0;
    if (gp.repeat_last_n < 0) gp.repeat_last_n = Session::kMaxHist;
    gp.repeat_last_n = std::min(gp.repeat_last_n, Session::kMaxHist);

    gp.draft = get_int({"draft", "draft_max", "n_draft"}, cfg.draft);
    if (body.has("speculative")) {
        gp.draft = body.at("speculative").as_bool(false)
                       ? std::max(1, cfg.draft > 0 ? cfg.draft : 4)
                       : 0;
    }
    gp.draft = std::max(0, std::min(7, gp.draft));
    gp.draft_ngram = get_int({"draft_ngram", "draft-ngram"}, cfg.draft_ngram);
    gp.draft_nmin = get_int({"draft_nmin", "draft-nmin"}, cfg.draft_nmin);

    uint64_t seed = cfg.seed_set ? cfg.default_seed : random_seed();
    if (body.has("seed")) {
        int64_t s = body.at("seed").as_i64((int64_t)seed);
        seed = s < 0 ? random_seed() : (uint64_t)s;
    }
    gp.seed = seed;

    gp.stop.clear();
    if (body.has("stop")) {
        const mj::Value& s = body.at("stop");
        if (s.is_string()) {
            if (!s.string.empty()) gp.stop.push_back(s.string);
        } else if (s.is_array()) {
            for (const auto& e : s.array) {
                if (e.is_string() && !e.string.empty()) gp.stop.push_back(e.string);
            }
        }
    }
    if (gp.stop.size() > 8) gp.stop.resize(8);

    gp.stop_token_ids.clear();
    if (body.has("stop_token_ids") && body.at("stop_token_ids").is_array()) {
        for (const auto& e : body.at("stop_token_ids").array) {
            int id = e.as_int(-1);
            if (id < 0 || id >= VOCAB) { err = "stop_token_ids contains an out-of-range token id"; return false; }
            gp.stop_token_ids.push_back(id);
        }
    }

    gp.logit_bias.clear();
    if (body.has("logit_bias") && body.at("logit_bias").is_object()) {
        for (const auto& kv : body.at("logit_bias").object) {
            char* endp = nullptr;
            long id = std::strtol(kv.first.c_str(), &endp, 10);
            if (!endp || *endp != '\0' || id < 0 || id >= VOCAB) continue;
            float bias = kv.second.as_float(0.0f);
            if (bias < -100.0f) bias = -100.0f;
            if (bias > 100.0f) bias = 100.0f;
            gp.logit_bias[(int)id] = bias;
        }
    }

    if (chat_request) {
        gp.logprobs = get_bool({"logprobs"}, false);
        gp.top_logprobs = get_int({"top_logprobs"}, 0);
        if (!gp.logprobs) gp.top_logprobs = 0;
    } else {
        if (body.has("logprobs")) {
            const mj::Value& lp = body.at("logprobs");
            if (lp.is_bool()) {
                gp.logprobs = lp.boolean;
                gp.top_logprobs = lp.boolean ? 5 : 0;
            } else if (lp.is_number()) {
                gp.top_logprobs = std::max(0, lp.as_int(0));
                gp.logprobs = gp.top_logprobs > 0;
            }
        }
        if (body.has("top_logprobs")) gp.top_logprobs = body.at("top_logprobs").as_int(gp.top_logprobs);
    }
    if (gp.top_logprobs < 0) gp.top_logprobs = 0;
    if (gp.top_logprobs > 20) gp.top_logprobs = 20;
    if (gp.logprobs && gp.top_logprobs == 0) gp.top_logprobs = 5;
    if (gp.top_logprobs > 0) gp.logprobs = true;

    return true;
}

static inline void parse_chat_template_options(const mj::Value& body, ChatTemplateOptions& opts) {
    if (body.has("enable_thinking")) opts.enable_thinking = body.at("enable_thinking").as_bool(opts.enable_thinking);
    if (body.has("medium_effort")) opts.medium_effort = body.at("medium_effort").as_bool(opts.medium_effort);
    if (body.has("truncate_history_thinking")) {
        opts.truncate_history_thinking = body.at("truncate_history_thinking").as_bool(opts.truncate_history_thinking);
    }
    if (body.has("reasoning_effort")) {
        std::string e = ascii_lower(body.at("reasoning_effort").as_string());
        if (e == "none" || e == "off") opts.enable_thinking = false;
        else if (e == "low" || e == "minimal" || e == "efficient") opts.medium_effort = true;
        else if (e == "medium" || e == "high") opts.medium_effort = false;
    }
    if (body.has("chat_template_kwargs") && body.at("chat_template_kwargs").is_object()) {
        const auto& kw = body.at("chat_template_kwargs");
        if (kw.has("enable_thinking")) opts.enable_thinking = kw.at("enable_thinking").as_bool(opts.enable_thinking);
        if (kw.has("medium_effort")) opts.medium_effort = kw.at("medium_effort").as_bool(opts.medium_effort);
        if (kw.has("truncate_history_thinking")) {
            opts.truncate_history_thinking =
                kw.at("truncate_history_thinking").as_bool(opts.truncate_history_thinking);
        }
    }
}

// ------------------------------------------------------------------ slot pool

static inline int acquire_slot(ServerContext& ctx, const std::vector<int>& prompt) {
    std::unique_lock<std::mutex> lk(ctx.slot_mu);
    ctx.slot_cv.wait(lk, [&] {
        if (srv_should_stop(ctx)) return true;
        for (char b : ctx.slot_busy) if (!b) return true;
        return false;
    });
    if (srv_should_stop(ctx)) return -1;
    int best = -1, best_hit = -1;
    for (int i = 0; i < (int)ctx.slot_busy.size(); ++i) {
        if (ctx.slot_busy[(size_t)i]) continue;
        // cache affinity: a returning conversation usually lands on the slot that
        // already holds its prefix
        int hit = ctx.sessions[(size_t)i]->cache_hit(prompt);
        if (hit > best_hit) { best_hit = hit; best = i; }
    }
    if (best < 0) return -1;
    ctx.slot_busy[(size_t)best] = 1;
    ctx.batcher.enter();
    return best;
}

static inline void release_slot(ServerContext& ctx, int slot_id) {
    if (slot_id < 0) return;
    {
        std::lock_guard<std::mutex> lk(ctx.slot_mu);
        ctx.slot_busy[(size_t)slot_id] = 0;
    }
    ctx.batcher.leave();
    ctx.slot_cv.notify_one();
}

// ------------------------------------------------------------------ live slot state

struct SlotGuard {
    ServerContext& ctx;
    int slot_id = -1;
    long long req_id = -1;

    SlotGuard(ServerContext& c, long long id, std::string kind, int prompt_tokens, int max_tokens,
              int sid)
        : ctx(c), slot_id(sid), req_id(id) {
        std::lock_guard<std::mutex> lk(ctx.state_mu);
        SlotState& s = ctx.slots[(size_t)slot_id];
        s.req_id = id;
        s.kind = std::move(kind);
        s.phase = "prefill";
        s.model = ctx.cfg.model_id;
        s.prompt_preview.clear();
        s.prompt_tokens = prompt_tokens;
        s.prompt_processed = 0;
        s.prompt_cached = 0;
        s.generated = 0;
        s.max_tokens = max_tokens;
        s.start_epoch = epoch_now();
        s.started_ms = now_ms();
    }

    ~SlotGuard() {
        {
            std::lock_guard<std::mutex> lk(ctx.state_mu);
            if (ctx.slots[(size_t)slot_id].req_id == req_id) ctx.slots[(size_t)slot_id] = SlotState{};
        }
        release_slot(ctx, slot_id);
    }

    SlotState& st() { return ctx.slots[(size_t)slot_id]; }

    void set_prompt_preview(const std::string& text) {
        std::lock_guard<std::mutex> lk(ctx.state_mu);
        if (st().req_id == req_id) st().prompt_preview = text;
    }

    void set_prompt(int tokens, int max_tokens) {
        std::lock_guard<std::mutex> lk(ctx.state_mu);
        if (st().req_id != req_id) return;
        st().prompt_tokens = tokens;
        st().max_tokens = max_tokens;
        st().phase = "prefill";
    }

    void prefill_progress(int processed, int cached) {
        std::lock_guard<std::mutex> lk(ctx.state_mu);
        if (st().req_id != req_id) return;
        st().phase = "prefill";
        st().prompt_processed = processed;
        st().prompt_cached = cached;
    }

    void prefill_done(const PromptStats& ps) {
        std::lock_guard<std::mutex> lk(ctx.state_mu);
        if (st().req_id != req_id) return;
        st().phase = "decode";
        st().prompt_tokens = ps.prompt_tokens;
        st().prompt_processed = ps.processed_tokens;
        st().prompt_cached = ps.cached_tokens;
    }

    void set_generated(int n) {
        std::lock_guard<std::mutex> lk(ctx.state_mu);
        if (st().req_id == req_id) st().generated = n;
    }

    void set_phase(const char* phase) {
        std::lock_guard<std::mutex> lk(ctx.state_mu);
        if (st().req_id == req_id) st().phase = phase;
    }
};

struct GenerationContext {
    ServerContext* ctx = nullptr;
    SlotGuard* slot = nullptr;
    long long req_id = 0;
    int slot_id = 0;
    double decode_start = 0.0;
    double last_status = 0.0;
    int prompt_len = 0;
    int max_tokens = 0;
};

static inline void gen_progress(GenerationContext& gc, int generated) {
    if (gc.slot) gc.slot->set_generated(generated);
    if (g_no_live) return;
    double now = now_ms();
    if (g_tty || now - gc.last_status > 500.0) {
        double el = now - gc.decode_start;
        double tps = el > 1.0 ? generated * 1000.0 / el : 0.0;
        int ctx_used = gc.prompt_len + generated;
        status_line("[bruce] #%lld s%d decode %d/%d | %.1f tok/s | ctx %d/%d | %.1fs | eta %.1fs",
                    gc.req_id, gc.slot_id, generated, gc.max_tokens, tps, ctx_used,
                    gc.ctx ? gc.ctx->cfg.n_ctx_slot : 0, el / 1000.0,
                    tps > 1.0 && generated < gc.max_tokens
                        ? (gc.max_tokens - generated) / tps
                        : 0.0);
        gc.last_status = now;
    }
}

static inline PromptStats prepare_logged(ServerContext& ctx, Session& sess, SlotGuard* slot,
                                         long long req_id, const char* kind,
                                         const std::vector<int>& prompt,
                                         bool use_cache, bool count_prompt_usage,
                                         const std::function<bool()>& cancel) {
    int hit = use_cache ? sess.cache_hit(prompt) : 0;
    log_line("[bruce] #%lld %s | %zu prompt tokens | cache hit %d (%.0f%%) | batch %d",
             req_id, kind, prompt.size(), hit,
             prompt.empty() ? 0.0 : 100.0 * hit / (double)prompt.size(), ctx.cfg.n_batch);
    if (slot) slot->prefill_progress(0, hit);
    double last = now_ms();
    auto progress = [&](int done, int total, double ms) {
        if (slot) slot->prefill_progress(done, hit);
        if (g_no_live) return;
        double now = now_ms();
        if (g_tty || now - last > 500.0) {
            double tps = ms > 1.0 ? done * 1000.0 / ms : 0.0;
            status_line("[bruce] #%lld prefill %d/%d | %.0f tok/s | %.2fs | cached %d",
                        req_id, done, total, tps, ms / 1000.0, hit);
            last = now;
        }
    };
    PromptStats ps = sess.prepare(prompt, use_cache, ctx.cfg.n_batch, progress, cancel);
    status_end();
    if (ps.cancelled) {
        log_line("[bruce] #%lld %s cancelled during prefill", req_id, kind);
        return ps;
    }
    if (slot) slot->prefill_done(ps);
    log_line("[bruce] #%lld prefill %d processed (%d cached of %d) in %.1f ms (%.0f tok/s)",
             req_id, ps.processed_tokens, ps.cached_tokens, ps.prompt_tokens, ps.prefill_ms,
             ps.prefill_ms > 1e-6 ? ps.processed_tokens * 1000.0 / ps.prefill_ms : 0.0);
    if (count_prompt_usage) ctx.prompt_tokens_total += ps.prompt_tokens;
    ctx.cached_tokens_total += ps.cached_tokens;
    ctx.prefill_us_total += (long long)(ps.prefill_ms * 1000.0);
    return ps;
}

static inline void log_decode_done(ServerContext& ctx, long long req_id, const PromptStats& ps,
                                   const GenResult& gr, int max_tokens) {
    status_end();
    double tps = gr.total_ms > 1e-6 ? gr.completion_tokens * 1000.0 / gr.total_ms : 0.0;
    log_line("[bruce] #%lld done | prompt %d (%d cached) | gen %d/%d | %.1f tok/s | finish=%s%s",
             req_id, ps.prompt_tokens, ps.cached_tokens, gr.completion_tokens, max_tokens, tps,
             gr.finish_reason.c_str(), gr.cancelled ? " cancelled" : "");
    ctx.completion_tokens_total += gr.completion_tokens;
    ctx.decode_us_total += (long long)(gr.total_ms * 1000.0);
}

static inline bool check_auth(const HttpRequest& req, const ServerConfig& cfg) {
    if (cfg.api_key.empty()) return true;
    auto it = req.headers.find("authorization");
    if (it != req.headers.end()) {
        const std::string& v = it->second;
        if (v.rfind("Bearer ", 0) == 0 && v.substr(7) == cfg.api_key) return true;
        if (v == cfg.api_key) return true;
    }
    auto it2 = req.headers.find("x-api-key");
    if (it2 != req.headers.end() && it2->second == cfg.api_key) return true;
    return false;
}

static inline bool send_unauthorized(socket_t s) {
    return send_json(s, 401, json_error("invalid or missing API key", "authentication_error"));
}

static inline bool parse_body_json(const HttpRequest& req, mj::Value& out, std::string& err) {
    if (req.body.empty()) { out = mj::Value::obj(); return true; }
    try {
        out = mj::parse(req.body);
        if (!out.is_object()) { err = "request body must be a JSON object"; return false; }
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

static inline mj::Value completion_logprobs_json(const GenResult& gr) {
    mj::Value out = mj::Value::obj();
    mj::Value toks = mj::Value::arr();
    mj::Value lps = mj::Value::arr();
    mj::Value tops = mj::Value::arr();
    mj::Value offs = mj::Value::arr();
    int off = 0;
    for (const auto& p : gr.pieces) {
        toks.push(p.text);
        lps.push(p.logprob);
        offs.push(off);
        mj::Value tm = mj::Value::obj();
        for (const auto& t : p.top) tm[t.token] = t.logprob;
        tops.push(std::move(tm));
        off += (int)p.text.size();
    }
    out["tokens"] = std::move(toks);
    out["token_logprobs"] = std::move(lps);
    out["top_logprobs"] = std::move(tops);
    out["text_offset"] = std::move(offs);
    return out;
}

static inline mj::Value completion_delta_logprobs_json(const DeltaEvent& ev) {
    if (!ev.has_logprob) return mj::Value();
    mj::Value out = mj::Value::obj();
    mj::Value toks = mj::Value::arr();
    mj::Value lps = mj::Value::arr();
    mj::Value tops = mj::Value::arr();
    mj::Value offs = mj::Value::arr();
    toks.push(ev.text);
    lps.push(ev.logprob);
    offs.push(0);
    mj::Value tm = mj::Value::obj();
    if (ev.top) {
        for (const auto& t : *ev.top) tm[t.token] = t.logprob;
    }
    tops.push(std::move(tm));
    out["tokens"] = std::move(toks);
    out["token_logprobs"] = std::move(lps);
    out["top_logprobs"] = std::move(tops);
    out["text_offset"] = std::move(offs);
    return out;
}

static inline mj::Value chat_logprobs_json(const GenResult& gr) {
    mj::Value out = mj::Value::obj();
    mj::Value content = mj::Value::arr();
    for (const auto& p : gr.pieces) content.push(logprob_entry(p));
    out["content"] = std::move(content);
    return out;
}

static inline mj::Value logprob_delta_json(const DeltaEvent& ev) {
    mj::Value out = mj::Value::obj();
    mj::Value content = mj::Value::arr();
    if (ev.has_logprob) {
        mj::Value e = mj::Value::obj();
        e["token"] = ev.text;
        e["logprob"] = ev.logprob;
        e["bytes"] = mj::Value();
        mj::Value top = mj::Value::arr();
        if (ev.top) {
            for (const auto& t : *ev.top) {
                mj::Value te = mj::Value::obj();
                te["token"] = t.token;
                te["logprob"] = t.logprob;
                te["bytes"] = mj::Value();
                top.push(std::move(te));
            }
        }
        e["top_logprobs"] = std::move(top);
        content.push(std::move(e));
    }
    out["content"] = std::move(content);
    return out;
}

static inline std::string last_user_preview(const mj::Value& messages) {
    std::string text;
    if (messages.is_array()) {
        for (const auto& m : messages.array) {
            if (m.is_object() && m.at("role").as_string() == "user") {
                text = chat_content_text(m.at("content"));
            }
        }
    }
    if (text.size() > 240) text.resize(240);
    return text;
}

// ------------------------------------------------------------------ chat completions

static inline bool handle_chat_completions(socket_t s, ServerContext& ctx, const mj::Value& body) {
    const mj::Value& messages = body.at("messages");
    if (!messages.is_array() || messages.array.empty()) {
        return send_json(s, 400, json_error("messages must be a non-empty array"));
    }

    const mj::Value& tools = body.at("tools");
    bool tools_enabled = tools.is_array() && !tools.array.empty();
    mj::Value filtered_tools;
    const mj::Value* tools_for_prompt = tools_enabled ? &tools : nullptr;
    if (body.has("tool_choice") && !body.at("tool_choice").is_null()) {
        const mj::Value& tc = body.at("tool_choice");
        if (tc.is_string()) {
            std::string tcs = ascii_lower(tc.string);
            if (tcs == "none") {
                tools_enabled = false;
                tools_for_prompt = nullptr;
            } else if (tcs != "auto" && tcs != "required") {
                return send_json(s, 400,
                                 json_error("tool_choice must be auto, none, required, or a function object"));
            }
        } else if (tc.is_object()) {
            if (tools_for_prompt) {
                const std::string want = tc.at("function").at("name").as_string();
                if (!want.empty()) {
                    filtered_tools = mj::Value::arr();
                    for (const auto& e : tools.array) {
                        const mj::Value& fn = e.at("function").is_object() ? e.at("function") : e;
                        if (fn.at("name").as_string() == want) filtered_tools.push(e);
                    }
                    if (!filtered_tools.array.empty()) tools_for_prompt = &filtered_tools;
                }
            }
        } else {
            return send_json(s, 400,
                             json_error("tool_choice must be auto, none, required, or a function object"));
        }
    }

    ChatTemplateOptions opts;
    opts.enable_thinking = ctx.cfg.enable_thinking;
    opts.medium_effort = ctx.cfg.medium_effort;
    opts.truncate_history_thinking = ctx.cfg.truncate_history_thinking;
    parse_chat_template_options(body, opts);

    std::string rendered = render_chat_prompt(messages, opts, true, tools_for_prompt);
    if (ctx.cfg.verbose) {
        const std::string first_role = messages.array[0].is_object()
                                           ? messages.array[0].at("role").as_string()
                                           : std::string("?");
        std::string last_user;
        for (const auto& m : messages.array) {
            if (m.is_object() && m.at("role").as_string() == "user") {
                last_user = chat_content_text(m.at("content"));
            }
        }
        log_line("[bruce] chat: %zu messages | first=%s | tools=%zu | last user: %.160s",
                 messages.array.size(), first_role.c_str(),
                 tools_for_prompt ? tools_for_prompt->array.size() : 0, last_user.c_str());
        const size_t tail_from = rendered.size() > 320 ? rendered.size() - 320 : 0;
        log_line("[bruce] prompt tail: %s", rendered.substr(tail_from).c_str());
    }
    const bool prompt_thinking = prompt_opens_thinking(rendered);
    std::vector<int> prompt = ctx.tok->encode(rendered, false, true);
    if (prompt.empty()) prompt.push_back(ctx.tok->bos_id >= 0 ? ctx.tok->bos_id : 1);
    if ((int)prompt.size() >= ctx.cfg.n_ctx_slot) {
        return send_json(s, 400, json_error("prompt is longer than the configured context"));
    }

    GenParams gp;
    std::string err;
    if (!parse_gen_params(ctx.cfg, body, true, gp, err)) {
        return send_json(s, 400, json_error(err));
    }
    int n_choices = body.has("n") ? body.at("n").as_int(1) : 1;
    n_choices = std::max(1, std::min(8, n_choices));

    const int room = ctx.cfg.n_ctx_slot - (int)prompt.size();
    if (gp.max_tokens < 0) gp.max_tokens = room;
    gp.max_tokens = std::min(gp.max_tokens, room);
    if (gp.max_tokens < 0) gp.max_tokens = 0;

    bool stream = body.has("stream") && body.at("stream").as_bool(false);
    bool include_usage = false;
    if (body.has("stream_options") && body.at("stream_options").is_object()) {
        include_usage = body.at("stream_options").at("include_usage").as_bool(false);
    }

    const long long req_id = ++ctx.next_req;
    ctx.requests_total += 1;
    std::string model = body.has("model") ? body.at("model").as_string(ctx.cfg.model_id) : ctx.cfg.model_id;

    const int slot_id = acquire_slot(ctx, prompt);
    if (slot_id < 0) {
        ctx.requests_failed += 1;
        return send_json(s, 503, json_error("server is shutting down", "server_error"));
    }
    SlotGuard slot(ctx, req_id, "chat", (int)prompt.size(), gp.max_tokens, slot_id);
    Session& sess = *ctx.sessions[(size_t)slot_id];
    slot.set_prompt_preview(last_user_preview(messages));

    GenerationContext gc;
    gc.ctx = &ctx;
    gc.slot = &slot;
    gc.req_id = req_id;
    gc.slot_id = slot_id;
    gc.prompt_len = (int)prompt.size();
    gc.max_tokens = gp.max_tokens;

    SseWriter sse;
    if (stream && !sse.begin(s)) {
        ctx.requests_failed += 1;
        return false;
    }
    const std::string stream_id = make_id("chatcmpl");
    auto emit_chunk = [&](int index, mj::Value delta, mj::Value finish) -> bool {
        mj::Value ch = mj::Value::obj();
        ch["index"] = index;
        ch["delta"] = std::move(delta);
        ch["finish_reason"] = std::move(finish);
        mj::Value chunk = mj::Value::obj();
        chunk["id"] = stream_id;
        chunk["object"] = "chat.completion.chunk";
        chunk["created"] = epoch_now();
        chunk["model"] = model;
        mj::Value arr = mj::Value::arr();
        arr.push(std::move(ch));
        chunk["choices"] = std::move(arr);
        return sse.event(mj::dump(chunk));
    };

    mj::Value choices = mj::Value::arr();
    int total_prompt = 0;
    int total_completion = 0;
    int total_cached = 0;
    double total_prompt_ms = 0.0;
    double total_decode_ms = 0.0;
    double ttft_ms = -1.0;
    bool send_failed = false;

    for (int ci = 0; ci < n_choices; ++ci) {
        PromptStats ps = prepare_logged(ctx, sess, &slot, req_id, "chat", prompt, gp.cache_prompt, ci == 0,
                                        [&]() { return srv_should_stop(ctx); });
        if (ps.cancelled) { send_failed = true; break; }
        if (ci == 0) {
            total_prompt += ps.prompt_tokens;
            total_cached += ps.cached_tokens;
        }
        total_prompt_ms += ps.prefill_ms;

        gc.prompt_len = (int)prompt.size();
        gc.max_tokens = gp.max_tokens;
        gc.decode_start = now_ms();
        gc.last_status = gc.decode_start;
        slot.set_phase("decode");

        if (stream) {
            mj::Value role = mj::Value::obj();
            role["role"] = "assistant";
            mj::Value delta = mj::Value::obj();
            delta["delta"] = std::move(role);
            delta["index"] = ci;
            delta["finish_reason"] = mj::Value();
            mj::Value chunk = mj::Value::obj();
            chunk["id"] = stream_id;
            chunk["object"] = "chat.completion.chunk";
            chunk["created"] = epoch_now();
            chunk["model"] = model;
            mj::Value arr = mj::Value::arr();
            arr.push(std::move(delta));
            chunk["choices"] = std::move(arr);
            if (!sse.event(mj::dump(chunk))) { send_failed = true; break; }
        }

        ThinkSplitter thin;
        thin.thinking = prompt_thinking;
        const DeltaEvent* current_ev = nullptr;
        DeltaSink sink;
        if (stream) {
            thin.emit = [&](const std::string& text, bool reasoning) -> bool {
                if (text.empty()) return sse.ok;
                mj::Value delta = mj::Value::obj();
                if (reasoning) delta["reasoning_content"] = text;
                else if (!tools_enabled) delta["content"] = text;
                else return sse.ok;
                mj::Value ch = mj::Value::obj();
                ch["index"] = ci;
                ch["delta"] = std::move(delta);
                ch["finish_reason"] = mj::Value();
                if (gp.logprobs && current_ev && current_ev->has_logprob) {
                    ch["logprobs"] = logprob_delta_json(*current_ev);
                }
                mj::Value chunk = mj::Value::obj();
                chunk["id"] = stream_id;
                chunk["object"] = "chat.completion.chunk";
                chunk["created"] = epoch_now();
                chunk["model"] = model;
                mj::Value arr = mj::Value::arr();
                arr.push(std::move(ch));
                chunk["choices"] = std::move(arr);
                return sse.event(mj::dump(chunk));
            };
            sink = [&](const DeltaEvent& ev) -> bool {
                current_ev = &ev;
                thin.feed(ev.text);
                current_ev = nullptr;
                return sse.ok;
            };
        } else {
            sink = [](const DeltaEvent&) -> bool { return true; };
        }

        apply_rng(sess, choice_seed(gp.seed, ci));
        gp.cancelled = [&]() { return srv_should_stop(ctx); };
        GenResult gr = sess.generate(gp, prompt, sink,
                                            [&](int n) { gen_progress(gc, n); });
        total_completion += gr.completion_tokens;
        total_decode_ms += gr.total_ms;
        if (ttft_ms < 0.0) ttft_ms = ps.prefill_ms + (gr.ttft_ms > 0.0 ? gr.ttft_ms : 0.0);
        if (stream) {
            thin.finish();
            if (!sse.ok) { send_failed = true; break; }
        }

        std::string reasoning;
        std::string content;
        split_think_text(gr.text, reasoning, content, prompt_thinking);

        std::vector<ParsedToolCall> tool_calls;
        if (tools_enabled) {
            std::string tool_content;
            tool_calls = parse_tool_calls(content, tool_content, *tools_for_prompt);
            content = std::move(tool_content);
            if (!tool_calls.empty()) {
                gr.finish_reason = "tool_calls";
                log_line("[bruce] #%lld %zu tool call(s)", req_id, tool_calls.size());
            }
        }

        if (stream) {
            if (tools_enabled && !content.empty()) {
                mj::Value delta = mj::Value::obj();
                delta["content"] = content;
                if (!emit_chunk(ci, std::move(delta), mj::Value())) { send_failed = true; break; }
            }
            if (!tool_calls.empty()) {
                mj::Value delta = mj::Value::obj();
                mj::Value tcs = mj::Value::arr();
                for (size_t ti = 0; ti < tool_calls.size(); ++ti) {
                    ParsedToolCall& c = tool_calls[ti];
                    if (c.id.empty()) c.id = make_id("call");
                    mj::Value tc = mj::Value::obj();
                    tc["index"] = (int)ti;
                    tc["id"] = c.id;
                    tc["type"] = "function";
                    mj::Value fn = mj::Value::obj();
                    fn["name"] = c.name;
                    fn["arguments"] = c.arguments;
                    tc["function"] = std::move(fn);
                    tcs.push(std::move(tc));
                }
                delta["tool_calls"] = std::move(tcs);
                if (!emit_chunk(ci, std::move(delta), mj::Value())) { send_failed = true; break; }
            }
            mj::Value ch = mj::Value::obj();
            ch["index"] = ci;
            ch["delta"] = mj::Value::obj();
            ch["finish_reason"] = gr.finish_reason;
            if (gp.logprobs) ch["logprobs"] = chat_logprobs_json(gr);
            mj::Value chunk = mj::Value::obj();
            chunk["id"] = stream_id;
            chunk["object"] = "chat.completion.chunk";
            chunk["created"] = epoch_now();
            chunk["model"] = model;
            mj::Value arr = mj::Value::arr();
            arr.push(std::move(ch));
            chunk["choices"] = std::move(arr);
            if (!sse.event(mj::dump(chunk))) { send_failed = true; break; }
        } else {
            mj::Value msg = mj::Value::obj();
            msg["role"] = "assistant";
            if (!tool_calls.empty()) {
                msg["content"] = content.empty() ? mj::Value() : mj::Value(content);
                mj::Value tcs = mj::Value::arr();
                for (auto& c : tool_calls) {
                    if (c.id.empty()) c.id = make_id("call");
                    mj::Value tc = mj::Value::obj();
                    tc["id"] = c.id;
                    tc["type"] = "function";
                    mj::Value fn = mj::Value::obj();
                    fn["name"] = c.name;
                    fn["arguments"] = c.arguments;
                    tc["function"] = std::move(fn);
                    tcs.push(std::move(tc));
                }
                msg["tool_calls"] = std::move(tcs);
            } else {
                msg["content"] = content;
            }
            if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
            mj::Value choice = mj::Value::obj();
            choice["index"] = ci;
            choice["message"] = std::move(msg);
            choice["finish_reason"] = gr.finish_reason;
            if (gp.logprobs) choice["logprobs"] = chat_logprobs_json(gr);
            choices.push(std::move(choice));
        }

        log_decode_done(ctx, req_id, ps, gr, gp.max_tokens);
        if (gr.cancelled) { send_failed = true; break; }
    }

    if (send_failed) {
        if (stream) sse.finish();
        log_line("[bruce] #%lld chat aborted (client disconnected or request cancelled)", req_id);
        ctx.requests_failed += 1;
        return false;
    }

    if (stream) {
        if (include_usage) {
            mj::Value chunk = mj::Value::obj();
            chunk["id"] = stream_id;
            chunk["object"] = "chat.completion.chunk";
            chunk["created"] = epoch_now();
            chunk["model"] = model;
            chunk["choices"] = mj::Value::arr();
            chunk["usage"] = usage_json(total_prompt, total_completion, total_cached);
            sse.event(mj::dump(chunk));
        }
        sse.event("[DONE]");
        sse.finish();
        ctx.requests_completed += 1;
        return true;
    }

    mj::Value out = mj::Value::obj();
    out["id"] = make_id("chatcmpl");
    out["object"] = "chat.completion";
    out["created"] = epoch_now();
    out["model"] = model;
    out["choices"] = std::move(choices);
    out["usage"] = usage_json(total_prompt, total_completion, total_cached);
    out["timings"] = timings_json(total_prompt, total_prompt_ms, total_completion,
                                  total_decode_ms, ttft_ms);
    ctx.requests_completed += 1;
    return send_json(s, 200, out);
}

// ------------------------------------------------------------------ completions

struct CompletionPrompt {
    std::string text;
    std::vector<int> ids;
    bool from_ids = false;
};

static inline bool collect_completion_prompts(ServerContext& ctx, const mj::Value& body,
                                              std::vector<CompletionPrompt>& out, std::string& err) {
    auto add_ids = [&](const mj::Value& arr) -> bool {
        CompletionPrompt cp;
        cp.from_ids = true;
        for (const auto& e : arr.array) {
            if (!e.is_number()) { err = "prompt token array must contain integers"; return false; }
            int id = e.as_int(-1);
            if (id < 0 || id >= ctx.tok->n_vocab) { err = "prompt token id out of range"; return false; }
            cp.ids.push_back(id);
        }
        cp.text = ctx.tok->decode(cp.ids, false);
        out.push_back(std::move(cp));
        return true;
    };

    const mj::Value& p = body.at("prompt");
    if (p.is_string()) {
        CompletionPrompt cp;
        cp.text = p.string;
        cp.ids = ctx.tok->encode(cp.text, false, true);
        out.push_back(std::move(cp));
        return true;
    }
    if (p.is_array()) {
        if (p.array.empty()) { err = "prompt array must not be empty"; return false; }
        if (p.array[0].is_string()) {
            for (const auto& e : p.array) {
                if (!e.is_string()) { err = "prompt array must contain strings"; return false; }
                CompletionPrompt cp;
                cp.text = e.string;
                cp.ids = ctx.tok->encode(cp.text, false, true);
                out.push_back(std::move(cp));
            }
            return true;
        }
        if (p.array[0].is_number()) return add_ids(p);
        if (p.array[0].is_array()) {
            for (const auto& e : p.array) {
                if (!e.is_array()) { err = "prompt array must contain token arrays"; return false; }
                if (!add_ids(e)) return false;
            }
            return true;
        }
        err = "unsupported prompt array";
        return false;
    }
    err = "prompt must be a string or an array";
    return false;
}

static inline bool handle_completions(socket_t s, ServerContext& ctx, const mj::Value& body,
                                      bool native) {
    std::vector<CompletionPrompt> prompts;
    std::string err;
    if (!collect_completion_prompts(ctx, body, prompts, err)) {
        return send_json(s, 400, json_error(err));
    }
    for (auto& cp : prompts) {
        if (cp.ids.empty()) cp.ids.push_back(ctx.tok->bos_id >= 0 ? ctx.tok->bos_id : 1);
        if ((int)cp.ids.size() >= ctx.cfg.n_ctx_slot) {
            return send_json(s, 400, json_error("prompt is longer than the configured context"));
        }
    }

    GenParams gp;
    if (!parse_gen_params(ctx.cfg, body, false, gp, err)) {
        return send_json(s, 400, json_error(err));
    }
    bool stream = body.has("stream") && body.at("stream").as_bool(false);
    int n_choices = body.has("n") ? body.at("n").as_int(1) : 1;
    n_choices = std::max(1, std::min(8, n_choices));
    if (native) n_choices = 1;
    bool echo = body.has("echo") && body.at("echo").as_bool(false);
    if (stream && echo && !native) {
        return send_json(s, 400, json_error("echo is not supported when streaming"));
    }
    bool include_usage = false;
    if (body.has("stream_options") && body.at("stream_options").is_object()) {
        include_usage = body.at("stream_options").at("include_usage").as_bool(false);
    }

    const long long req_id = ++ctx.next_req;
    ctx.requests_total += 1;
    std::string model = body.has("model") ? body.at("model").as_string(ctx.cfg.model_id) : ctx.cfg.model_id;
    const int base_max = gp.max_tokens;

    const int slot_id = acquire_slot(ctx, prompts[0].ids);
    if (slot_id < 0) {
        ctx.requests_failed += 1;
        return send_json(s, 503, json_error("server is shutting down", "server_error"));
    }
    SlotGuard slot(ctx, req_id, native ? "completion" : "completions",
                   (int)prompts[0].ids.size(), base_max < 0 ? ctx.cfg.n_ctx_slot : base_max, slot_id);
    Session& sess = *ctx.sessions[(size_t)slot_id];

    GenerationContext gc;
    gc.ctx = &ctx;
    gc.slot = &slot;
    gc.req_id = req_id;
    gc.slot_id = slot_id;

    SseWriter sse;
    if (stream && !sse.begin(s)) {
        ctx.requests_failed += 1;
        return false;
    }
    const std::string stream_id = make_id("cmpl");

    mj::Value choices = mj::Value::arr();
    int total_prompt = 0;
    int total_completion = 0;
    int total_cached = 0;
    double total_prompt_ms = 0.0;
    double total_decode_ms = 0.0;
    double ttft_ms = -1.0;
    bool send_failed = false;

    for (size_t pi = 0; pi < prompts.size() && !send_failed; ++pi) {
        CompletionPrompt& cp = prompts[pi];
        const int room = ctx.cfg.n_ctx_slot - (int)cp.ids.size();
        int max_tokens = base_max < 0 ? room : std::min(base_max, room);
        if (max_tokens < 0) max_tokens = 0;
        gp.max_tokens = max_tokens;

        slot.set_prompt((int)cp.ids.size(), max_tokens);
        std::string preview = cp.text;
        if (preview.size() > 240) preview.resize(240);
        slot.set_prompt_preview(preview);

        gc.prompt_len = (int)cp.ids.size();
        gc.max_tokens = max_tokens;

        for (int ci = 0; ci < n_choices; ++ci) {
            PromptStats ps = prepare_logged(ctx, sess, &slot, req_id, native ? "completion" : "completions",
                                            cp.ids, gp.cache_prompt, ci == 0,
                                            [&]() { return srv_should_stop(ctx); });
            if (ps.cancelled) { send_failed = true; break; }
            if (ci == 0) {
                total_prompt += ps.prompt_tokens;
                total_cached += ps.cached_tokens;
            }
            total_prompt_ms += ps.prefill_ms;

            gc.decode_start = now_ms();
            gc.last_status = gc.decode_start;
            slot.set_phase("decode");

            int streamed_tokens = 0;
            DeltaSink sink;
            if (stream) {
                if (native) {
                    sink = [&](const DeltaEvent& ev) -> bool {
                        if (ev.token_id >= 0) streamed_tokens += 1;
                        mj::Value chunk = mj::Value::obj();
                        chunk["content"] = ev.text;
                        chunk["stop"] = false;
                        chunk["id_slot"] = 0;
                        chunk["model"] = model;
                        chunk["tokens_predicted"] = streamed_tokens;
                        chunk["tokens_evaluated"] = ps.prompt_tokens;
                        return sse.event(mj::dump(chunk));
                    };
                } else {
                    sink = [&](const DeltaEvent& ev) -> bool {
                        mj::Value ch = mj::Value::obj();
                        ch["text"] = ev.text;
                        ch["index"] = (int)(pi * (size_t)n_choices + (size_t)ci);
                        ch["logprobs"] = gp.logprobs ? completion_delta_logprobs_json(ev) : mj::Value();
                        ch["finish_reason"] = mj::Value();
                        mj::Value chunk = mj::Value::obj();
                        chunk["id"] = stream_id;
                        chunk["object"] = "text_completion";
                        chunk["created"] = epoch_now();
                        chunk["model"] = model;
                        mj::Value arr = mj::Value::arr();
                        arr.push(std::move(ch));
                        chunk["choices"] = std::move(arr);
                        return sse.event(mj::dump(chunk));
                    };
                }
            } else {
                sink = [](const DeltaEvent&) -> bool { return true; };
            }

            apply_rng(sess, choice_seed(gp.seed, (int)(pi * (size_t)n_choices + (size_t)ci)));
            gp.cancelled = [&]() { return srv_should_stop(ctx); };
            GenResult gr = sess.generate(gp, cp.ids, sink,
                                                [&](int n) { gen_progress(gc, n); });
            total_completion += gr.completion_tokens;
            total_decode_ms += gr.total_ms;
            if (ttft_ms < 0.0) ttft_ms = ps.prefill_ms + (gr.ttft_ms > 0.0 ? gr.ttft_ms : 0.0);

            if (stream) {
                mj::Value chunk = mj::Value::obj();
                if (native) {
                    chunk["content"] = "";
                    chunk["stop"] = true;
                    chunk["id_slot"] = 0;
                    chunk["model"] = model;
                    chunk["tokens_predicted"] = gr.completion_tokens;
                    chunk["tokens_evaluated"] = ps.prompt_tokens;
                    chunk["stopped_eos"] = gr.finish_reason == "stop";
                    chunk["stopped_limit"] = gr.finish_reason == "length";
                    chunk["stopped_word"] = false;
                    chunk["stopping_word"] = "";
                    chunk["timings"] = timings_json(ps.prompt_tokens, ps.prefill_ms,
                                                    gr.completion_tokens, gr.total_ms,
                                                    ps.prefill_ms + gr.ttft_ms);
                } else {
                    mj::Value ch = mj::Value::obj();
                    ch["text"] = "";
                    ch["index"] = (int)(pi * (size_t)n_choices + (size_t)ci);
                    ch["logprobs"] = mj::Value();
                    ch["finish_reason"] = gr.finish_reason;
                    chunk["id"] = stream_id;
                    chunk["object"] = "text_completion";
                    chunk["created"] = epoch_now();
                    chunk["model"] = model;
                    mj::Value arr = mj::Value::arr();
                    arr.push(std::move(ch));
                    chunk["choices"] = std::move(arr);
                }
                if (!sse.event(mj::dump(chunk))) { send_failed = true; break; }
            } else {
                mj::Value ch = mj::Value::obj();
                std::string text = gr.text;
                if (echo) text = cp.text + text;
                ch["text"] = text;
                ch["index"] = (int)(pi * (size_t)n_choices + (size_t)ci);
                ch["logprobs"] = gp.logprobs ? completion_logprobs_json(gr) : mj::Value();
                ch["finish_reason"] = gr.finish_reason;
                choices.push(std::move(ch));
            }
            log_decode_done(ctx, req_id, ps, gr, gp.max_tokens);
            if (gr.cancelled) { send_failed = true; break; }
        }
    }

    if (send_failed) {
        if (stream) sse.finish();
        log_line("[bruce] #%lld completion aborted (client disconnected or request cancelled)", req_id);
        ctx.requests_failed += 1;
        return false;
    }

    if (stream) {
        if (!native && include_usage) {
            mj::Value chunk = mj::Value::obj();
            chunk["id"] = stream_id;
            chunk["object"] = "text_completion";
            chunk["created"] = epoch_now();
            chunk["model"] = model;
            chunk["choices"] = mj::Value::arr();
            chunk["usage"] = usage_json(total_prompt, total_completion, total_cached);
            sse.event(mj::dump(chunk));
        }
        sse.event("[DONE]");
        sse.finish();
        ctx.requests_completed += 1;
        return true;
    }

    if (native) {
        mj::Value out = mj::Value::obj();
        out["content"] = choices.array.empty() ? "" : choices.array[0].at("text").as_string();
        out["stop"] = true;
        out["id_slot"] = 0;
        out["slot_id"] = 0;
        out["model"] = model;
        out["tokens_predicted"] = total_completion;
        out["tokens_evaluated"] = total_prompt;
        out["tokens_cached"] = total_cached;
        out["prompt"] = prompts.empty() ? "" : prompts[0].text;
        out["truncated"] = false;
        out["stopped_eos"] = choices.array.empty() ? false
                                                   : choices.array[0].at("finish_reason").as_string() == "stop";
        out["stopped_limit"] = choices.array.empty() ? false
                                                     : choices.array[0].at("finish_reason").as_string() == "length";
        out["stopped_word"] = false;
        out["stopping_word"] = "";
        out["timings"] = timings_json(total_prompt, total_prompt_ms, total_completion,
                                      total_decode_ms, ttft_ms);
        out["generation_settings"] = generation_settings_json(ctx.cfg, &gp);
        ctx.requests_completed += 1;
        return send_json(s, 200, out);
    }

    mj::Value out = mj::Value::obj();
    out["id"] = make_id("cmpl");
    out["object"] = "text_completion";
    out["created"] = epoch_now();
    out["model"] = model;
    out["choices"] = std::move(choices);
    out["usage"] = usage_json(total_prompt, total_completion, total_cached);
    out["timings"] = timings_json(total_prompt, total_prompt_ms, total_completion,
                                  total_decode_ms, ttft_ms);
    ctx.requests_completed += 1;
    return send_json(s, 200, out);
}

// ------------------------------------------------------------------ misc endpoints

static inline bool handle_tokenize(socket_t s, ServerContext& ctx, const mj::Value& body) {
    std::string content = body.has("content") ? body.at("content").as_string()
                                              : body.at("text").as_string();
    bool add_special = body.has("add_special") && body.at("add_special").as_bool(false);
    bool parse_special = body.has("parse_special") ? body.at("parse_special").as_bool(true) : true;
    bool with_pieces = body.has("with_pieces") && body.at("with_pieces").as_bool(false);
    std::vector<int> ids = ctx.tok->encode(content, add_special, parse_special);
    mj::Value out = mj::Value::obj();
    mj::Value arr = mj::Value::arr();
    for (int id : ids) {
        if (with_pieces) {
            mj::Value e = mj::Value::obj();
            e["id"] = id;
            e["piece"] = ctx.tok->decode_token(id, true);
            arr.push(std::move(e));
        } else {
            arr.push(id);
        }
    }
    out["tokens"] = std::move(arr);
    return send_json(s, 200, out);
}

static inline bool handle_detokenize(socket_t s, ServerContext& ctx, const mj::Value& body) {
    const mj::Value& toks = body.at("tokens");
    if (!toks.is_array()) return send_json(s, 400, json_error("tokens must be an array"));
    std::vector<int> ids;
    for (const auto& e : toks.array) {
        int id = e.as_int(-1);
        if (id < 0 || id >= ctx.tok->n_vocab) return send_json(s, 400, json_error("token id out of range"));
        ids.push_back(id);
    }
    mj::Value out = mj::Value::obj();
    out["content"] = ctx.tok->decode(ids, false);
    return send_json(s, 200, out);
}

static inline bool handle_props(socket_t s, ServerContext& ctx) {
    mj::Value out = mj::Value::obj();
    mj::Value def = generation_settings_json(ctx.cfg);
    out["default_generation_settings"] = def;
    out["generation_settings"] = std::move(def);
    out["total_slots"] = (int)ctx.slots.size();
    out["model_path"] = ctx.cfg.model_path;
    out["model_name"] = ctx.cfg.model_name;
    out["model_id"] = ctx.cfg.model_id;
    out["n_ctx"] = ctx.cfg.n_ctx;
    out["n_ctx_per_slot"] = ctx.cfg.n_ctx_slot;
    out["n_batch"] = ctx.cfg.n_batch;
    out["n_loops"] = ctx.cfg.n_loops;
    out["n_predict"] = ctx.cfg.n_predict;
    out["bos_token_id"] = ctx.tok->bos_id;
    out["eos_token_id"] = ctx.tok->eos_id;
    out["chat_template"] = ctx.tok->chat_template;
    out["enable_thinking"] = ctx.cfg.enable_thinking;
    out["medium_effort"] = ctx.cfg.medium_effort;
    out["truncate_history_thinking"] = ctx.cfg.truncate_history_thinking;
    out["default_cache_prompt"] = ctx.cfg.default_cache_prompt;
    out["draft"] = ctx.cfg.draft;
    out["draft_ngram"] = ctx.cfg.draft_ngram;
    out["draft_nmin"] = ctx.cfg.draft_nmin;
    out["build_info"] = "bruce";
    return send_json(s, 200, out);
}

static inline mj::Value model_json(const ServerConfig& cfg) {
    mj::Value m = mj::Value::obj();
    m["id"] = cfg.model_id;
    m["object"] = "model";
    m["created"] = epoch_now();
    m["owned_by"] = "local";
    // llama.cpp clients (pi-llama-cpp, llama-server UIs, ...) read the model's
    // context window from meta.n_ctx.  In multi-slot mode one request owns a
    // single KV partition, so the usable window is n_ctx / slots, not the
    // whole pool.  Reporting the pool here makes clients send prompts the
    // server then rejects with "prompt is longer than the configured context".
    mj::Value meta = mj::Value::obj();
    meta["n_ctx"] = cfg.n_ctx_slot;
    meta["n_vocab"] = VOCAB;
    m["meta"] = std::move(meta);
    m["n_ctx_per_slot"] = cfg.n_ctx_slot;
    return m;
}

static inline bool handle_models(socket_t s, ServerContext& ctx) {
    mj::Value data = mj::Value::arr();
    data.push(model_json(ctx.cfg));
    mj::Value out = mj::Value::obj();
    out["object"] = "list";
    out["data"] = std::move(data);
    return send_json(s, 200, out);
}

static inline bool handle_model(socket_t s, ServerContext& ctx, const std::string& id) {
    if (id != ctx.cfg.model_id) {
        return send_json(s, 404, json_error("model not found", "invalid_request_error"));
    }
    return send_json(s, 200, model_json(ctx.cfg));
}

static inline int slots_processing(ServerContext& ctx) {
    std::lock_guard<std::mutex> lk(ctx.state_mu);
    int n = 0;
    for (const auto& s : ctx.slots) if (s.req_id >= 0) ++n;
    return n;
}

static inline bool handle_health(socket_t s, ServerContext& ctx) {
    const int busy = slots_processing(ctx);
    const int total = (int)ctx.slots.size();
    mj::Value out = mj::Value::obj();
    out["status"] = "ok";
    out["model"] = ctx.cfg.model_id;
    out["n_ctx"] = ctx.cfg.n_ctx;
    out["n_ctx_per_slot"] = ctx.cfg.n_ctx_slot;
    out["slots_idle"] = total - busy;
    out["slots_processing"] = busy;
    return send_json(s, 200, out);
}

static inline bool handle_slots(socket_t s, ServerContext& ctx) {
    std::vector<SlotState> st;
    {
        std::lock_guard<std::mutex> lk(ctx.state_mu);
        st = ctx.slots;
    }
    mj::Value arr = mj::Value::arr();
    for (int i = 0; i < (int)st.size(); ++i) {
        const SlotState& s = st[(size_t)i];
        const bool processing = s.req_id >= 0;
        mj::Value slot = mj::Value::obj();
        slot["id"] = i;
        slot["id_task"] = s.req_id;
        slot["kind"] = s.kind;
        slot["state"] = processing ? s.phase : "idle";
        slot["is_processing"] = processing;
        slot["n_ctx"] = ctx.cfg.n_ctx_slot;
        slot["speculative"] = ctx.cfg.draft > 0;
        slot["n_past"] = s.prompt_tokens + s.generated;
        slot["n_prompt_tokens"] = s.prompt_tokens;
        slot["n_prompt_tokens_processed"] = s.prompt_processed;
        slot["n_prompt_tokens_cached"] = s.prompt_cached;
        slot["n_decoded"] = s.generated;
        slot["n_remain"] = std::max(0, s.max_tokens - s.generated);
        slot["prompt"] = s.prompt_preview;
        slot["model"] = s.model;
        slot["t_start"] = s.start_epoch;
        mj::Value next = mj::Value::obj();
        next["id"] = -1;
        next["token"] = "";
        next["logprob"] = 0.0;
        next["top_logprobs"] = mj::Value::arr();
        slot["next_token"] = std::move(next);
        arr.push(std::move(slot));
    }
    return send_json(s, 200, arr);
}

static inline bool handle_metrics(socket_t s, ServerContext& ctx) {
    const int busy = slots_processing(ctx);
    const int total = (int)ctx.slots.size();
    const double uptime = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - ctx.start_time)
                              .count();
    std::ostringstream out;
    auto counter = [&](const char* name, const char* help, long long value) {
        out << "# HELP " << name << " " << help << "\n";
        out << "# TYPE " << name << " counter\n";
        out << name << " " << value << "\n";
    };
    counter("bruce_requests_total", "HTTP generation requests received", ctx.requests_total.load());
    counter("bruce_requests_completed_total", "Generation requests completed", ctx.requests_completed.load());
    counter("bruce_requests_failed_total", "Generation requests failed or cancelled",
            ctx.requests_failed.load());
    counter("bruce_prompt_tokens_total", "Prompt tokens evaluated", ctx.prompt_tokens_total.load());
    counter("bruce_prompt_tokens_cached_total", "Prompt tokens served from prefix cache",
            ctx.cached_tokens_total.load());
    counter("bruce_completion_tokens_total", "Generated tokens", ctx.completion_tokens_total.load());
    out << "# HELP bruce_prefill_seconds_total Time spent in prompt evaluation\n";
    out << "# TYPE bruce_prefill_seconds_total counter\n";
    out << "bruce_prefill_seconds_total " << ctx.prefill_us_total.load() / 1.0e6 << "\n";
    out << "# HELP bruce_decode_seconds_total Time spent generating tokens\n";
    out << "# TYPE bruce_decode_seconds_total counter\n";
    out << "bruce_decode_seconds_total " << ctx.decode_us_total.load() / 1.0e6 << "\n";
    out << "# HELP bruce_slots_processing Engine slots currently processing\n";
    out << "# TYPE bruce_slots_processing gauge\n";
    out << "bruce_slots_processing " << busy << "\n";
    out << "# HELP bruce_slots_total Engine slots\n";
    out << "# TYPE bruce_slots_total gauge\n";
    out << "bruce_slots_total " << total << "\n";
    if (total > 1) {
        const long long ns = ctx.batcher.n_steps.load();
        const long long nr = ctx.batcher.n_rows.load();
        const double st = ns > 0 ? ctx.batcher.t_step_us.load() / 1000.0 / (double)ns : 0.0;
        const double sw = ns > 0 ? ctx.batcher.t_wait_us.load() / 1000.0 / (double)ns : 0.0;
        out << "# HELP bruce_batch_step_ms Wall ms per batched decode step (engine + sync)\n";
        out << "# TYPE bruce_batch_step_ms gauge\n";
        out << "bruce_batch_step_ms " << st << "\n";
        out << "# HELP bruce_batch_wait_ms Wall ms spent forming each decode batch\n";
        out << "# TYPE bruce_batch_wait_ms gauge\n";
        out << "bruce_batch_wait_ms " << sw << "\n";
        out << "# HELP bruce_batch_rows Average rows per decode step\n";
        out << "# TYPE bruce_batch_rows gauge\n";
        out << "bruce_batch_rows " << (ns > 0 ? (double)nr / (double)ns : 0.0) << "\n";
        out << "# HELP bruce_batch_steps_total Batched decode steps\n";
        out << "# TYPE bruce_batch_steps_total counter\n";
        out << "bruce_batch_steps_total " << ns << "\n";
    }
    out << "# HELP bruce_requests_processing HTTP requests currently open\n";
    out << "# TYPE bruce_requests_processing gauge\n";
    out << "bruce_requests_processing " << ctx.active.load() << "\n";
    out << "# HELP bruce_uptime_seconds Server uptime\n";
    out << "# TYPE bruce_uptime_seconds gauge\n";
    out << "bruce_uptime_seconds " << uptime << "\n";
    HttpResponse r;
    r.status = 200;
    r.content_type = "text/plain; version=0.0.4; charset=utf-8";
    r.body = out.str();
    return send_http_response(s, r);
}

static inline bool handle_root(socket_t s, ServerContext& ctx) {
    std::string html =
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Bruce server</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:820px;margin:40px auto;padding:0 20px;"
        "line-height:1.5}code,pre{background:#f3f3f3;padding:2px 5px;border-radius:4px}"
        "pre{padding:12px;overflow:auto}</style></head><body>"
        "<h1>Bruce local server</h1>"
        "<p>OpenAI-compatible endpoints:</p><ul>"
        "<li><code>POST /v1/chat/completions</code></li>"
        "<li><code>POST /v1/completions</code></li>"
        "<li><code>POST /completion</code> (llama.cpp-style)</li>"
        "<li><code>POST /tokenize</code>, <code>POST /detokenize</code></li>"
        "<li><code>GET /v1/models</code>, <code>GET /health</code>, <code>GET /props</code>, "
        "<code>GET /slots</code>, <code>GET /metrics</code></li>"
        "</ul><p>Model: <code>" + ctx.cfg.model_id + "</code>, context: " +
        std::to_string(ctx.cfg.n_ctx) + " tokens.</p>"
        "<pre>curl http://" + ctx.cfg.host + ":" + std::to_string(ctx.cfg.port) +
        "/v1/chat/completions \\\n  -H \"Content-Type: application/json\" \\\n"
        "  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":64}'</pre>"
        "</body></html>";
    HttpResponse r;
    r.status = 200;
    r.content_type = "text/html; charset=utf-8";
    r.body = std::move(html);
    return send_http_response(s, r);
}

// ------------------------------------------------------------------ routing / server loop

static inline bool route_request(socket_t s, ServerContext& ctx, const HttpRequest& req) {
    std::string path = path_only(req.path);
    if (req.method == "OPTIONS") {
        HttpResponse r;
        r.status = 204;
        r.body.clear();
        return send_http_response(s, r);
    }
    if (path == "/health" || path == "/healthz") return handle_health(s, ctx);
    if (path == "/v1/models" || path == "/models") return handle_models(s, ctx);
    if (path.rfind("/v1/models/", 0) == 0) return handle_model(s, ctx, path.substr(11));
    if (path == "/props") return handle_props(s, ctx);
    if (path == "/slots") return handle_slots(s, ctx);
    if (path == "/metrics") return handle_metrics(s, ctx);
    if (path == "/") return handle_root(s, ctx);

    if (req.method != "POST") {
        return send_json(s, 405, json_error("method not allowed"));
    }

    mj::Value body;
    std::string err;
    if (!parse_body_json(req, body, err)) return send_json(s, 400, json_error(err));

    if (path == "/tokenize") return handle_tokenize(s, ctx, body);
    if (path == "/detokenize") return handle_detokenize(s, ctx, body);
    if (path == "/v1/embeddings") {
        return send_json(s, 501, json_error("embeddings are not implemented", "not_implemented_error"));
    }

    if (path != "/v1/chat/completions" && path != "/v1/completions" && path != "/completion") {
        return send_json(s, 404, json_error("unknown endpoint"));
    }

    // Slots are acquired inside the handlers, so concurrent requests run in
    // parallel; with --slots 1 the single slot serialises them as before.
    if (path == "/v1/chat/completions") return handle_chat_completions(s, ctx, body);
    if (path == "/v1/completions") return handle_completions(s, ctx, body, false);
    return handle_completions(s, ctx, body, true);
}

struct ClientGuard {
    ServerContext* ctx;
    explicit ClientGuard(ServerContext* c) : ctx(c) { ctx->active.fetch_add(1); }
    ~ClientGuard() { ctx->active.fetch_sub(1); }
};

static inline void handle_client(socket_t client, std::shared_ptr<ServerContext> ctx) {
    ClientGuard guard(ctx.get());
    set_socket_timeout(client, 120);
#ifdef _WIN32
    int one = 1;
    setsockopt(client, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));
#else
    int one = 1;
    setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#endif
    HttpRequest req;
    bool ok = false;
    try {
        ok = read_http_request(client, req);
        if (!ok) {
            if (req.error_status > 0) send_json(client, req.error_status, json_error(req.error_message));
            close_socket(client);
            return;
        }
        const std::string path = path_only(req.path);
        const bool public_health = path == "/health" || path == "/healthz";
        const bool public_request = public_health || req.method == "OPTIONS";
        if (!public_request && !check_auth(req, ctx->cfg)) {
            send_unauthorized(client);
            close_socket(client);
            return;
        }
        route_request(client, *ctx, req);
    } catch (const std::exception& e) {
        log_line("[bruce] request error: %s", e.what());
        send_json(client, 500, json_error(e.what(), "server_error"));
    } catch (...) {
        log_line("[bruce] request error: unknown exception");
        send_json(client, 500, json_error("internal server error", "server_error"));
    }
    close_socket(client);
}

#ifdef _WIN32
static BOOL WINAPI bruce_ctrl_handler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT) {
        g_srv_stop.store(true);
        return TRUE;
    }
    return FALSE;
}
#else
static void bruce_signal_handler(int) { g_srv_stop.store(true); }
#endif

static inline int run_server(Tokenizer& tok, const ServerConfig& cfg) {
    g_srv_stop.store(false);
    auto ctx = std::make_shared<ServerContext>(tok, cfg);
    const int nslots = std::max(1, std::min(MAX_SLOTS, cfg.slots));
    ctx->sessions.resize((size_t)nslots);
    ctx->slots.resize((size_t)nslots);
    ctx->slot_busy.assign((size_t)nslots, 0);
    for (int i = 0; i < nslots; ++i) {
        ctx->sessions[(size_t)i] = std::make_unique<Session>(tok);
        ctx->sessions[(size_t)i]->init(i, nslots > 1);
        if (nslots > 1) ctx->sessions[(size_t)i]->batcher = &ctx->batcher;
    }
    if (nslots > 1) {
        // --n-batch is the prompt chunk cap; the burst/shared defaults sit under it.
        const int nb = std::max(1, std::min(MBUF, cfg.n_batch));
        ctx->batcher.chunk_solo = nb;
        ctx->batcher.chunk_burst = nb;
        ctx->batcher.chunk_shared = std::min(ctx->batcher.chunk_shared, nb);
        if (const char* e = getenv("BRUCE_SERVE_BATCH_WAIT")) ctx->batcher.batch_wait_us = atoi(e);
        if (const char* e = getenv("BRUCE_SERVE_CHUNK"))
            ctx->batcher.chunk_shared = std::max(1, std::min(MBUF, atoi(e)));
        if (const char* e = getenv("BRUCE_SERVE_CHUNK_SOLO")) {
            ctx->batcher.chunk_solo = std::max(1, std::min(MBUF, atoi(e)));
            ctx->batcher.chunk_burst = ctx->batcher.chunk_solo;
        }
        if (const char* e = getenv("BRUCE_SERVE_CHUNK_BURST"))
            ctx->batcher.chunk_burst = std::max(1, std::min(MBUF, atoi(e)));
    }

#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        log_line("[bruce] WSAStartup failed");
        return 1;
    }
#endif

    socket_t listen_sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_sock == k_invalid_socket) {
        log_line("[bruce] cannot create socket");
        return 1;
    }
    int one = 1;
#ifdef _WIN32
    setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof(one));
#else
    setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)cfg.port);
    std::string host = cfg.host;
    if (host == "localhost") host = "127.0.0.1";
    if (host.empty()) host = "127.0.0.1";
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        log_line("[bruce] invalid --host '%s' (use an IPv4 address or localhost)", host.c_str());
        close_socket(listen_sock);
        return 1;
    }
    if (bind(listen_sock, (sockaddr*)&addr, sizeof(addr)) != 0) {
        log_line("[bruce] bind failed on %s:%d", host.c_str(), cfg.port);
        close_socket(listen_sock);
        return 1;
    }
    if (listen(listen_sock, 16) != 0) {
        log_line("[bruce] listen failed");
        close_socket(listen_sock);
        return 1;
    }
    int actual_port = cfg.port;
    {
        sockaddr_in got;
        socklen_t glen = sizeof(got);
        if (getsockname(listen_sock, (sockaddr*)&got, &glen) == 0) actual_port = ntohs(got.sin_port);
    }

#ifdef _WIN32
    SetConsoleCtrlHandler(bruce_ctrl_handler, TRUE);
#else
    signal(SIGINT, bruce_signal_handler);
    signal(SIGTERM, bruce_signal_handler);
#endif

    log_line("[bruce] model: %s", cfg.model_name.empty() ? cfg.model_path.c_str() : cfg.model_name.c_str());
    log_line("[bruce] server: ctx %d (%d/slot) | batch %d | default n_predict %d | slots %d | draft %d | thinking %s",
             cfg.n_ctx, cfg.n_ctx_slot, cfg.n_batch, cfg.n_predict, nslots, cfg.draft,
             cfg.enable_thinking ? "on" : "off");
    if (nslots > 1)
        log_line("[bruce] continuous batching: up to %d concurrent sequences share one weight pass",
                 nslots);
    log_line("[bruce] OpenAI-compatible server listening on http://%s:%d", host.c_str(), actual_port);
    log_line("[bruce] endpoints: /v1/chat/completions /v1/completions /completion /tokenize /detokenize "
             "/v1/models /health /props /slots /metrics");
    if (!cfg.api_key.empty()) log_line("[bruce] API key required (Authorization: Bearer ...)");
    if (cfg.verbose) log_line("[bruce] verbose logging enabled");

    if (nslots > 1) ctx->batcher.start();

    while (!g_srv_stop.load() && !ctx->stop.load()) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(listen_sock, &fds);
        timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 500000;
#ifdef _WIN32
        int sel = select(0, &fds, nullptr, nullptr, &tv);
#else
        int sel = select(listen_sock + 1, &fds, nullptr, nullptr, &tv);
#endif
        if (sel <= 0) continue;
        sockaddr_in peer;
        socklen_t plen = sizeof(peer);
        socket_t client = accept(listen_sock, (sockaddr*)&peer, &plen);
        if (client == k_invalid_socket) continue;
        std::thread(handle_client, client, ctx).detach();
    }

    log_line("[bruce] shutting down");
    ctx->stop.store(true);
    close_socket(listen_sock);
    for (int i = 0; i < 100 && ctx->active.load() > 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (ctx->active.load() > 0) {
        log_line("[bruce] %d request(s) still active during shutdown", ctx->active.load());
    }
    if (nslots > 1) ctx->batcher.shutdown();
#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
