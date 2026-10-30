// Reusable generation session on top of the Bruce CUDA engine.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "http.h"

struct Session;

// ---------------------------------------------------------------------------
// Batch coordinator (multi-slot serving only).
//
// One CUDA-owning worker thread.  A request thread submits a decode token or a
// prompt chunk and blocks until the worker has run it.  Every decode job that is
// already queued when the worker picks up the next step is fused into a single
// forward_batch_ms pass, so N concurrent sequences share one weight stream.
// A prefill job is split into sub-chunks and queued decode jobs run between them,
// which keeps token latency bounded while a long prompt is being ingested.
// ---------------------------------------------------------------------------
struct BatchCoordinator {
    struct Job {
        enum Kind { DECODE, PREFILL, VERIFY } kind = DECODE;
        Session* s = nullptr;
        int slot = 0;
        std::vector<int> toks;
        int pos = 0;
        bool need_logits = true;
        bool want_argmax = false;   // session can use the batch argmax directly
        int* amax_out = nullptr;
        bool done = false;
    };

    std::mutex mu;
    std::condition_variable cv_work, cv_done;
    std::deque<std::shared_ptr<Job>> q;
    std::thread th;
    bool stop = false;
    int active = 0;              // sessions between enter() and leave()
    int decoders = 0;            // sessions currently inside generate()
    int max_rows = MAX_ROWS;
    int chunk_solo = MBUF;       // prompt rows per chunk when nothing else runs
    int chunk_shared = 384;      // prompt rows per chunk while other slots decode
    int chunk_burst = MBUF;      // prompt rows per chunk while several prompts are queued
    int batch_wait_us = 2000;    // grace period to fuse concurrent decode submits
    std::atomic<long long> t_step_us{0}, n_steps{0}, n_rows{0}, t_wait_us{0};

    void start() { th = std::thread([this] { run(); }); }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv_work.notify_all();
        if (th.joinable()) th.join();
    }

    void enter() { std::lock_guard<std::mutex> lk(mu); ++active; }
    void leave() { std::lock_guard<std::mutex> lk(mu); --active; }
    void enter_decode() { std::lock_guard<std::mutex> lk(mu); ++decoders; }
    void leave_decode() { std::lock_guard<std::mutex> lk(mu); --decoders; }
    bool solo() { std::lock_guard<std::mutex> lk(mu); return active <= 1; }

    void submit(const std::shared_ptr<Job>& j) {
        std::unique_lock<std::mutex> lk(mu);
        q.push_back(j);
        cv_work.notify_one();
        cv_done.wait(lk, [&] { return j->done; });
    }

    void decode(Session* s, int slot, int tok, int pos, bool want_argmax = false) {
        auto j = std::make_shared<Job>();
        j->kind = Job::DECODE;
        j->s = s;
        j->slot = slot;
        j->toks.assign(1, tok);
        j->pos = pos;
        j->want_argmax = want_argmax;
        submit(j);
    }

    void prefill(Session* s, int slot, const std::vector<int>& toks, int pos, bool need_logits) {
        auto j = std::make_shared<Job>();
        j->kind = Job::PREFILL;
        j->s = s;
        j->slot = slot;
        j->toks = toks;
        j->pos = pos;
        j->need_logits = need_logits;
        submit(j);
    }

    void verify(Session* s, int slot, const std::vector<int>& toks, int pos, int* amax_out) {
        auto j = std::make_shared<Job>();
        j->kind = Job::VERIFY;
        j->s = s;
        j->slot = slot;
        j->toks = toks;
        j->pos = pos;
        j->amax_out = amax_out;
        submit(j);
    }

    void finish(const std::shared_ptr<Job>& j) {
        {
            std::lock_guard<std::mutex> lk(mu);
            j->done = true;
        }
        cv_done.notify_all();
    }

    // Steal every queued decode job (up to max_rows).  Returns the batch.
    std::vector<std::shared_ptr<Job>> take_decodes() {
        std::vector<std::shared_ptr<Job>> out;
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = q.begin(); it != q.end() && (int)out.size() < max_rows;) {
            if ((*it)->kind == Job::DECODE) { out.push_back(*it); it = q.erase(it); }
            else ++it;
        }
        return out;
    }

    // Build a decode batch: everything already queued plus (up to batch_wait_us)
    // the submits still in flight from other generating sessions.  Without the
    // grace period each session would arrive just after the previous step ended
    // and every step would run as a 1-row pass.
    std::vector<std::shared_ptr<Job>> gather_more(std::vector<std::shared_ptr<Job>> batch) {
        const double tw0 = now_ms();
        std::unique_lock<std::mutex> lk(mu);
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::microseconds(std::max(0, batch_wait_us));
        for (;;) {
            for (auto it = q.begin(); it != q.end() && (int)batch.size() < max_rows;) {
                if ((*it)->kind == Job::DECODE) { batch.push_back(*it); it = q.erase(it); }
                else ++it;
            }
            if ((int)batch.size() >= max_rows) break;
            if ((int)batch.size() >= decoders) break;
            if (batch_wait_us <= 0 || stop) break;
            if (cv_work.wait_until(lk, deadline) == std::cv_status::timeout) break;
        }
        t_wait_us += (long long)((now_ms() - tw0) * 1000.0);
        return batch;
    }

    std::vector<std::shared_ptr<Job>> gather_decodes(std::shared_ptr<Job> first) {
        std::vector<std::shared_ptr<Job>> batch;
        batch.push_back(std::move(first));
        return gather_more(std::move(batch));
    }

    void run_decodes(std::vector<std::shared_ptr<Job>> batch);

    void run() {
        for (;;) {
            std::shared_ptr<Job> j;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv_work.wait(lk, [&] { return stop || !q.empty(); });
                if (q.empty()) {
                    if (stop) return;
                    continue;
                }
                j = q.front();
                q.pop_front();
            }
            if (j->kind == Job::DECODE) {
                run_decodes(gather_decodes(j));
            } else if (j->kind == Job::PREFILL) {
                run_prefill(j);
                finish(j);
            } else {
                bruce_verify_round(j->slot, j->toks.data(), (int)j->toks.size(), j->pos, j->amax_out);
                finish(j);
            }
        }
    }

    void run_prefill(const std::shared_ptr<Job>& j) {
        const int n = (int)j->toks.size();
        int done = 0;
        while (done < n) {
            int chunk;
            {
                std::lock_guard<std::mutex> lk(mu);
                // Several prompts queued at once (a burst of agents): the extra
                // latency of a wide chunk is paid by requests that are already
                // waiting for their own prompt, so drain the burst with the wide
                // GEMM.  A single prompt behind live decoders keeps the narrow
                // chunk: that is the interactive case the ~150 ms gap is for.
                const int pending = active - decoders;
                chunk = (decoders > 0 && pending <= 1) ? chunk_shared : chunk_burst;
                if (pending <= 0) chunk = chunk_solo;
            }
            chunk = std::max(1, std::min(chunk, n - done));
            const bool last = (done + chunk == n);
            bruce_prefill_chunk(j->slot, j->toks.data() + done, chunk, j->pos + done,
                               last && j->need_logits);
            CK(cudaStreamSynchronize(S));
            done += chunk;
            // Give waiting decode requests a batched step between prompt chunks.
            {
                auto dec = take_decodes();
                if (!dec.empty()) run_decodes(gather_more(std::move(dec)));
            }
        }
    }
};

struct TokenLogprob {
    int id = -1;
    std::string token;
    float logprob = 0.0f;
};

struct GenParams {
    int max_tokens = 256;
    bool greedy = false;
    float temp = 1.0f;
    float top_p = 0.95f;
    int top_k = 40;
    float min_p = 0.0f;
    float repeat_penalty = 1.0f;
    int repeat_last_n = 64;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    std::unordered_map<int, float> logit_bias;
    uint64_t seed = 0x1234567890ABCDEFull;
    std::vector<std::string> stop;
    std::vector<int> stop_token_ids;
    bool ignore_eos = false;
    int min_tokens = 0;
    int draft = 0;
    int draft_ngram = 4;
    int draft_nmin = -1;
    bool logprobs = false;
    int top_logprobs = 0;
    bool cache_prompt = true;
    int n_keep = 0;                       // accepted for llama.cpp compatibility
    std::function<bool()> cancelled;      // checked once per generated token / prefill chunk
};

struct PromptStats {
    int prompt_tokens = 0;
    int cached_tokens = 0;
    int processed_tokens = 0;
    double prefill_ms = 0.0;
    bool cancelled = false;
};

struct TokenPieceInfo {
    int id = -1;
    std::string text;
    bool has_logprob = false;
    float logprob = 0.0f;
    std::vector<TokenLogprob> top;
};

struct GenResult {
    std::string text;
    std::vector<TokenPieceInfo> pieces;
    std::string finish_reason = "stop";
    int completion_tokens = 0;
    bool cancelled = false;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    double prefill_ms = 0.0;
};

struct DeltaEvent {
    int token_id = -1;
    std::string text;
    bool has_logprob = false;
    float logprob = 0.0f;
    const std::vector<TokenLogprob>* top = nullptr;
};

using DeltaSink = std::function<bool(const DeltaEvent&)>;

struct Session {
    Tokenizer* tok = nullptr;
    std::vector<int> cache_ids;
    int slot = 0;                       // KV partition / sampler slot
    BatchCoordinator* batcher = nullptr;  // null: direct single-slot engine path
    float* d_logits = nullptr;          // this slot's logits row
    unsigned long long* d_rng = nullptr;  // this slot's sampler RNG state
    int amax[MBUF + 1];                 // verification round argmax (batched path)
    bool want_batch_argmax = false;     // greedy + no penalties/logprobs
    bool batch_tok_valid = false;       // batch worker already sampled this row
    int batch_tok = 0;

    int* d_hist = nullptr;
    int* d_bias_ids = nullptr;
    float* d_bias_vals = nullptr;
    int* d_amax = nullptr;

    std::vector<float> host_logits;
    std::vector<int> host_order;

    static constexpr int kMaxHist = 4096;
    static constexpr int kMaxBias = 64;

    explicit Session(Tokenizer& t) : tok(&t) {}

    void init(int slot_id = 0, bool batched = false) {
        slot = slot_id;
        batcher = nullptr;
        d_logits = batched ? M.slot_logits[slot_id] : M.logits;
        d_rng = batched ? M.slot_rng[slot_id] : M.d_rng;
        CK(cudaMalloc(&d_hist, sizeof(int) * kMaxHist));
        CK(cudaMalloc(&d_bias_ids, sizeof(int) * kMaxBias));
        CK(cudaMalloc(&d_bias_vals, sizeof(float) * kMaxBias));
        CK(cudaMalloc(&d_amax, sizeof(int) * (MBUF + 1)));
        host_logits.resize(VOCAB);
        host_order.resize(VOCAB);
        for (int i = 0; i < VOCAB; ++i) host_order[(size_t)i] = i;
    }

    void set_rng(uint64_t seed) {
        CK(cudaMemcpyAsync(d_rng, &seed, 8, cudaMemcpyHostToDevice, S));
    }

    static int common_prefix(const std::vector<int>& a, const std::vector<int>& b) {
        int n = (int)std::min(a.size(), b.size());
        int i = 0;
        while (i < n && a[(size_t)i] == b[(size_t)i]) ++i;
        return i;
    }

    int cache_hit(const std::vector<int>& prompt) const {
        return common_prefix(prompt, cache_ids);
    }

    void forward_one(int token, int pos, bool decode = false) {
        if (batcher) {
            batch_tok_valid = false;
            batcher->decode(this, slot, token, pos, want_batch_argmax);
            return;
        }
        set_tok_pos(token, pos);
        if (M.ex_fwd) {
            if (decode && g_adaptive_loops) launch_decode_graph(S);
            else CK(cudaGraphLaunch(M.ex_fwd, S));
        } else forward_launch(S);
    }

    void forward_batch_tokens(const std::vector<int>& toks, int pos, bool all_logits, bool need_logits) {
        if (batcher) { batcher->prefill(this, slot, toks, pos, need_logits); return; }
        CK(cudaMemcpyAsync(M.bm_tok, toks.data(), sizeof(int) * toks.size(), cudaMemcpyHostToDevice, S));
        g_batch_pos0 = pos;   // prefill shape dispatch (wide attention block at depth)
        CK(cudaMemcpyAsync(M.d_pos, &pos, 4, cudaMemcpyHostToDevice, S));
        forward_batch((int)toks.size(), all_logits, need_logits, S);
    }

    PromptStats prepare(const std::vector<int>& prompt, bool use_cache, int chunk,
                        const std::function<void(int, int, double)>& progress,
                        const std::function<bool()>& cancelled = {}) {
        PromptStats st;
        st.prompt_tokens = (int)prompt.size();
        const double t0 = now_ms();
        if (prompt.empty()) return st;

        int common = use_cache ? common_prefix(prompt, cache_ids) : 0;
        if (common > (int)prompt.size()) common = (int)prompt.size();
        cache_ids.resize((size_t)common);
        st.cached_tokens = common;

        int pos = common;
        if (common == (int)prompt.size()) {
            if (cancelled && cancelled()) {
                st.cancelled = true;
                st.prefill_ms = now_ms() - t0;
                return st;
            }
            // Logits for the final prompt token are not kept between requests.  Re-run that
            // one position; it overwrites its own KV slot with identical values.
            forward_one(prompt.back(), (int)prompt.size() - 1);
            CK(cudaStreamSynchronize(S));
            if (progress) progress((int)prompt.size(), (int)prompt.size(), now_ms() - t0);
        } else {
            chunk = std::max(1, std::min(MBUF, chunk));
            while (pos < (int)prompt.size()) {
                if (cancelled && cancelled()) {
                    st.cancelled = true;
                    st.prefill_ms = now_ms() - t0;
                    return st;
                }
                int n = std::min(chunk, (int)prompt.size() - pos);
                bool last = (pos + n == (int)prompt.size());
                if (n == 1) {
                    forward_one(prompt[(size_t)pos], pos);
                } else {
                    std::vector<int> part(prompt.begin() + pos, prompt.begin() + pos + n);
                    forward_batch_tokens(part, pos, false, last);
                }
                CK(cudaStreamSynchronize(S));
                pos += n;
                if (progress) progress(pos, (int)prompt.size(), now_ms() - t0);
            }
        }

        cache_ids = prompt;
        st.processed_tokens = (int)prompt.size() - common;
        st.prefill_ms = now_ms() - t0;

        if (!batcher) {
            int end_pos = (int)prompt.size();
            CK(cudaMemcpyAsync(M.d_pos, &end_pos, 4, cudaMemcpyHostToDevice, S));
            CK(cudaStreamSynchronize(S));
        }
        return st;
    }

    void push_history(std::vector<int>& hist, int token, int window) {
        if (window <= 0) return;
        hist.push_back(token);
        if ((int)hist.size() > window) hist.erase(hist.begin(), hist.begin() + ((int)hist.size() - window));
    }

    void compute_logprobs(int selected, float temp, int topn, float& selected_lp,
                          std::vector<TokenLogprob>& top_out) {
        selected_lp = 0.0f;
        top_out.clear();
        CK(cudaMemcpyAsync(host_logits.data(), d_logits, sizeof(float) * VOCAB,
                           cudaMemcpyDeviceToHost, S));
        CK(cudaStreamSynchronize(S));

        float t = temp > 0.0f ? temp : 1.0f;
        double mx = -1.0e30;
        for (int i = 0; i < VOCAB; ++i) {
            double v = (double)host_logits[(size_t)i] / (double)t;
            if (v > mx) mx = v;
        }
        double sum = 0.0;
        for (int i = 0; i < VOCAB; ++i) sum += std::exp((double)host_logits[(size_t)i] / (double)t - mx);
        double logz = mx + std::log(sum);

        int ntop = std::max(0, std::min(topn, 64));
        if (ntop > 0) {
            std::partial_sort(host_order.begin(), host_order.begin() + ntop, host_order.end(),
                              [&](int a, int b) {
                                  return (double)host_logits[(size_t)a] / (double)t >
                                         (double)host_logits[(size_t)b] / (double)t;
                              });
            top_out.reserve((size_t)ntop);
            for (int i = 0; i < ntop; ++i) {
                int id = host_order[(size_t)i];
                TokenLogprob lp;
                lp.id = id;
                lp.token = tok->decode_token(id, true);
                lp.logprob = (float)((double)host_logits[(size_t)id] / (double)t - logz);
                top_out.push_back(std::move(lp));
            }
        }
        selected_lp = (float)((double)host_logits[(size_t)selected] / (double)t - logz);
    }

    int sample_token(const GenParams& gp, const std::vector<int>& hist,
                     bool want_logprobs, float& logprob, std::vector<TokenLogprob>& top) {
        logprob = 0.0f;
        top.clear();

        bool need_pen = gp.repeat_penalty != 1.0f || gp.frequency_penalty != 0.0f ||
                        gp.presence_penalty != 0.0f;
        // Fast path: the batch worker already produced this row's argmax, and
        // nothing else would change the result.
        if (batch_tok_valid) {
            batch_tok_valid = false;
            if (gp.greedy && !need_pen && gp.logit_bias.empty() && !want_logprobs)
                return batch_tok;
        }
        int nhist = need_pen ? (int)std::min<size_t>(hist.size(), kMaxHist) : 0;
        int nbias = (int)std::min<size_t>(gp.logit_bias.size(), (size_t)kMaxBias);

        // The sampler scratch (probs / radix sort) is shared by all slots, so only
        // one slot samples at a time.  get_tok() synchronises the stream inside the
        // lock, so the next slot's kernels cannot overlap this one's scratch.
        std::lock_guard<std::mutex> sample_lk(g_sample_mu);
        if (nhist > 0 || nbias > 0) {
            if (nhist > 0) {
                CK(cudaMemcpyAsync(d_hist, hist.data() + (hist.size() - (size_t)nhist),
                                   sizeof(int) * nhist, cudaMemcpyHostToDevice, S));
            }
            if (nbias > 0) {
                std::vector<int> ids;
                std::vector<float> vals;
                ids.reserve((size_t)nbias);
                vals.reserve((size_t)nbias);
                for (const auto& kv : gp.logit_bias) {
                    if ((int)ids.size() >= nbias) break;
                    if (kv.first < 0 || kv.first >= VOCAB) continue;
                    ids.push_back(kv.first);
                    vals.push_back(kv.second);
                }
                nbias = (int)ids.size();
                if (nbias > 0) {
                    CK(cudaMemcpyAsync(d_bias_ids, ids.data(), sizeof(int) * nbias, cudaMemcpyHostToDevice, S));
                    CK(cudaMemcpyAsync(d_bias_vals, vals.data(), sizeof(float) * nbias, cudaMemcpyHostToDevice, S));
                }
            }
            k_apply_penalties<<<(VOCAB + 511) / 512, 512, 0, S>>>(
                d_logits, VOCAB, d_hist, nhist, gp.repeat_penalty,
                gp.frequency_penalty, gp.presence_penalty,
                d_bias_ids, d_bias_vals, nbias);
        }

        sample_tokens_ex(d_logits, d_rng, gp.greedy, gp.temp, gp.top_p, gp.top_k, gp.min_p);
        int t = get_tok();
        if (want_logprobs) compute_logprobs(t, gp.temp, gp.top_logprobs, logprob, top);
        return t;
    }

    GenResult generate(const GenParams& gp, const std::vector<int>& prompt, const DeltaSink& sink,
                       const std::function<void(int)>& progress = {}) {
        GenResult res;
        if (gp.max_tokens <= 0) {
            res.finish_reason = "length";
            return res;
        }

        struct DecGuard {
            BatchCoordinator* b = nullptr;
            ~DecGuard() { if (b) b->leave_decode(); }
        } dec_guard{batcher};
        if (batcher) batcher->enter_decode();

        const double t0 = now_ms();
        std::string finish = "stop";
        bool first_delta = true;
        auto stopped = [&]() -> bool { return gp.cancelled && gp.cancelled(); };

        int window = gp.repeat_last_n;
        if (window < 0) window = kMaxHist;
        window = std::min(window, kMaxHist);
        bool need_pen = gp.repeat_penalty != 1.0f || gp.frequency_penalty != 0.0f ||
                        gp.presence_penalty != 0.0f;
        std::vector<int> hist;
        if (need_pen && window > 0) {
            size_t start = prompt.size() > (size_t)window ? prompt.size() - (size_t)window : 0;
            hist.assign(prompt.begin() + (ptrdiff_t)start, prompt.end());
        }

        bool spec = gp.draft > 0 && gp.greedy && !need_pen && gp.logit_bias.empty() &&
                    !gp.logprobs && gp.top_logprobs == 0;
        NgramDrafter drafter;
        if (spec) {
            drafter.dmax = std::max(1, std::min(7, gp.draft));
            drafter.nmax = std::max(2, std::min(11, gp.draft_ngram));
            drafter.nmin = std::max(1, std::min(drafter.nmax, gp.draft_nmin < 0 ? 2 : gp.draft_nmin));
            for (int id : prompt) drafter.push(id);
        }

        StopFilter stops;
        stops.set(gp.stop);
        Utf8Carry utf8;
        int pos = (int)prompt.size();

        auto append_delta = [&](const std::string& text, int token_id, bool has_lp, float lp,
                                const std::vector<TokenLogprob>* top) -> bool {
            if (text.empty()) return true;
            if (first_delta) {
                res.ttft_ms = now_ms() - t0;
                first_delta = false;
            }
            res.text += text;
            if (sink && !sink(DeltaEvent{token_id, text, has_lp, lp, top})) {
                res.cancelled = true;
                finish = "stop";
                return false;
            }
            return true;
        };

        auto emit_text = [&](const std::string& raw, int token_id, bool has_lp, float lp,
                             const std::vector<TokenLogprob>* top) -> bool {
            std::string complete;
            utf8.feed(raw, complete);
            std::string safe;
            bool hit = stops.feed(complete, safe);
            if (!safe.empty()) {
                if (!append_delta(safe, token_id, has_lp, lp, top)) return false;
            }
            if (hit) { finish = "stop"; return false; }
            return true;
        };

        auto emit_token = [&](int token_id, bool has_lp, float lp,
                              const std::vector<TokenLogprob>* top) -> bool {
            if (token_id == tok->eos_id && !gp.ignore_eos) {
                if (res.completion_tokens < gp.min_tokens) return true;
                finish = "stop";
                return false;
            }
            for (int st : gp.stop_token_ids) {
                if (token_id == st && res.completion_tokens >= gp.min_tokens) {
                    finish = "stop";
                    return false;
                }
            }
            if (token_id == tok->eos_id && gp.ignore_eos) return true;

            std::string raw = tok->decode_token(token_id, false);
            TokenPieceInfo info;
            info.id = token_id;
            info.text = raw;
            info.has_logprob = has_lp;
            info.logprob = lp;
            if (top) info.top = *top;
            res.pieces.push_back(std::move(info));
            res.completion_tokens += 1;
            if (progress) progress(res.completion_tokens);
            return emit_text(raw, token_id, has_lp, lp, top);
        };

        float lp = 0.0f;
        std::vector<TokenLogprob> top;
        const bool want_lp = gp.logprobs || gp.top_logprobs > 0;
        want_batch_argmax = gp.greedy && !need_pen && gp.logit_bias.empty() && !want_lp;
        int t = sample_token(gp, hist, want_lp, lp, top);
        int spec_kmax = spec ? std::min(drafter.dmax, 4) : 0;   // adaptive verify depth

        // Fast path state: after an accepted verification round the next target token is
        // already known and must not be sampled again.
        while (true) {
            if (stopped()) {
                res.cancelled = true;
                finish = "stop";
                break;
            }
            if (res.completion_tokens >= gp.max_tokens) { finish = "length"; break; }
            if (!emit_token(t, want_lp, lp, &top)) break;
            if (res.completion_tokens >= gp.max_tokens) { finish = "length"; break; }

            if (spec && (!batcher || batcher->solo())) {
                drafter.push(t);
                std::vector<int> drafts;
                drafter.draft(drafts);
                int k = (int)drafts.size();
                k = std::min(k, spec_kmax);
                k = std::min(k, gp.max_tokens - res.completion_tokens - 1);
                if (k <= 0) {
                    forward_one(t, pos, true);
                    pos += 1;
                    cache_ids.push_back(t);
                    push_history(hist, t, window);
                    t = sample_token(gp, hist, want_lp, lp, top);
                    continue;
                }

                std::vector<int> rtok;
                rtok.reserve((size_t)k + 1);
                rtok.push_back(t);
                for (int i = 0; i < k; ++i) rtok.push_back(drafts[(size_t)i]);
                int* amax = this->amax;
                if (batcher) {
                    batcher->verify(this, slot, rtok, pos, amax);
                } else {
                    CK(cudaMemcpyAsync(M.bm_tok, rtok.data(), sizeof(int) * (size_t)(k + 1),
                                       cudaMemcpyHostToDevice, S));
                    CK(cudaMemcpyAsync(M.d_pos, &pos, 4, cudaMemcpyHostToDevice, S));
                    verify_launch(k + 1, S);
                    CK(cudaMemcpyAsync(amax, M.d_amax, sizeof(int) * (size_t)(k + 1), cudaMemcpyDeviceToHost, S));
                    CK(cudaStreamSynchronize(S));
                    verify_update();
                }

                int j = k;
                for (int i = 0; i < k; ++i) {
                    if (amax[i] != drafts[(size_t)i]) { j = i; break; }
                }
                if (j == k) spec_kmax = std::min(drafter.dmax, spec_kmax + 1);
                else if (j == 0) spec_kmax = std::max(1, spec_kmax - 1);
                cache_ids.push_back(t);
                for (int i = 0; i < j; ++i) cache_ids.push_back(drafts[(size_t)i]);
                pos += j + 1;
                push_history(hist, t, window);
                for (int i = 0; i < j; ++i) push_history(hist, drafts[(size_t)i], window);

                bool stop_now = false;
                for (int i = 0; i < j; ++i) {
                    if (!emit_token(drafts[(size_t)i], false, 0.0f, nullptr)) { stop_now = true; break; }
                    if (res.completion_tokens >= gp.max_tokens) { finish = "length"; stop_now = true; break; }
                }
                if (stop_now) break;
                t = amax[j];
                lp = 0.0f;
                top.clear();
                continue;
            }

            forward_one(t, pos, true);
            pos += 1;
            cache_ids.push_back(t);
            push_history(hist, t, window);
            t = sample_token(gp, hist, want_lp, lp, top);
        }

        std::string tail_complete;
        utf8.flush(tail_complete);
        std::string tail_safe;
        if (stops.feed(tail_complete, tail_safe)) finish = "stop";
        if (!tail_safe.empty()) {
            if (!append_delta(tail_safe, -1, false, 0.0f, nullptr)) res.cancelled = true;
        }
        std::string final_tail;
        stops.flush(final_tail);
        if (!final_tail.empty()) {
            if (!append_delta(final_tail, -1, false, 0.0f, nullptr)) res.cancelled = true;
        }

        res.finish_reason = finish;
        res.total_ms = now_ms() - t0;
        return res;
    }
};

// Defined after Session: the batch worker hands each row's argmax straight to the
// waiting request thread, so a greedy session never has to launch its own sampler.
inline void BatchCoordinator::run_decodes(std::vector<std::shared_ptr<Job>> batch) {
    if (batch.empty()) return;
    const double t0 = now_ms();
    int toks[MAX_ROWS], pos[MAX_ROWS], slots[MAX_ROWS];
    const int rows = (int)batch.size();
    bool any_argmax = false;
    for (int i = 0; i < rows; ++i) {
        toks[i] = batch[(size_t)i]->toks[0];
        pos[i] = batch[(size_t)i]->pos;
        slots[i] = batch[(size_t)i]->slot;
        any_argmax = any_argmax || batch[(size_t)i]->want_argmax;
    }
    bruce_decode_step(rows, toks, pos, slots);
    int host_amax[MAX_ROWS] = {0};
    if (any_argmax) {
        // rows >= 2 already ran k_argmax_rows inside the batch graph; a 1-row step
        // runs the M=1 graph, so take the argmax of its logits row here.
        if (rows == 1)
            k_argmax_rows<<<1, 1024, 0, S>>>(M.d_amax, M.slot_logits[slots[0]], VOCAB, 1);
        CK(cudaMemcpyAsync(host_amax, M.d_amax, sizeof(int) * rows, cudaMemcpyDeviceToHost, S));
    }
    CK(cudaStreamSynchronize(S));
    t_step_us += (long long)((now_ms() - t0) * 1000.0);
    n_steps += 1;
    n_rows += rows;
    for (int i = 0; i < rows; ++i) {
        if (batch[(size_t)i]->want_argmax) {
            batch[(size_t)i]->s->batch_tok = host_amax[i];
            batch[(size_t)i]->s->batch_tok_valid = true;
        }
        finish(batch[(size_t)i]);
    }
}
