// Dump per-layer activations from the llama.cpp reference build for a token sequence.
// Processes one token per llama_decode call, matching the engine's sequential prefill.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define N_EMBD   2560
#define N_VIRT   112
#define N_VOCAB  131072

struct dump_state {
    std::vector<float> x;      // [n_tokens][N_VIRT][N_EMBD]
    std::vector<float> logits; // [n_tokens][N_VOCAB]
    int n_tokens = 0;
    int cur = 0;
};

static bool cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    const bool want_x = strncmp(t->name, "l_out-", 6) == 0;
    const bool want_l = strcmp(t->name, "result_output") == 0;
    if (ask) {
        return want_x || want_l;
    }
    if (want_x) {
        const int il = atoi(t->name + 6);
        if (il < 0 || il >= N_VIRT || st->cur >= st->n_tokens) {
            fprintf(stderr, "bad l_out name %s\n", t->name);
            return true;
        }
        float * dst = st->x.data() + ((size_t)st->cur * N_VIRT + il) * N_EMBD;
        const size_t n = ggml_nelements(t);
        if (n != N_EMBD) {
            fprintf(stderr, "l_out-%d has %zu elems\n", il, n);
            return true;
        }
        if (ggml_backend_buffer_is_host(t->buffer)) {
            memcpy(dst, t->data, n * sizeof(float));
        } else {
            ggml_backend_tensor_get(t, dst, 0, n * sizeof(float));
        }
    }
    return true;
}

static const llama_model_kv_override kvs[] = {
    { LLAMA_KV_OVERRIDE_TYPE_INT, "dory.num_loops", { 2 } },
    { LLAMA_KV_OVERRIDE_TYPE_STR, "dory.recurrent_kv_cache_mode", { } },
    { LLAMA_KV_OVERRIDE_TYPE_INT, "", { 0 } },
};

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL OUT. BIN id1,id2,...\n", argv[0]);
        return 1;
    }
    std::string model_path = argv[1];
    std::string out_path   = argv[2];

    std::vector<int> ids;
    {
        const char * s = argv[3];
        while (*s) {
            ids.push_back(atoi(s));
            while (*s && *s != ',') s++;
            if (*s == ',') s++;
        }
    }

    llama_model_kv_override * ov = (llama_model_kv_override *) malloc(sizeof(kvs));
    memcpy(ov, kvs, sizeof(kvs));
    strcpy(ov[1].val_str, "last_loop");

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.kv_overrides = ov;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx        = (uint32_t)(ids.size() + 256);
    cp.n_batch      = (uint32_t)ids.size();
    cp.n_ubatch     = 1;
    cp.n_seq_max    = 1;
    cp.n_threads    = 6;
    cp.n_threads_batch = 6;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k       = GGML_TYPE_Q8_0;
    cp.type_v       = GGML_TYPE_Q8_0;
    cp.offload_kqv  = true;
    cp.no_perf      = false;
    cp.swa_full     = false;

    dump_state st;
    st.n_tokens = (int)ids.size();
    st.x.assign((size_t)st.n_tokens * N_VIRT * N_EMBD, 0.0f);
    st.logits.assign((size_t)st.n_tokens * N_VOCAB, 0.0f);

    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &st;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    llama_batch batch = llama_batch_init(1, 0, 1);
    for (size_t i = 0; i < ids.size(); ++i) {
        st.cur = (int)i;
        batch.n_tokens = 1;
        batch.token[0] = ids[i];
        batch.pos[0] = (llama_pos)i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at token %zu\n", i);
            return 1;
        }
        llama_synchronize(ctx);
        const float * lg = llama_get_logits_ith(ctx, 0);
        if (!lg) {
            fprintf(stderr, "no logits at token %zu\n", i);
            return 1;
        }
        memcpy(st.logits.data() + i * N_VOCAB, lg, N_VOCAB * sizeof(float));
    }

    FILE * f = fopen(out_path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", out_path.c_str());
        return 1;
    }
    const int32_t hdr[4] = { st.n_tokens, N_VIRT, N_EMBD, N_VOCAB };
    fwrite("DORYDUMP", 1, 8, f);
    fwrite(hdr, sizeof(int32_t), 4, f);
    fwrite(st.x.data(), sizeof(float), st.x.size(), f);
    fwrite(st.logits.data(), sizeof(float), st.logits.size(), f);
    fclose(f);

    fprintf(stderr, "dumped %d tokens to %s\n", st.n_tokens, out_path.c_str());

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    free(ov);
    return 0;
}
