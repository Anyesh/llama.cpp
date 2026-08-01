// greedy-decodes the same prompt with expert streaming off and on and requires
// bitwise-identical logits at every step: the same kernels run on the same
// bytes, only the addressing through the slot pools differs

#include "llama.h"

#include "get-model.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct run_state {
    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;
};

static run_state make_run(const char * model_path, bool stream, int32_t slots) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers     = 999;
    mparams.moe_stream       = stream;
    mparams.moe_stream_slots = slots;
    // repacked extra buffer types would give the baseline different kernels than
    // the plain-CPU tensors streaming requires; identity needs identical kernels
    mparams.use_extra_bufts  = false;

    run_state r;
    r.model = llama_model_load_from_file(model_path, mparams);
    if (!r.model) {
        return r;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;

    r.ctx = llama_init_from_model(r.model, cparams);
    return r;
}

int main(int argc, char * argv[]) {
    char * model_path = get_model_or_exit(argc, argv);

    const int n_decode = argc > 2 ? atoi(argv[2]) : 160;
    // fewer slots than experts so the identity test also exercises eviction
    const int n_slots  = argc > 3 ? atoi(argv[3]) : 4;

    run_state base   = make_run(model_path, false, n_slots);
    run_state stream = make_run(model_path, true,  n_slots);
    if (!base.model || !base.ctx || !stream.model || !stream.ctx) {
        fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(base.model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const char * prompt = "the quick brown fox jumps over the lazy dog";
    std::vector<llama_token> tokens(256);
    const int n_prompt = llama_tokenize(vocab, prompt, (int32_t) strlen(prompt), tokens.data(), (int32_t) tokens.size(), true, false);
    if (n_prompt < 1) {
        fprintf(stderr, "tokenization failed (%d)\n", n_prompt);
        return 1;
    }
    tokens.resize(n_prompt);

    llama_batch batch = llama_batch_init(512, 0, 1);

    auto decode = [&](llama_context * ctx, const std::vector<llama_token> & toks, int n_past) {
        batch.n_tokens = (int32_t) toks.size();
        for (size_t i = 0; i < toks.size(); i++) {
            batch.token[i]     = toks[i];
            batch.pos[i]       = n_past + (int32_t) i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = i == toks.size() - 1;
        }
        return llama_decode(ctx, batch);
    };

    if (decode(base.ctx, tokens, 0) != 0 || decode(stream.ctx, tokens, 0) != 0) {
        fprintf(stderr, "prompt decode failed\n");
        return 1;
    }

    int n_past = n_prompt;
    int n_checked = 0;

    for (int step = 0; step < n_decode; step++) {
        const float * logits_base   = llama_get_logits_ith(base.ctx,   -1);
        const float * logits_stream = llama_get_logits_ith(stream.ctx, -1);

        if (memcmp(logits_base, logits_stream, n_vocab * sizeof(float)) != 0) {
            int n_diff = 0;
            float max_diff = 0.0f;
            for (int i = 0; i < n_vocab; i++) {
                if (logits_base[i] != logits_stream[i]) {
                    n_diff++;
                    max_diff = std::fmax(max_diff, std::fabs(logits_base[i] - logits_stream[i]));
                }
            }
            fprintf(stderr, "FAIL: logits diverge at step %d: %d/%d values differ, max abs diff %g\n",
                    step, n_diff, n_vocab, max_diff);
            return 1;
        }
        n_checked++;

        llama_token best = 0;
        for (int i = 1; i < n_vocab; i++) {
            if (logits_base[i] > logits_base[best]) {
                best = i;
            }
        }

        const std::vector<llama_token> next = {best};
        if (decode(base.ctx, next, n_past) != 0 || decode(stream.ctx, next, n_past) != 0) {
            fprintf(stderr, "decode failed at step %d\n", step);
            return 1;
        }
        n_past++;
    }

    llama_batch_free(batch);
    llama_free(base.ctx);
    llama_free(stream.ctx);
    llama_model_free(base.model);
    llama_model_free(stream.model);

    printf("ok: %d steps bitwise-identical logits (%d slots)\n", n_checked, n_slots);
    return 0;
}
