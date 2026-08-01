// greedy-decodes the same prompt with expert streaming off and on and requires
// bitwise-identical logits at every step: the same kernels run on the same
// bytes, only the addressing through the slot pools differs.
// the two runs are sequential so peak memory stays at one model, which lets the
// test run on machines whose RAM holds only one copy of the dense weights

#include "llama.h"

#include "ggml-backend.h"

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

    // the baseline must keep experts on the CPU like --cpu-moe, or a GPU machine
    // whose VRAM cannot hold the full expert set fails to load it
    static llama_model_tensor_buft_override cpu_moe_overrides[] = {
        { "\\.ffn_(up|down|gate|gate_up)_(ch|)exps", nullptr },
        { nullptr,                                   nullptr },
    };
    if (!stream) {
        cpu_moe_overrides[0].buft = ggml_backend_cpu_buffer_type();
        mparams.tensor_buft_overrides = cpu_moe_overrides;
    }

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

static void free_run(run_state & r) {
    if (r.ctx) {
        llama_free(r.ctx);
        r.ctx = nullptr;
    }
    if (r.model) {
        llama_model_free(r.model);
        r.model = nullptr;
    }
}

static bool decode_tokens(llama_context * ctx, llama_batch & batch, const std::vector<llama_token> & toks, int n_past) {
    batch.n_tokens = (int32_t) toks.size();
    for (size_t i = 0; i < toks.size(); i++) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = n_past + (int32_t) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = i == toks.size() - 1;
    }
    return llama_decode(ctx, batch) == 0;
}

int main(int argc, char * argv[]) {
    char * model_path = get_model_or_exit(argc, argv);

    const int n_decode = argc > 2 ? atoi(argv[2]) : 160;
    // fewer slots than experts so the identity test also exercises eviction
    const int n_slots  = argc > 3 ? atoi(argv[3]) : 4;

    const char * prompt = "the quick brown fox jumps over the lazy dog";

    std::vector<llama_token> prompt_tokens(256);
    int n_vocab = 0;

    std::vector<std::vector<float>> ref_logits;
    std::vector<llama_token>        ref_tokens;

    // pass 1: baseline, record logits and the greedy continuation
    {
        run_state base = make_run(model_path, false, n_slots);
        if (!base.model || !base.ctx) {
            fprintf(stderr, "failed to load %s (baseline)\n", model_path);
            return 1;
        }

        const llama_vocab * vocab = llama_model_get_vocab(base.model);
        n_vocab = llama_vocab_n_tokens(vocab);

        const int n_prompt = llama_tokenize(vocab, prompt, (int32_t) strlen(prompt),
                prompt_tokens.data(), (int32_t) prompt_tokens.size(), true, false);
        if (n_prompt < 1) {
            fprintf(stderr, "tokenization failed (%d)\n", n_prompt);
            return 1;
        }
        prompt_tokens.resize(n_prompt);

        llama_batch batch = llama_batch_init(512, 0, 1);
        if (!decode_tokens(base.ctx, batch, prompt_tokens, 0)) {
            fprintf(stderr, "baseline prompt decode failed\n");
            return 1;
        }

        int n_past = n_prompt;
        for (int step = 0; step < n_decode; step++) {
            const float * logits = llama_get_logits_ith(base.ctx, -1);
            ref_logits.emplace_back(logits, logits + n_vocab);

            llama_token best = 0;
            for (int i = 1; i < n_vocab; i++) {
                if (logits[i] > logits[best]) {
                    best = i;
                }
            }
            ref_tokens.push_back(best);

            if (!decode_tokens(base.ctx, batch, {best}, n_past)) {
                fprintf(stderr, "baseline decode failed at step %d\n", step);
                return 1;
            }
            n_past++;
        }
        llama_batch_free(batch);
        free_run(base);
    }

    // pass 2: streamed, replay the recorded continuation and compare per step
    {
        run_state stream = make_run(model_path, true, n_slots);
        if (!stream.model || !stream.ctx) {
            fprintf(stderr, "failed to load %s (streamed)\n", model_path);
            return 1;
        }

        llama_batch batch = llama_batch_init(512, 0, 1);
        if (!decode_tokens(stream.ctx, batch, prompt_tokens, 0)) {
            fprintf(stderr, "streamed prompt decode failed\n");
            return 1;
        }

        int n_past = (int) prompt_tokens.size();
        for (int step = 0; step < n_decode; step++) {
            const float * logits = llama_get_logits_ith(stream.ctx, -1);

            if (memcmp(ref_logits[step].data(), logits, n_vocab * sizeof(float)) != 0) {
                int n_diff = 0;
                float max_diff = 0.0f;
                for (int i = 0; i < n_vocab; i++) {
                    if (ref_logits[step][i] != logits[i]) {
                        n_diff++;
                        max_diff = std::fmax(max_diff, std::fabs(ref_logits[step][i] - logits[i]));
                    }
                }
                fprintf(stderr, "FAIL: logits diverge at step %d: %d/%d values differ, max abs diff %g\n",
                        step, n_diff, n_vocab, max_diff);
                return 1;
            }

            if (!decode_tokens(stream.ctx, batch, {ref_tokens[step]}, n_past)) {
                fprintf(stderr, "streamed decode failed at step %d\n", step);
                return 1;
            }
            n_past++;
        }
        llama_batch_free(batch);
        free_run(stream);
    }

    printf("ok: %d steps bitwise-identical logits (%d slots)\n", n_decode, n_slots);
    return 0;
}
