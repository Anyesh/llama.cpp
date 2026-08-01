// writes a GGUF with 3D expert tensors, then verifies that the expert streamer
// loads slabs into pool slots byte-identical to the file contents

#include "../src/llama-expert-streamer.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void fill_deterministic(void * dst, size_t nbytes, uint32_t seed) {
    uint8_t * p = (uint8_t *) dst;
    uint32_t state = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < nbytes; i++) {
        state = state * 1664525u + 1013904223u;
        p[i] = (uint8_t) (state >> 24);
    }
}

int main() {
    const int64_t ne0       = 64;
    const int64_t ne1       = 32;
    const int32_t n_expert  = 8;
    const int32_t n_slots   = 4;
    const int32_t n_layers  = 3;

    const std::string fname = "test-expert-streamer.gguf";

    // build a gguf with one gate_up-like and one down-like 3D tensor per layer
    {
        ggml_init_params ip = { 256u*1024*1024, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        gguf_context * gguf = gguf_init_empty();
        gguf_set_val_str(gguf, "general.architecture", "test");

        for (int il = 0; il < n_layers; il++) {
            for (int k = 0; k < 2; k++) {
                ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k == 0 ? ne0 : ne1, k == 0 ? ne1 : ne0, n_expert);
                char name[128];
                snprintf(name, sizeof(name), "blk.%d.ffn_%s_exps.weight", il, k == 0 ? "gate_up" : "down");
                ggml_set_name(t, name);
                fill_deterministic(t->data, ggml_nbytes(t), (uint32_t) (il*2 + k + 1));
                gguf_add_tensor(gguf, t);
            }
        }

        if (!gguf_write_to_file(gguf, fname.c_str(), false)) {
            fprintf(stderr, "failed to write %s\n", fname.c_str());
            return 1;
        }
        gguf_free(gguf);
        ggml_free(ctx);
    }

    // read the whole file back as raw reference bytes
    std::vector<uint8_t> file_data;
    {
        FILE * f = fopen(fname.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "failed to reopen %s\n", fname.c_str());
            return 1;
        }
        fseek(f, 0, SEEK_END);
        file_data.resize((size_t) ftell(f));
        fseek(f, 0, SEEK_SET);
        if (fread(file_data.data(), 1, file_data.size(), f) != file_data.size()) {
            fprintf(stderr, "failed to read %s\n", fname.c_str());
            return 1;
        }
        fclose(f);
    }

    // reopen metadata to get per-tensor file offsets, mirroring what the model
    // loader snapshots at load time
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gp = { true, &meta_ctx };
    gguf_context * gguf = gguf_init_from_file(fname.c_str(), gp);
    if (!gguf) {
        fprintf(stderr, "failed to parse %s\n", fname.c_str());
        return 1;
    }
    const size_t data_offs = gguf_get_data_offset(gguf);

    // pools allocated in a CPU backend buffer, as the real integration does
    ggml_init_params pip = { ggml_tensor_overhead() * 64, nullptr, true };
    ggml_context * pool_ctx = ggml_init(pip);

    llama_expert_streamer streamer(n_slots, 2);
    streamer.set_files({fname});

    struct pool_ref {
        ggml_tensor * pool;
        size_t        offs;        // file offset of expert 0
        size_t        slab_nbytes;
    };
    std::vector<std::vector<pool_ref>> pools(n_layers);

    for (int il = 0; il < n_layers; il++) {
        for (int k = 0; k < 2; k++) {
            char name[128];
            snprintf(name, sizeof(name), "blk.%d.ffn_%s_exps.weight", il, k == 0 ? "gate_up" : "down");
            const int64_t ti = gguf_find_tensor(gguf, name);
            if (ti < 0) {
                fprintf(stderr, "tensor %s missing after round-trip\n", name);
                return 1;
            }
            ggml_tensor * meta = ggml_get_tensor(meta_ctx, name);
            ggml_tensor * pool = ggml_new_tensor_3d(pool_ctx, meta->type, meta->ne[0], meta->ne[1], n_slots);
            const size_t offs = data_offs + gguf_get_tensor_offset(gguf, ti);
            pools[il].push_back({pool, offs, meta->nb[2]});
        }
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(pool_ctx, ggml_backend_cpu_buffer_type());
    if (!buf) {
        fprintf(stderr, "pool allocation failed\n");
        return 1;
    }

    for (int il = 0; il < n_layers; il++) {
        for (const auto & p : pools[il]) {
            streamer.add_tensor(il, 0, p.offs, p.slab_nbytes, n_expert, p.pool);
        }
    }

    // several rounds per layer: cold fill, full hit, eviction churn, duplicates
    const std::vector<std::vector<int32_t>> rounds = {
        {0, 1, 2, 3},
        {0, 1, 2, 3},
        {4, 5, 0, 1},
        {6, 7, 6, 7},
        {2, 3, 4, 5},
    };

    int n_checked = 0;
    for (int il = 0; il < n_layers; il++) {
        for (const auto & ids : rounds) {
            std::vector<llama_expert_streamer::miss> misses;
            std::vector<int32_t> slot_ids(ids.size());
            streamer.plan(il, ids.data(), (int32_t) ids.size(), misses, slot_ids.data());
            streamer.execute(il, misses);

            for (size_t i = 0; i < ids.size(); i++) {
                if (slot_ids[i] < 0 || slot_ids[i] >= n_slots) {
                    fprintf(stderr, "layer %d: slot id %d out of range\n", il, slot_ids[i]);
                    return 1;
                }
                // duplicate ids must map to the same slot
                for (size_t j = 0; j < i; j++) {
                    if (ids[j] == ids[i] && slot_ids[j] != slot_ids[i]) {
                        fprintf(stderr, "layer %d: duplicate id %d mapped to slots %d and %d\n", il, ids[i], slot_ids[j], slot_ids[i]);
                        return 1;
                    }
                }
                for (const auto & p : pools[il]) {
                    const uint8_t * expect = file_data.data() + p.offs + (size_t) ids[i] * p.slab_nbytes;
                    const uint8_t * actual = (const uint8_t *) p.pool->data + (size_t) slot_ids[i] * p.slab_nbytes;
                    if (memcmp(expect, actual, p.slab_nbytes) != 0) {
                        fprintf(stderr, "layer %d expert %d slot %d: pool bytes differ from file\n", il, ids[i], slot_ids[i]);
                        return 1;
                    }
                    n_checked++;
                }
            }
        }
    }

    const auto stats = streamer.get_stats();
    if (stats.n_misses == 0 || stats.n_hits == 0 || stats.n_bytes == 0) {
        fprintf(stderr, "implausible stats: hits=%llu misses=%llu bytes=%llu\n",
                (unsigned long long) stats.n_hits, (unsigned long long) stats.n_misses, (unsigned long long) stats.n_bytes);
        return 1;
    }

    printf("ok: %d slab comparisons, hits=%llu misses=%llu bytes=%llu\n",
           n_checked, (unsigned long long) stats.n_hits, (unsigned long long) stats.n_misses, (unsigned long long) stats.n_bytes);

    ggml_backend_buffer_free(buf);
    ggml_free(pool_ctx);
    ggml_free(meta_ctx);
    gguf_free(gguf);
    remove(fname.c_str());
    return 0;
}
