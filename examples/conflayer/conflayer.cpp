// Standalone ConfLayers self-speculative decoding example.
//
// Runs the llama_conflayer module (src/llama-conflayer.cpp) against any GGUF
// model: greedy draft along a skip graph, bouquet-tree verification with the
// full model, greedy acceptance, optional dynamic_optimization layer-set search.
//
// With --baseline, also runs plain greedy decoding on a second context and prints
// both texts so the two can be compared for output equivalence.

#include "llama.h"
#include "llama-conflayer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr,
        "usage: %s -m model.gguf [options]\n"
        "\n"
        "options:\n"
        "  -h, --help               show this message\n"
        "  -m, --model FILE         model path (required)\n"
        "  -p, --prompt PROMPT      prompt (default: \"Hello my name is\")\n"
        "  -n, --n-predict N        max new tokens (default: 128)\n"
        "      --skip-ratio F       draft skip ratio (default: 0.6)\n"
        "      --optimize           run the dynamic_optimization layer-set search\n"
        "      --topk N             draft top-k (default: 5)\n"
        "      --max-step-draft N   max draft steps per round (default: 25)\n"
        "      --max-tree-seq N     seq/output budget of the verification tree (default: 32)\n"
        "      --stop-threshold F   draft stop confidence (default: 0.8)\n"
        "      --search-interval N  rounds between optimizations (default: 25)\n"
        "      --context-window N   validation window size (default: 32)\n"
        "      --max-opt-iter N     max optimization iterations (default: 1000)\n"
        "      --max-score F        stop optimizing at this acceptance score (default: 0.95)\n"
        "      --model-id ID        architecture id, used by dynamic_optimization (default: qwen3)\n"
        "      --layer-chunks N     flash-DRAM offload: layers streamed per step; the draft's\n"
        "                           non-skip layers stay resident (shared with verify), only the\n"
        "                           skip layers stream from flash (0=off)\n"
        "      --cluster            enable cluster-based KV cache (bookkeeping runs; sparse\n"
        "                           cluster attention stays off, full KV view kept)\n"
        "      --cluster-sparse     cluster with REAL sparse attention in the verify phase\n"
        "                           (top-k cluster K/V only); draft phase still keeps full KV\n"
        "  -s, --seed N             rng seed (default: 2024)\n"
        "  -ngl N                   GPU layers (default: 0)\n"
        "  -v, --verbose            verbose per-round output\n"
        "      --baseline           also run plain greedy decoding for comparison\n"
        "      --greedy-only        run only plain greedy decoding and exit\n"
        "      --selftest           run the tree-vs-sequential logits selftest and exit\n",
        prog);
}

static std::string piece(const llama_vocab * vocab, llama_token id, bool special) {
    char buf[128];
    const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, special);
    if (n < 0) {
        return std::string();
    }
    return std::string(buf, n);
}

static void print_tokens(const llama_vocab * vocab, const std::vector<llama_token> & toks, bool special) {
    for (llama_token t : toks) {
        printf("%s", piece(vocab, t, special).c_str());
    }
    printf("\n");
}

// plain greedy decoding on its own context, for comparison with the ConfLayers output
static std::vector<llama_token> run_baseline(llama_model * model,
                                             const llama_vocab * vocab,
                                             const std::vector<llama_token> & prompt,
                                             int n_predict) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = (uint32_t) (prompt.size() + n_predict + 32);
    cparams.n_batch = (uint32_t) prompt.size();

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        fprintf(stderr, "baseline: failed to create context\n");
        return {};
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);

    // llama_batch_get_one needs a mutable token array
    std::vector<llama_token> prompt_work(prompt);

    std::vector<llama_token> out;
    llama_batch batch = llama_batch_get_one(prompt_work.data(), (int32_t) prompt_work.size());
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "baseline: prompt decode failed\n");
        llama_free(ctx);
        return {};
    }

    const int t0 = (int) prompt.size();
    for (int pos = t0; pos < t0 + n_predict; ) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        llama_token id = 0;
        float best = -INFINITY;
        for (int i = 0; i < n_vocab; i++) {
            if (logits[i] > best) {
                best = logits[i];
                id   = (llama_token) i;
            }
        }
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }
        out.push_back(id);
        batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "baseline: decode failed\n");
            break;
        }
        pos += 1;
    }

    llama_free(ctx);
    return out;
}

// Reproduce a single verification-tree round in isolation and compare the tree's
// node logits against a sequential single-token decode. Exercises the multi-seq
// tree mask path of kv_unified with the same shape as the ConfLayers round loop.
//
// The history, root, and candidate tokens come from greedy-decoding the prompt,
// so the selftest uses the same real token ids the ConfLayers round sees
// (space-prefixed " capital", " of", ...), not re-tokenized substrings.
static int run_selftest(llama_model * model, const llama_vocab * vocab) {
    const int n_vocab = llama_vocab_n_tokens(vocab);

    auto argmax_ = [&](const float * logits) {
        int best = 0;
        float mx = logits[0];
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > mx) { mx = logits[i]; best = i; }
        }
        return (llama_token) best;
    };

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx         = 4096;
    cp.n_batch       = 128;
    cp.n_ubatch      = 128;
    cp.n_seq_max     = 256;
    cp.n_outputs_max = 256;
    cp.kv_unified    = true;

    // ---- context A: greedy decode the prompt to get the real generated tokens ----
    llama_context * cta = llama_init_from_model(model, cp);
    if (!cta) {
        fprintf(stderr, "selftest: ctx init failed\n");
        return 1;
    }
    std::vector<llama_token> prompt;
    {
        const char * s = "The capital of France is";
        const int n = -llama_tokenize(vocab, s, strlen(s), NULL, 0, true, true);
        prompt.resize(n);
        llama_tokenize(vocab, s, strlen(s), prompt.data(), prompt.size(), true, true);
    }
    auto decode1 = [&](llama_context * ctx, llama_token t, llama_pos pos) {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.n_tokens = 1;
        b.token[0] = t;
        b.pos[0]   = pos;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0] = true;
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        return rc;
    };

    llama_batch bp = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
    if (llama_decode(cta, bp)) { fprintf(stderr, "selftest: prompt decode failed\n"); return 1; }
    // greedy 5 more: positions 5..9 -> Paris . The capital of
    std::vector<llama_token> gen(5);
    for (int i = 0; i < 5; i++) {
        gen[i] = argmax_(llama_get_logits_ith(cta, -1));
        if (decode1(cta, gen[i], (llama_pos) (prompt.size() + i))) {
            fprintf(stderr, "selftest: gen decode failed\n");
            return 1;
        }
    }
    // sequential reference: the argmax the model gives at the "of"@9 state
    const llama_token seq_argmax = argmax_(llama_get_logits_ith(cta, -1));
    char sb[128];
    llama_token_to_piece(vocab, seq_argmax, sb, sizeof(sb), 0, true);
    fprintf(stderr, "selftest: seq tokens: %d[%s] %d[%s] %d[%s] %d[%s] %d[%s]\n",
            gen[0], piece(vocab, gen[0], true).c_str(), gen[1], piece(vocab, gen[1], true).c_str(),
            gen[2], piece(vocab, gen[2], true).c_str(), gen[3], piece(vocab, gen[3], true).c_str(),
            gen[4], piece(vocab, gen[4], true).c_str());
    fprintf(stderr, "selftest: sequential argmax=%d(%s)\n", seq_argmax, sb);
    llama_free(cta);

    // ---- tree variants: history = prompt + gen[0..2] (Paris . The), positions 0..7;
    //      root " capital"@8 (gen[3]), candidate " of"@9 (gen[4]).
    //      Compare each variant's "of" node argmax against the sequential 32961(Belgium).
    auto run_tree_variant = [&](const char * label, int of_node, int n_leaves, bool multi_root_seqs) {
        llama_context * ctb = llama_init_from_model(model, cp);
        if (!ctb) {
            fprintf(stderr, "selftest: ctx init failed\n");
            return;
        }
        std::vector<llama_token> hist(prompt);
        hist.insert(hist.end(), gen.begin(), gen.begin() + 3);
        llama_batch bh = llama_batch_get_one(hist.data(), (int32_t) hist.size());
        if (llama_decode(ctb, bh)) { fprintf(stderr, "selftest: hist decode failed\n"); return; }

        llama_memory_t mem = llama_get_memory(ctb);
        for (int s = 1; s <= n_leaves; s++) {
            llama_memory_seq_cp(mem, 0, (llama_seq_id) s, 0, (llama_pos) hist.size());
        }

        const int N = 1 + n_leaves;
        llama_batch tb = llama_batch_init(N, 0, (int32_t) n_leaves + 1);
        tb.n_tokens = N;
        for (int i = 0; i < N; i++) {
            if (i == 0) {
                tb.token[i]   = gen[3];
                tb.pos[i]     = (llama_pos) hist.size();
                tb.n_seq_id[i] = multi_root_seqs ? n_leaves : 1;
                for (int s = 0; s < (multi_root_seqs ? n_leaves : 1); s++) tb.seq_id[i][s] = (llama_seq_id) (s + 1);
            } else {
                tb.token[i]   = (i == of_node) ? gen[4] : prompt[(i - 1) % (int) prompt.size()];
                tb.pos[i]     = (llama_pos) hist.size() + 1;
                tb.n_seq_id[i] = 1;
                tb.seq_id[i][0] = (llama_seq_id) i;
            }
            tb.logits[i] = true;
        }
        if (llama_decode(ctb, tb)) { fprintf(stderr, "selftest: tree decode failed\n"); return; }
        llama_batch_free(tb);

        const float * tree_logits = llama_get_logits_ith(ctb, of_node);
        const llama_token tree_argmax = argmax_(tree_logits);
        char t1[128], t2[128];
        llama_token_to_piece(vocab, gen[3], t1, sizeof(t1), 0, true);
        llama_token_to_piece(vocab, gen[4], t2, sizeof(t2), 0, true);
        fprintf(stderr, "selftest: [%s] root@8=%d(%s) n_leaves=%d of@%d, of-node argmax=%d(%s)  %s\n",
                label, gen[3], t1, n_leaves, of_node, tree_argmax, t2,
                seq_argmax == tree_argmax ? "MATCH" : "MISMATCH");
        llama_free(ctb);
    };

    run_tree_variant("no-sibling  single-root-seq", 1, 1, false);
    run_tree_variant("no-sibling  multi-root-seqs ", 1, 1, true);
    run_tree_variant("full-tree   of@node3        ", 3, 10, true);

    // E1: seq0 decode of " of"@8 WITHOUT the root (" capital") before it.
    // If the tree "of" node behaves as if it never saw the root, its argmax == this.
    {
        llama_context * c = llama_init_from_model(model, cp);
        std::vector<llama_token> h(prompt);
        h.insert(h.end(), gen.begin(), gen.begin() + 3); // 0..7
        llama_batch bh = llama_batch_get_one(h.data(), (int32_t) h.size());
        llama_decode(c, bh);
        if (decode1(c, gen[4], (llama_pos) h.size())) { return 1; } // " of"@8, no root
        const llama_token no_root = argmax_(llama_get_logits_ith(c, -1));
        char nr[128];
        llama_token_to_piece(vocab, no_root, nr, sizeof(nr), 0, true);
        fprintf(stderr, "selftest: [E1 no-root ] seq0 \" of\"@8 without root, argmax=%d(%s)\n", no_root, nr);
        llama_free(c);
    }

    // E2/E3: one decode batch [capital@8, of@9] both seq 0, no seq_cp, no tree.
    // E2 = kv_unified, E3 = classic cache. Isolates a multi-token batch in one decode.
    for (int variant = 0; variant < 2; variant++) {
        llama_context_params cp2 = cp;
        cp2.kv_unified = (variant == 0);
        llama_context * c = llama_init_from_model(model, cp2);
        std::vector<llama_token> h(prompt);
        h.insert(h.end(), gen.begin(), gen.begin() + 3); // 0..7
        llama_batch bh = llama_batch_get_one(h.data(), (int32_t) h.size());
        llama_decode(c, bh);
        llama_batch tb = llama_batch_init(2, 0, 1);
        tb.n_tokens = 2;
        tb.token[0] = gen[3]; tb.pos[0] = (llama_pos) h.size();       tb.n_seq_id[0] = 1; tb.seq_id[0][0] = 0; tb.logits[0] = true;
        tb.token[1] = gen[4]; tb.pos[1] = (llama_pos) h.size() + 1;   tb.n_seq_id[1] = 1; tb.seq_id[1][0] = 0; tb.logits[1] = true;
        if (llama_decode(c, tb)) { fprintf(stderr, "selftest: E2/E3 decode failed\n"); return 1; }
        llama_batch_free(tb);
        const llama_token b2 = argmax_(llama_get_logits_ith(c, 1));
        char b2s[128];
        llama_token_to_piece(vocab, b2, b2s, sizeof(b2s), 0, true);
        fprintf(stderr, "selftest: [E%d batch2  ] %s seq0 [capital@8, of@9] one decode, of-node argmax=%d(%s)  %s\n",
                variant == 0 ? 2 : 3, variant == 0 ? "kv_unified" : "classic   ", b2, b2s,
                seq_argmax == b2 ? "MATCH" : "MISMATCH");
        llama_free(c);
    }

    // E4: single full batch [history + capital@8 + of@9] (all 10 tokens one decode).
    // If this gives Belgium, multi-token batches are fine and only SPLIT batches break.
    {
        llama_context * c = llama_init_from_model(model, cp);
        std::vector<llama_token> h(prompt);
        h.insert(h.end(), gen.begin(), gen.begin() + 5); // 0..9: ... The capital of
        llama_batch tb = llama_batch_init((int32_t) h.size(), 0, 1);
        tb.n_tokens = (int32_t) h.size();
        for (int i = 0; i < (int32_t) h.size(); i++) {
            tb.token[i] = h[i]; tb.pos[i] = (llama_pos) i;
            tb.n_seq_id[i] = 1; tb.seq_id[i][0] = 0; tb.logits[i] = true;
        }
        if (llama_decode(c, tb)) { fprintf(stderr, "selftest: E4 decode failed\n"); return 1; }
        llama_batch_free(tb);
        const llama_token b4 = argmax_(llama_get_logits_ith(c, (int) h.size() - 1));
        char b4s[128];
        llama_token_to_piece(vocab, b4, b4s, sizeof(b4s), 0, true);
        fprintf(stderr, "selftest: [E4 full    ] one decode 0..9, of@9 argmax=%d(%s)  %s\n",
                b4, b4s, seq_argmax == b4 ? "MATCH" : "MISMATCH");
        llama_free(c);
    }

    // E5: batch1 [0..8] (history + capital@8), batch2 ["of"@9] single token.
    // If this gives Belgium, the root decoded at the end of a batch is fine.
    {
        llama_context * c = llama_init_from_model(model, cp);
        std::vector<llama_token> h(prompt);
        h.insert(h.end(), gen.begin(), gen.begin() + 4); // 0..8: ... The capital
        llama_batch bh = llama_batch_get_one(h.data(), (int32_t) h.size());
        llama_decode(c, bh);
        const llama_token cap8 = argmax_(llama_get_logits_ith(c, -1)); // argmax of "capital"@8
        char cs[128];
        llama_token_to_piece(vocab, cap8, cs, sizeof(cs), 0, true);
        fprintf(stderr, "selftest: [E5 batch1  ] \" capital\"@8 (in 9-token batch) argmax=%d(%s)  %s\n",
                cap8, cs, cap8 == gen[4] ? "OK(=of)" : "BAD");
        if (decode1(c, gen[4], (llama_pos) h.size())) { return 1; } // " of"@9
        const llama_token b5 = argmax_(llama_get_logits_ith(c, -1));
        char b5s[128];
        llama_token_to_piece(vocab, b5, b5s, sizeof(b5s), 0, true);
        fprintf(stderr, "selftest: [E5 split2  ] [0..8] then \" of\"@9, argmax=%d(%s)  %s\n",
                b5, b5s, seq_argmax == b5 ? "MATCH" : "MISMATCH");
        llama_free(c);
    }

    // E5b: same as E5 but the whole 0..9 history was decoded as one batch, then read the
    // last-token argmax -- i.e. standard prompt processing of the full 10-token prefix.
    {
        llama_context * c = llama_init_from_model(model, cp);
        std::vector<llama_token> h(prompt);
        h.insert(h.end(), gen.begin(), gen.begin() + 5); // 0..9: ... The capital of
        llama_batch bh = llama_batch_get_one(h.data(), (int32_t) h.size());
        llama_decode(c, bh);
        const llama_token last = argmax_(llama_get_logits_ith(c, -1)); // argmax of "of"@9
        char ls[128];
        llama_token_to_piece(vocab, last, ls, sizeof(ls), 0, true);
        fprintf(stderr, "selftest: [E5b prompt ] one decode 0..9, last(\" of\"@9) argmax=%d(%s)  %s\n",
                last, ls, seq_argmax == last ? "MATCH" : "MISMATCH");
        llama_free(c);
    }

    // E8: dump per-token argmax from the full-batch path (E4), compared against the
    // greedy continuation tokens (gen). If the batch path diverges early, hidden states
    // are wrong from that point; if only the last token differs, it's the batch's own
    // last-token attention that's inconsistent.
    {
        llama_context * c = llama_init_from_model(model, cp);
        std::vector<llama_token> h(prompt);
        h.insert(h.end(), gen.begin(), gen.begin() + 5); // 0..9
        llama_batch fb = llama_batch_init((int32_t) h.size(), 0, 1);
        fb.n_tokens = (int32_t) h.size();
        for (int i = 0; i < (int32_t) h.size(); i++) {
            fb.token[i] = h[i]; fb.pos[i] = (llama_pos) i;
            fb.n_seq_id[i] = 1; fb.seq_id[i][0] = 0; fb.logits[i] = true;
        }
        llama_decode(c, fb);
        llama_batch_free(fb);
        std::string line = "selftest: [E8 batch  ] per-token argmax: ";
        for (int i = 0; i < (int32_t) h.size(); i++) {
            const llama_token a = argmax_(llama_get_logits_ith(c, i));
            char pp[96]; llama_token_to_piece(vocab, a, pp, sizeof(pp), 0, true);
            char seg[96]; snprintf(seg, sizeof(seg), "%d:%s ", i, pp);
            line += seg;
        }
        fprintf(stderr, "%s\n", line.c_str());
        line = "selftest: [E8 single ] greedy continuation:   ";
        for (int i = 0; i < 5; i++) {
            char pp[96]; llama_token_to_piece(vocab, gen[i], pp, sizeof(pp), 0, true);
            char seg[96]; snprintf(seg, sizeof(seg), "%d:%s ", prompt.size() + i, pp);
            line += seg;
        }
        char sp[96]; llama_token_to_piece(vocab, seq_argmax, sp, sizeof(sp), 0, true);
        char seg[96]; snprintf(seg, sizeof(seg), "%d:%s ", (int) prompt.size() + 5, sp);
        line += seg;
        fprintf(stderr, "%s\n", line.c_str());
        llama_free(c);
    }

    // E7: dump top-5 softmax of " of"@9 in both the single-token path (context A) and
    // the full-batch path (E4), to see whether the divergence is a near-tie precision issue.
    auto softmax_top5 = [&](const float * logits, llama_token * tk, float * pr) {
        std::vector<std::pair<float, llama_token>> top;
        for (int i = 0; i < n_vocab; i++) top.emplace_back(logits[i], (llama_token) i);
        std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                          [](const auto & a, const auto & b) { return a.first > b.first; });
        double mx = -1e30;
        for (const auto & e : top) mx = std::max(mx, (double) e.first);
        double z = 0;
        for (int i = 0; i < 5; i++) z += std::exp((double) top[i].first - mx);
        for (int i = 0; i < 5; i++) { tk[i] = top[i].second; pr[i] = (float) (std::exp((double) top[i].first - mx) / z); }
    };
    {
        // single-token path: re-run context A (fresh ctx), capture "of"@9 top-5
        llama_context * c = llama_init_from_model(model, cp);
        llama_batch bh = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
        llama_decode(c, bh);
        for (int i = 0; i < 5; i++) decode1(c, gen[i], (llama_pos) (prompt.size() + i));
        const float * l = llama_get_logits_ith(c, -1);
        llama_token tk[5]; float pr[5];
        softmax_top5(l, tk, pr);
        char line[512] = "";
        for (int i = 0; i < 5; i++) {
            char pp[96]; llama_token_to_piece(vocab, tk[i], pp, sizeof(pp), 0, true);
            char seg[128]; snprintf(seg, sizeof(seg), "%.3f %s | ", pr[i], pp);
            strncat(line, seg, sizeof(line) - strlen(line) - 1);
        }
        fprintf(stderr, "selftest: [E7 single ] \" of\"@9 top5: %s\n", line);
        llama_free(c);

        // full-batch path
        c = llama_init_from_model(model, cp);
        std::vector<llama_token> h(prompt);
        h.insert(h.end(), gen.begin(), gen.begin() + 5);
        llama_batch fb = llama_batch_get_one(h.data(), (int32_t) h.size());
        llama_decode(c, fb);
        l = llama_get_logits_ith(c, -1);
        softmax_top5(l, tk, pr);
        line[0] = 0;
        for (int i = 0; i < 5; i++) {
            char pp[96]; llama_token_to_piece(vocab, tk[i], pp, sizeof(pp), 0, true);
            char seg[128]; snprintf(seg, sizeof(seg), "%.3f %s | ", pr[i], pp);
            strncat(line, seg, sizeof(line) - strlen(line) - 1);
        }
        fprintf(stderr, "selftest: [E7 batch  ] \" of\"@9 top5: %s\n", line);
        llama_free(c);
    }

    // E6: E4 again but with flash attention disabled. If the mismatch flips to Belgium,
    // the FA kernel for multi-token batches is the culprit.
    for (int fa = 1; fa >= 0; fa--) {
        llama_context_params cp3 = cp;
        cp3.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        llama_context * c = llama_init_from_model(model, cp3);
        std::vector<llama_token> h(prompt);
        h.insert(h.end(), gen.begin(), gen.begin() + 5); // 0..9
        llama_batch bh = llama_batch_get_one(h.data(), (int32_t) h.size());
        llama_decode(c, bh);
        const llama_token last = argmax_(llama_get_logits_ith(c, -1));
        char ls[128];
        llama_token_to_piece(vocab, last, ls, sizeof(ls), 0, true);
        fprintf(stderr, "selftest: [E6 %s    ] full 0..9 batch, last argmax=%d(%s)  %s\n",
                fa ? "FA-on" : "FA-off", last, ls, seq_argmax == last ? "MATCH" : "MISMATCH");
        llama_free(c);
    }

    // E9: negative control for the CPU FA kernel split. The single-token path uses a
    // different kernel (split-KV chunked partials) than multi-token batches (one_chunk)
    // when the KV cache length is >= 512 (ggml-cpu/ops.cpp use_split_kv_path).
    // With n_ctx = 256 the split-KV path is disabled, so both paths run one_chunk.
    // Result: single still gives Belgium, batch still gives Germany -- so the FA kernel
    // split is NOT the cause. The divergence is broader: CPU matmul kernels compute
    // slightly different values for n_tokens=1 vs n_tokens>1, amplified across the 36
    // layers to flip one near-tie argmax. This is core llama.cpp numerics, reproducible
    // with the plain llama.h API (E2/E3), independent of the ConfLayers port.
    {
        llama_context_params cp9 = cp;
        cp9.n_ctx = 256;
        llama_context * cs = llama_init_from_model(model, cp9);
        llama_batch bh = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
        llama_decode(cs, bh);
        for (int i = 0; i < 5; i++) decode1(cs, gen[i], (llama_pos) (prompt.size() + i));
        const llama_token e9s = argmax_(llama_get_logits_ith(cs, -1));
        char e9ss[128];
        llama_token_to_piece(vocab, e9s, e9ss, sizeof(e9ss), 0, true);
        fprintf(stderr, "selftest: [E9 single ] n_ctx=256 \" of\"@9 argmax=%d(%s)\n", e9s, e9ss);
        llama_free(cs);

        llama_context * cb = llama_init_from_model(model, cp9);
        std::vector<llama_token> h(prompt);
        h.insert(h.end(), gen.begin(), gen.begin() + 3); // 0..7
        llama_batch hb = llama_batch_get_one(h.data(), (int32_t) h.size());
        llama_decode(cb, hb);
        llama_batch tb = llama_batch_init(2, 0, 1);
        tb.n_tokens = 2;
        tb.token[0] = gen[3]; tb.pos[0] = (llama_pos) h.size();      tb.n_seq_id[0] = 1; tb.seq_id[0][0] = 0; tb.logits[0] = true;
        tb.token[1] = gen[4]; tb.pos[1] = (llama_pos) h.size() + 1;  tb.n_seq_id[1] = 1; tb.seq_id[1][0] = 0; tb.logits[1] = true;
        llama_decode(cb, tb);
        llama_batch_free(tb);
        const llama_token e9b = argmax_(llama_get_logits_ith(cb, 1));
        char e9bs[128];
        llama_token_to_piece(vocab, e9b, e9bs, sizeof(e9bs), 0, true);
        fprintf(stderr, "selftest: [E9 batch  ] n_ctx=256 [capital@8, of@9] of-node argmax=%d(%s)  %s\n",
                e9b, e9bs, e9s == e9b ? "AGREE" : "DIFFER");
        llama_free(cb);
    }

    return 0;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "Hello my name is";
    std::string model_id = "qwen3";
    int n_predict = 128;
    int ngl = 0;
    bool baseline = false;
    bool greedy_only = false;
    bool verbose = false;
    bool selftest = false;

    llama_conflayer_params cparams;
    cparams.optimize = false;

    for (int i = 1; i < argc; i++) {
        auto need_arg = [&](const char * name) -> bool {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", name);
                return false;
            }
            return true;
        };

        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (need_arg(argv[i])) model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            if (need_arg(argv[i])) prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--n-predict") == 0) {
            if (need_arg(argv[i])) n_predict = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--skip-ratio") == 0) {
            if (need_arg(argv[i])) cparams.skip_ratio = (float) std::atof(argv[++i]);
        } else if (strcmp(argv[i], "--optimize") == 0) {
            cparams.optimize = true;
        } else if (strcmp(argv[i], "--topk") == 0) {
            if (need_arg(argv[i])) cparams.topk = (uint32_t) std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-step-draft") == 0) {
            if (need_arg(argv[i])) cparams.max_step_draft = (uint32_t) std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-tree-seq") == 0) {
            if (need_arg(argv[i])) cparams.max_seq = (uint32_t) std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--stop-threshold") == 0) {
            if (need_arg(argv[i])) cparams.stop_threshold = (float) std::atof(argv[++i]);
        } else if (strcmp(argv[i], "--search-interval") == 0) {
            if (need_arg(argv[i])) cparams.search_interval = (uint32_t) std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--context-window") == 0) {
            if (need_arg(argv[i])) cparams.context_window = (uint32_t) std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-opt-iter") == 0) {
            if (need_arg(argv[i])) cparams.max_opt_iter = (uint32_t) std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-score") == 0) {
            if (need_arg(argv[i])) cparams.max_score = (float) std::atof(argv[++i]);
        } else if (strcmp(argv[i], "--model-id") == 0) {
            if (need_arg(argv[i])) model_id = argv[++i];
        } else if (strcmp(argv[i], "--layer-chunks") == 0) {
            if (need_arg(argv[i])) cparams.layer_chunks = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cluster") == 0) {
            cparams.cluster_attn = true;
        } else if (strcmp(argv[i], "--cluster-sparse") == 0) {
            cparams.cluster_attn = true;
            cparams.cluster_sparse = true;
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (need_arg(argv[i])) cparams.seed = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-ngl") == 0) {
            if (need_arg(argv[i])) ngl = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--verbosity") == 0) {
            if (need_arg(argv[i])) cparams.verbosity = (uint32_t) std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--baseline") == 0) {
            baseline = true;
        } else if (strcmp(argv[i], "--greedy-only") == 0) {
            greedy_only = true;
        } else if (strcmp(argv[i], "--selftest") == 0) {
            selftest = true;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    if (n_predict <= 0) {
        fprintf(stderr, "n_predict must be positive\n");
        return 1;
    }

    cparams.model_id = model_id;
    cparams.max_new_tokens = (uint32_t) n_predict;
    if (verbose) {
        cparams.verbosity = 2; // --verbose upgrades the default, --verbosity wins
    }

    // load dynamic backends
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == NULL) {
        fprintf(stderr, "unable to load model: %s\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (selftest) {
        const int rc = run_selftest(model, vocab);
        llama_model_free(model);
        return rc;
    }

    std::vector<llama_token> prompt_tokens;
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    prompt_tokens.resize(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "failed to tokenize the prompt\n");
        llama_model_free(model);
        return 1;
    }

    printf("prompt (%d tokens):\n", (int) prompt_tokens.size());
    print_tokens(vocab, prompt_tokens, false);
    printf("\n");

    if (greedy_only) {
        const auto g0 = ggml_time_us();
        std::vector<llama_token> gres = run_baseline(model, vocab, prompt_tokens, n_predict);
        const double gdt = (ggml_time_us() - g0) / 1000.0;
        printf("baseline greedy (%d new tokens in %.1f ms):\n", (int) gres.size(), gdt);
        print_tokens(vocab, gres, false);
        llama_model_free(model);
        return 0;
    }

    llama_conflayer conf(model, cparams);
    std::vector<llama_token> result;
    llama_conflayer_metrics metrics;

    const auto t0 = ggml_time_us();
    const bool ok = conf.generate(prompt_tokens, result, &metrics);
    const double dt_ms = (ggml_time_us() - t0) / 1000.0;

    if (!ok) {
        fprintf(stderr, "generate() failed\n");
        llama_model_free(model);
        return 1;
    }

    printf("conflayers (%d new tokens in %.1f ms, %.2f t/s):\n",
           (int) result.size(), dt_ms, result.size() / (dt_ms / 1000.0));
    print_tokens(vocab, result, false);

    fprintf(stderr, "\n  rounds        : %llu\n", (unsigned long long) metrics.n_rounds);
    fprintf(stderr, "  draft tokens  : %llu\n", (unsigned long long) metrics.n_draft_tokens);
    if (metrics.n_rounds > 0) {
        fprintf(stderr, "  tree nodes    : %llu (%.1f nodes/round)\n", (unsigned long long) metrics.n_tree_nodes,
                (double) metrics.n_tree_nodes / (double) metrics.n_rounds);
    }
    fprintf(stderr, "  accepted      : %llu (incl. root)\n", (unsigned long long) metrics.n_accepted + metrics.n_rounds);
    fprintf(stderr, "  accept rate   : %.3f\n", metrics.accept_rate);
    fprintf(stderr, "  opt rounds    : %llu\n", (unsigned long long) metrics.n_opt_rounds);
    if (metrics.decode_us > 0) {
        const double dd = metrics.draft_us / 1000.0, dv = metrics.verify_us / 1000.0, do_ = metrics.opt_us / 1000.0;
        const double dtot = metrics.decode_us / 1000.0;
        fprintf(stderr, "  phase split   : draft %.1fms (%.0f%%)  verify %.1fms (%.0f%%)  opt %.1fms (%.0f%%)  other %.1fms\n",
                dd, 100.0 * dd / dtot, dv, 100.0 * dv / dtot, do_, 100.0 * do_ / dtot, dtot - dd - dv - do_);
    }
    if (cparams.optimize) {
        const auto & sk = conf.get_skip_layers();
        fprintf(stderr, "  final skip set: [");
        for (size_t i = 0; i < sk.size(); i++) {
            fprintf(stderr, "%s%d", i ? ", " : "", sk[i]);
        }
        fprintf(stderr, "] (%zu layers)\n", sk.size());
    }

    if (baseline) {
        const auto b0 = ggml_time_us();
        std::vector<llama_token> bres = run_baseline(model, vocab, prompt_tokens, n_predict);
        const double bdt = (ggml_time_us() - b0) / 1000.0;
        printf("\nbaseline greedy (%d new tokens in %.1f ms):\n", (int) bres.size(), bdt);
        print_tokens(vocab, bres, false);
    }

    llama_model_free(model);
    return 0;
}
