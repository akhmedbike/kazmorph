#!/usr/bin/env python3
"""End-to-end verification of the ported modules (kazmorph.correction, cv,
official_eval, significance, device) under the project's target runtime.

This is NOT a unit test suite — it's a single-script smoke + functional check
that proves the ported code imports against the real kazmorph engine and
behaves correctly on real data. Run inside the docker container.

Exit code 0 = all checks passed.
"""
from __future__ import annotations
import sys, os, traceback, json

PASS, FAIL = [], []
def check(name, fn):
    try:
        fn(); PASS.append(name); print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, repr(e))); print(f"  FAIL  {name}: {e!r}")

print("=" * 72)
print("  kazmorph ported-modules verification (Python %d.%d)" % sys.version_info[:2])
print("=" * 72)

# ── 1. Imports (the thing that failed on Python 3.9 due to rma.py PEP 604) ──
def t_imports():
    import kazmorph.correction  # full subpackage incl. corrector_3way -> candidate_generator -> suggestor -> rma
    from kazmorph.correction import (correct_token_3way, correct_sentence_3way,
        correct_token_binary, DecisionResult, generate_candidates, candidate_probs)
    from kazmorph.correction import correction_metrics, keyboards, corrupt, freq
    import kazmorph.official_eval
    import kazmorph.cv
    import kazmorph.significance
    from kazmorph.device import pick_device, enable_determinism
    from kazmorph.third_party.conll18_ud_eval import evaluate, load_conllu_file
check("1. all new modules import against real engine", t_imports)

# ── 2. official_eval: gold-vs-gold = 1.0 (acceptance guarantee) ──────────────
# NOTE: the official UD scorer requires (a) a numeric HEAD column and (b) the
# file to end with a blank line. data/test.conllu / data/train.conllu are
# morpho-only (HEAD="_"), and the committed split files lack the trailing
# newline, so we normalise into a temp file first. These are properties of the
# *data files*, not the scorer — the scorer enforces the UD spec correctly.
import shutil
_FULL_UD_SRC = "data/dev_split.conllu"
_FULL_UD = "/tmp/_gold_normalized.conllu"
shutil.copy(_FULL_UD_SRC, _FULL_UD)
with open(_FULL_UD, "a", encoding="utf-8") as _fh:
    _fh.write("\n")  # guarantee the trailing blank line the scorer requires

def t_official_gold():
    from kazmorph.official_eval import evaluate_conllu
    r = evaluate_conllu(_FULL_UD, _FULL_UD)
    assert all(abs(v - 1.0) < 1e-9 for v in r.values()), f"expected all 1.0, got {r}"
check(f"2. official_eval gold-vs-gold = 1.0 on {_FULL_UD_SRC}", t_official_gold)

# ── 3. official_eval: detects a real error (F1 < 1.0) ───────────────────────
def t_official_error(tmp="/tmp/_sys_normalized.conllu"):
    from kazmorph.official_eval import evaluate_conllu
    src = open(_FULL_UD, encoding="utf-8").read()
    # corrupt exactly one lemma to force Lemmas/UPOS < 1.0
    sys_text = src.replace("\tерназар\t", "\tерназарX\t", 1)
    if sys_text == src:  # fallback: corrupt any NOUN lemma
        import re
        sys_text = re.sub(r"(\t)([әғқңөұүһіа-яA-Za-z]+)(\tNOUN\t)", r"\1XX\3", src, count=1)
    open(tmp, "w", encoding="utf-8").write(sys_text)
    r = evaluate_conllu(_FULL_UD, tmp)
    assert r["Lemmas"] < 1.0 or r["UPOS"] < 1.0, f"expected <1.0 after corruption, got {r}"
check("3. official_eval detects lemma corruption (metric < 1.0)", t_official_error)

# ── 4. cv: leakage-free k-fold over the real gold pool ──────────────────────
def t_cv():
    from kazmorph.cv import make_folds
    from kazmorph.corpus import load_pool
    sents = load_pool()  # dedup union of all non-silver .conllu in data/
    n = len(sents)
    assert n > 0, "load_pool() returned no sentences"
    folds = make_folds(sents, k=10, seed=42, strategy="stratified")
    # jack-knife: every sentence tested exactly once
    all_test = sorted(i for f in folds for i in f.test_idx)
    assert all_test == list(range(n)), "stratified jack-knife violated"
    # disjointness within each fold
    for f in folds:
        tr, dv, te = set(f.train_idx), set(f.dev_idx), set(f.test_idx)
        assert not (tr & dv) and not (tr & te) and not (dv & te)
        assert te and tr
    # determinism
    folds2 = make_folds(sents, k=10, seed=42, strategy="stratified")
    assert all(a.test_idx == b.test_idx for a, b in zip(folds, folds2))
    print(f"       (pool={n} sentences, 10 folds, invariants hold)")
check("4. cv.make_folds stratified 10-fold over real pool (jack-knife+disjoint+deterministic)", t_cv)

# ── 5. significance: pure functions + Holm step-down ────────────────────────
def t_significance():
    from kazmorph import significance as sig
    # Holm textbook example
    adj = sig.holm_bonferroni([0.01, 0.04, 0.03])
    assert adj == [0.03, 0.06, 0.06], adj  # step-down: 0.04 cannot drop below 0.06
    # Cohen's d: identical -> 0, clearly-different -> large
    assert sig.cohens_d_paired([1,2,3],[1,2,3]) == 0.0
    d = sig.cohens_d_paired([0.9,0.85,0.88,0.92,0.87],[0.82,0.80,0.84,0.86,0.83])
    assert d > 0.8 and sig.effect_label(d) == "large"
    # bootstrap CI reproducible + brackets the mean diff
    lo, hi = sig.bootstrap_ci_diff([0.9,0.85,0.88,0.92,0.87],[0.82,0.80,0.84,0.86,0.83])
    lo2, hi2 = sig.bootstrap_ci_diff([0.9,0.85,0.88,0.92,0.87],[0.82,0.80,0.84,0.86,0.83])
    assert (lo,hi) == (lo2,hi2)
check("5. significance: Holm step-down + Cohen's d + reproducible bootstrap CI", t_significance)

# ── 6. significance: paired_test (needs scipy) ──────────────────────────────
def t_paired_test():
    from kazmorph import significance as sig
    r = sig.paired_test([0.90,0.85,0.88,0.92,0.87,0.89,0.91,0.86,0.88,0.90],
                        [0.82,0.80,0.84,0.86,0.83,0.81,0.85,0.82,0.84,0.83])
    assert "t_p" in r and "wilcoxon_p" in r and "cohens_d" in r
    assert r["mean_diff"] > 0  # first config is clearly better
    assert not (r["wilcoxon_w"] != r["wilcoxon_w"])  # not NaN
check("6. significance.paired_test runs (scipy present)", t_paired_test)

# ── 7. correction_metrics: FCR / correction_prf on a controlled trace ───────
def t_metrics():
    from kazmorph.correction import correction_metrics as m
    decisions = [
        # clean token, kept correctly
        {"gold":"а","original":"а","predicted":"а","decision":"KEEP","was_corrupted":False},
        # corrupted token, corrected to gold
        {"gold":"ана","original":"анм","predicted":"ана","decision":"CORRECT","was_corrupted":True},
        # corrupted token, wrong correction
        {"gold":"ана","original":"анм","predicted":"аны","decision":"CORRECT","was_corrupted":True},
        # corrupted token, not corrected (KEEP) -> error survived
        {"gold":"бала","original":"бalm","predicted":"бalm","decision":"KEEP","was_corrupted":True},
        # clean token, suspicious but not changed (REVIEW)
        {"gold":"жүз","original":"жүз","predicted":"жүз","decision":"REVIEW","was_corrupted":False},
    ]
    prf = m.correction_prf(decisions)
    # tp=1 (the correct CORRECT), fp=1 (the wrong CORRECT), fn=2 (wrong CORRECT was corrupted + KEEP-on-corrupted)
    assert prf["tp"]==1 and prf["fp"]==1 and prf["fn"]==2, prf
    fcr = m.false_correction_rate(decisions)
    # clean tokens: 2 (the two was_corrupted=False); changed: 0 (REVIEW doesn't change; the CORRECTs were on corrupted)
    assert fcr["n_correct"]==2 and fcr["n_changed"]==0, fcr
    assert fcr["fcr"]==0.0
check("7. correction_metrics.correction_prf + false_correction_rate counting", t_metrics)

# ── 8. corrupt: generates synthetic errors over the real pool ───────────────
def t_corrupt():
    import random
    from kazmorph.corpus import load_pool
    from kazmorph.correction.corrupt import corrupt_sentence, summarise_pairs, OP_DELETION, OP_HARMONY
    from kazmorph.rma import ReverseMorphAnalyzer
    sents = load_pool()
    rng = random.Random(0)
    rma = ReverseMorphAnalyzer()
    pairs = [corrupt_sentence(s, rng, rma, corrupt_prob=1.0) for s in sents[:200]]
    summ = summarise_pairs(pairs)
    assert summ["n_sentences"] == 200
    assert summ["n_corrupted"] > 0, "no corruptions produced — operators broken"
    # at least 2 distinct operator categories fired (variety)
    assert len(summ["category_counts"]) >= 2, summ["category_counts"]
    print(f"       (200 sentences: {summ['n_corrupted']} corrupted, "
          f"{len(summ['category_counts'])} op categories: "
          f"{list(summ['category_counts'].keys())[:4]}...)")
check("8. corrupt generates multi-operator synthetic errors over real pool", t_corrupt)

# ── 9. corrector_3way + candidate_generator: runs on a real Pipeline ────────
def t_3way_pipeline():
    from kazmorph.pipeline import Pipeline
    from kazmorph.correction.corrector_3way import correct_sentence_3way
    p = Pipeline(
        crf_model_path="models/crf_model.joblib" if os.path.exists("models/crf_model.joblib") else None,
        neural_model_path="models/neural_pos" if os.path.isdir("models/neural_pos") else None,
        lemma_model_path="models/neural_lemma" if os.path.isdir("models/neural_lemma") else None,
    )
    text = "Менің атым Данияр бүгін жаңа мектепке бардым."
    decisions = correct_sentence_3way(p, text)
    valid = {"KEEP","CORRECT","REVIEW"}
    assert all(d.decision in valid for d in decisions), "unexpected decision label"
    # high-confidence tokens should mostly be KEEP; clean text shouldn't be over-corrected
    n_keep = sum(1 for d in decisions if d.decision=="KEEP")
    assert n_keep >= len(decisions)//2, f"too few KEEP on clean text: {n_keep}/{len(decisions)}"
    print(f"       ({len(decisions)} tokens: KEEP={n_keep}, "
          f"CORRECT={sum(1 for d in decisions if d.decision=='CORRECT')}, "
          f"REVIEW={sum(1 for d in decisions if d.decision=='REVIEW')})")
check("9. corrector_3way runs on real Pipeline (clean text → mostly KEEP)", t_3way_pipeline)

# ── 10. candidate_generator: all 4 modes produce ranked candidates ──────────
def t_candidate_modes():
    from kazmorph.correction.candidate_generator import generate_candidates, candidate_probs
    token = "мектепке"  # dative 'to school' — a real Kazakh word
    for mode in ("dl","dl_freq","dl_fst_filter","dl_freq_morph"):
        cands = generate_candidates(token, mode=mode, max_candidates=5)
        # modes may return different counts, but each candidate is (str, float)
        for w, e in cands:
            assert isinstance(w, str) and isinstance(e, float)
        # candidate_probs over the set including original
        probs = candidate_probs(cands, include_original=True, original_token=token)
        total = sum(p for _, p in probs)
        assert abs(total - 1.0) < 1e-6, f"{mode}: probs sum to {total}, not 1.0"
check("10. candidate_generator: 4 modes + softmax normalisation (sum=1)", t_candidate_modes)

# ── 11. device: pick_device + enable_determinism ────────────────────────────
def t_device():
    from kazmorph.device import pick_device, enable_determinism
    d = pick_device()
    assert str(d) in ("cpu","cuda","mps"), d
    enable_determinism(42)
    import torch; assert torch.initial_seed() == 42
check("11. device.pick_device + enable_determinism(42) seeds torch", t_device)

# ── 12. freq: leakage-free per-fold dict excludes test sentences ────────────
def t_freq_leakage():
    from kazmorph.corpus import load_pool
    from kazmorph.cv import make_folds
    from kazmorph.correction.freq import build_fold_frequency_dict, get_frequency_dict
    sents = load_pool()
    folds = make_folds(sents, k=10, seed=42)
    test_sents = [sents[i] for i in folds[0].test_idx]
    global_dict = get_frequency_dict()
    fold_dict = build_fold_frequency_dict(test_sents)
    # last_counts is stashed as a function attribute AFTER the call above
    provenance = build_fold_frequency_dict.last_counts
    assert provenance["n_source_sentences_excluded"] > 0, provenance
    # the fold dict excludes held-out sentences, so it has at most as many
    # unique forms as the global dict
    assert len(fold_dict) <= len(global_dict), (len(fold_dict), len(global_dict))
    print(f"       (global={len(global_dict)} forms, fold={len(fold_dict)} forms, "
          f"excluded {provenance['n_source_sentences_excluded']} held-out sentences)")
check("12. freq.build_fold_frequency_dict excludes held-out sentences", t_freq_leakage)

# ── summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"  RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILURES:")
    for name, err in FAIL:
        print(f"    - {name}: {err}")
print("=" * 72)
sys.exit(0 if not FAIL else 1)
