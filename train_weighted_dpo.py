"""
Weighted DPO training on anonymous/Dolci-Instruct-DPO-en, starting from
allenai/Olmo-3-7B-Instruct-SFT, using the local WeightedDPOTrainer.

Each token in the chosen/rejected completion is scaled by a per-token weight
before the log-prob sum in the DPO loss.  When all weights are 1.0, this
reduces to standard DPO.

Hyperparameters follow the Olmo 3 paper (arXiv:2512.13961), Table 48,
"7B Instruct DPO" column, with β scaled down to 0.05 for TRL's
non-length-normalized sigmoid DPO loss.

Expected dataset columns after preprocessing:
    prompt, chosen, rejected, chosen_weights, rejected_weights
"""
from dotenv import load_dotenv
load_dotenv(".env")
import json
import os
import random
import torch
import wandb

from collections import namedtuple

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from trl import DPOConfig
from weighted_dpo_trainer import WeightedDPOTrainer
import cluster_common as CL

SEED = 42
set_seed(SEED)

# ---- Config ----
MODEL_NAME = "allenai/Olmo-3-7B-Instruct-SFT"
DATASET_NAME = "anonymous/Dolci-Instruct-DPO-en"
# Must match the SAVE_DIR of the gradient-products script whose dumps you train
# on (compute_dpo_gradient_products_pissa_lora.py here).
GRADIENT_PRODUCTS_DIR = "/data/weighted-dpo/dpo-gradient-products-pissa-lora"

# ---- Which saved "chosen direction" the per-token cosine is measured against --
#
# compute_dpo_gradient_products_pissa_lora.py dumps each rejected token's
# gradient dotted against SIX chosen directions: {single (this example's chosen
# gradient), aggregated (the dataset-mean chosen gradient)} x {unweighted,
# weighted_norm1, weighted_norm2}. The per-token cosine is
#     cos = products_<dir> / (norms_rejected * ||chosen_dir||).
# GRADIENT_DIRECTION picks which dump to use. The chosen-direction norm comes
# from a per-example file for the "single_*" dirs, but is a GLOBAL scalar (read
# once from mean_chosen_norms.json) for the "aggregated_*" dirs -- so each entry
# records (products_file, per_example_norm_file, global_norm_key); exactly one of
# the latter two is set. Add a row to extend.
#
# The "max_*" directions are SYNTHETIC (mirror the viz's max_<suffix>): there is
# no saved dump for them. Their per-token cosine is the elementwise max of the
# single and aggregated cosines for the same weighting, so each token follows
# whichever of the two chosen directions it aligns with more strongly. They are
# resolved to their two base directions (MAX_DIRECTIONS) and computed on the fly.
GRADIENT_DIRECTION = "max_weighted_norm1" #"maxclu_mean_K8" #"maxclu_mean_K4"
GRADIENT_DIRECTIONS = {
    # name                          products file                              per-example norm file      global norm key
    "single_unweighted":          ("products_single_unweighted.pt",          "norm_unweighted.pt",     None),
    "single_weighted_norm1":      ("products_single_weighted_norm1.pt",      "norm_weighted_norm1.pt", None),
    "single_weighted_norm2":      ("products_single_weighted_norm2.pt",      "norm_weighted_norm2.pt", None),
    "aggregated_unweighted":      ("products_aggregated_unweighted.pt",      None, "norm_mean_unweighted"),
    "aggregated_weighted_norm1":  ("products_aggregated_weighted_norm1.pt",  None, "norm_mean_weighted_norm1"),
    "aggregated_weighted_norm2":  ("products_aggregated_weighted_norm2.pt",  None, "norm_mean_weighted_norm2"),
}
# Synthetic max directions: name -> (single base, aggregated base).
MAX_DIRECTIONS = {
    "max_unweighted":     ("single_unweighted",     "aggregated_unweighted"),
    "max_weighted_norm1": ("single_weighted_norm1", "aggregated_weighted_norm1"),
    "max_weighted_norm2": ("single_weighted_norm2", "aggregated_weighted_norm2"),
}

# ---- Cluster directions (the K-modes replacement for the one dataset mean) ----
#
# compute_cluster_products.py dumps products_cluster_<kind>_<w>.pt, an fp32 [K, T]
# matrix whose row k is every rejected token dotted against cluster k's mean
# chosen gradient. These directions max over those K rows instead of (or as well
# as) the single dataset-mean direction, so a token follows whichever chosen MODE
# it aligns with rather than an average that may sit between all of them:
#   maxclu_<kind>       max over {this example's own chosen gradient} u {K means}
#                       -- the drop-in replacement for max_weighted_norm1
#   clusteronly_<kind>  max over the K means alone (no per-example direction)
# <kind> is how the centroid was built: "mean" (raw per-cluster mean, the exact
# analogue of aggregated_*) or "meannorm" (mean of unit-norm gradients). Cluster
# dot products are rescaled by CL.CLUSTER_DOT_SCALE first -- without that, the
# max is decided by which centroid is longest rather than by alignment; see
# cluster_common. Names match 05_correlate.py's so a direction that wins on the
# ground-truth subsample is trained here under the same name.
# A cluster direction is named maxclu_<kind>_K<k> / clusteronly_<kind>_K<k>, and
# that name is PARSED rather than looked up: the kind and K it encodes are what
# select the products file, so several K can coexist in one products directory
# and be trained by name. Names match 05_correlate.py's exactly, so whichever row
# wins the ground-truth table is trained by copying its name here.
def cluster_direction_spec(name):
    """(kind, k, include_single) for a cluster direction, or None."""
    parsed = CL.parse_direction(name)
    if parsed is None:
        return None
    prefix, kind, k = parsed
    return kind, k, prefix == "maxclu"
# The per-example direction a maxclu_* maxes against (the weighting is fixed by
# the cluster run: centroids and products only exist for CL.WEIGHTING).
CLUSTER_SINGLE_BASE = f"single_{CL.WEIGHTING}"
# Global norm key of the dataset-mean chosen gradient, the reference scale for
# CLUSTER_DOT_SCALE == "aggregated".
CLUSTER_AGG_NORM_KEY = f"norm_mean_{CL.WEIGHTING}"

CLUSTER_SPEC = cluster_direction_spec(GRADIENT_DIRECTION)
IS_CLUSTER = CLUSTER_SPEC is not None
_ALL_DIRECTIONS = set(GRADIENT_DIRECTIONS) | set(MAX_DIRECTIONS)
assert IS_CLUSTER or GRADIENT_DIRECTION in _ALL_DIRECTIONS, (
    f"GRADIENT_DIRECTION must be one of {sorted(_ALL_DIRECTIONS)} or a cluster "
    f"direction <{'|'.join(CL.CLUSTER_PREFIXES)}>_<kind>_K<k> "
    f"(e.g. 'maxclu_mean_K8'), got {GRADIENT_DIRECTION!r}"
)
# Parallelism for the dataset .map(). Formatting is I/O-bound (each example
# reads three small .pt files off a network filesystem), so oversubscribing
# cores speeds it up. Override with DATASET_NUM_PROC env var.
DATASET_NUM_PROC = int(os.environ.get("DATASET_NUM_PROC", min(32, (os.cpu_count() or 8))))
# WEIGHT_MODE: "weighted" to use precomputed gradient-product weights,
#              "uniform" for all weights = 1.0 (standard DPO)
WEIGHT_MODE = "weighted"

# ---- How each rejected token's weight is calculated from its gradient scores --
#
# Each rejected token comes with two scores against the chosen direction:
#   c  = cos(grad_token, chosen_direction) in [-1, 1]  (d = 1 - c is the cosine
#        distance in [0, 2]; high d => the token's gradient is dissimilar to the
#        chosen direction, i.e. the tokens we want to penalize most)
#   p  = grad_token . chosen_direction, the RAW dot product (the cosine's
#        numerator, so it keeps the token's gradient magnitude -- an unnormalized,
#        example-dependent scale rather than a bounded one)
#   pn = p / max|p| over this example's tokens, in [-1, 1] (the dot rescaled per
#        example so it is comparable across examples like the cosine is)
# A "weight method" turns those into a per-token weight via the SAME three
# orthogonal knobs the live visualizer exposes
# (gradient_groundtruth_eval/06_visualize_token_scores.py), so any formula you
# settle on in the browser can be reproduced here by name:
#
#   1. formula  f(c, p, pn) -> raw per-token weight   (see WEIGHT_FORMULAS below)
#   2. norm     per-sequence normalization over the rejected completion:
#                 "none"    -> raw weights, as-is
#                 "softmax" -> softmax(raw / T)     (sums to 1)
#                 "l1"      -> raw / sum(raw)        (sums to 1)
#   3. T        softmax temperature WEIGHT_TEMPERATURE (only used by "softmax")
#
# This is the SOLE place the score -> weight mapping happens; the resulting
# weights are handed to the trainer, which then applies WEIGHT_SUM_MODE (an
# orthogonal, trainer-side rescale -- see below). To extend, add a lambda to
# WEIGHT_FORMULAS and/or a row to WEIGHT_METHODS; keep names in sync with the
# PRESETS list in 06_visualize_token_scores.py.

# Per-token score bundle handed to every formula: the viz's c / p / pn variables,
# each a 1-D tensor over the rejected completion's tokens.
TokenScores = namedtuple("TokenScores", ["c", "p", "pn"])

# Slope k of the "sigmoid_*" formulas: sigmoid(-score * k), a soft step at
# score = 0 that keeps tokens whose gradient OPPOSES the chosen direction and
# suppresses the aligned ones. k controls how graded that step is: k ~ 1-10
# weights tokens by HOW strongly they oppose (k = 1 keeps everything in
# [0.27, 0.73] for a score in [-1, 1]), while large k (100, the viz's steep end)
# degenerates into a hard 0/1 mask on sign(score) -- and since sign(p) == sign(c)
# always, at that end sigmoid_c and sigmoid_dot collapse onto the same mask.
# Both formulas take bounded scores (c and pn, each in [-1, 1]), so k means the
# same thing to both.
SIGMOID_SLOPE = 1.0

# f(v) registry: name -> vectorized op on a TokenScores bundle.
WEIGHT_FORMULAS = {
    # --- cosine-based (bounded scores) ---
    "token_scale": lambda v: ((1.0 - v.c) / 2.0).clamp(0.0, 1.0),  # (1-c)/2, in [0, 1]
    "exp_half":    lambda v: torch.exp(1.0 - v.c) / 2.0,           # exp(1-c)/2
    "exp_m1":      lambda v: torch.exp(1.0 - v.c) - 1.0,           # exp(1-c)-1
    "dist_sq":     lambda v: (1.0 - v.c).pow(2) / 2.0,             # (1-c)^2 / 2
    "sigmoid_c":   lambda v: torch.sigmoid(-v.c * SIGMOID_SLOPE),  # sigmoid(-c*k)
    "one_minus_c": lambda v: 1.0 - v.c,                            # 1-c
    "raw_cos":     lambda v: v.c,                                  # c (raw cosine)
    # --- dot-product-based ---
    # NOTE: p is unnormalized, so "raw_dot"/"neg_dot" are only meaningful under a
    # per-sequence norm ("softmax"/"l1") or a trainer-side WEIGHT_SUM_MODE that
    # rescales ("normalize"/"token_count"); pairing them with "none" + a
    # "token_scale" sum mode feeds raw gradient magnitudes into the loss. The
    # bounded pn is what the fixed-scale formulas ("dot_scale", "sigmoid_dot")
    # use, so their curve means the same thing on every example.
    "raw_dot":     lambda v: v.p,                                  # p (raw dot)
    "norm_dot":    lambda v: v.pn,                                 # p / max|p|
    "neg_dot":     lambda v: -v.p,                                 # -p
    "dot_scale":   lambda v: ((1.0 - v.pn) / 2.0).clamp(0.0, 1.0),  # (1-pn)/2, in [0, 1]
    "sigmoid_dot": lambda v: torch.sigmoid(-v.pn * SIGMOID_SLOPE),  # sigmoid(-pn*k)
}

# Method registry: name -> (formula key, normalization). Mirrors the viz presets.
WEIGHT_METHODS = {
    # name           formula        norm
    "token_scale": ("token_scale", "none"),     # length-independent (1-cos)/2
    "exp_half":    ("exp_half",    "none"),
    "exp_m1":      ("exp_m1",      "none"),
    "dist_sq":     ("dist_sq",     "none"),
    "sigmoid_c":   ("sigmoid_c",   "none"),      # sigmoid(-cos*k)
    "sigmoid":     ("sigmoid_c",   "none"),      # back-compat alias of "sigmoid_c"
    "altcos":      ("one_minus_c", "softmax"),   # softmax((1-cos)/T)
    "raw_cos":     ("raw_cos",     "none"),
    # dot-product counterparts of the above (viz: "raw dot product p",
    # "normalized dot pn", "-p +softmax(/T)")
    "raw_dot":     ("raw_dot",     "none"),      # p, as-is
    "norm_dot":    ("norm_dot",    "none"),      # pn = p / max|p|
    "altdot":      ("neg_dot",     "softmax"),   # softmax(-p/T)
    "dot_scale":   ("dot_scale",   "none"),      # length-independent (1-pn)/2
    "sigmoid_dot": ("sigmoid_dot", "none"),      # sigmoid(-pn*k)
}

# Pick the weight-calculation method here (key into WEIGHT_METHODS).
WEIGHT_METHOD = "sigmoid_dot"
# Softmax temperature T; used only by softmax-normalized methods (e.g. "altcos").
WEIGHT_TEMPERATURE = 0.1

assert WEIGHT_METHOD in WEIGHT_METHODS, (
    f"WEIGHT_METHOD must be one of {sorted(WEIGHT_METHODS)}, got {WEIGHT_METHOD!r}"
)
WEIGHT_FORMULA, WEIGHT_NORM = WEIGHT_METHODS[WEIGHT_METHOD]

# WEIGHT_SUM_MODE: orthogonal TRAINER-side rescale applied to the weights above,
#   inside the loss (needs ref log-probs / post-truncation token counts the
#   dataset can't see):
#     "normalize"    -> weights sum to 1 per sequence
#     "reward_match" -> weighted reward sum = unweighted (standard DPO) reward sum
#     "token_count"  -> weights sum to the number of completion tokens (mean = 1)
#     "token_scale"  -> apply the weights as-is (no per-sequence renormalization,
#                       so the weight does not depend on the token count)
#     "ignore"       -> ignore weights; every token weight = 1 (standard DPO)
#   Recommended pairings: a "none"-normalized method (e.g. "token_scale") with
#   "token_scale" here (keeps length-independence); a softmax/l1 method with
#   "token_count" (current production default) or "normalize".
WEIGHT_SUM_MODE = "token_scale"
# Force chosen-side token weights to 1 (standard DPO on the chosen half),
# regardless of WEIGHT_SUM_MODE; only rejected tokens carry the weighting.
FORCE_CHOSEN_WEIGHT_ONE = True

# CONTROL EXPERIMENT. Set to a float to give EVERY rejected token that constant
# weight, ignoring the per-token scores entirely. None = normal operation.
#
# Why this exists: with WEIGHT_METHOD = "sigmoid_dot" at SIGMOID_SLOPE = 1 the
# weights are confined to [sigmoid(-1), sigmoid(1)] = [0.269, 0.731] and average
# near 0.5, so the rejected branch is uniformly attenuated ~2x relative to the
# chosen branch, which is held at 1. That alone changes the DPO fixed point --
# the margin optimised becomes beta*(s_c - 0.5*s_r) -- and is an alternative
# explanation for the method's low-beta robustness that has nothing to do with
# gradient alignment. Setting this to the measured mean weight isolates the two:
# if the constant reproduces the result, the alignment signal is not what is
# doing the work.
#
# The dataset is still filtered by GRADIENT_DIRECTION's product files, so this
# trains on EXACTLY the examples the weighted run does; only the weight values
# change. Without that the control would confound the weighting with the
# training set.
CONSTANT_REJECTED_WEIGHT = None

# CONTROL EXPERIMENT, second rung. None = off; (lo, hi) draws every rejected
# token's weight from U[lo, hi], ignoring the scores entirely.
#
# Where the constant control removes all variance, this keeps variance but makes
# it meaningless: the weights still spread across tokens, they just carry no
# information about which token opposes the chosen direction. Together the two
# separate "the rejected branch is attenuated" from "the attenuation is targeted".
#
# Note the distribution is NOT matched to the real one: sigmoid_dot at slope 1
# is confined to [0.269, 0.731] with a far smaller spread than U[0, 1], so this
# changes the weight distribution as well as the correspondence. Use (0.269,
# 0.731) instead if you want the range matched.
#
# Mutually exclusive with CONSTANT_REJECTED_WEIGHT. Draws are seeded per example
# from SEED and the example index, so they are identical across ranks and across
# num_proc map() workers, and change when SEED does.
RANDOM_REJECTED_WEIGHTS = (0.0, 1.0)

assert CONSTANT_REJECTED_WEIGHT is None or RANDOM_REJECTED_WEIGHTS is None, (
    "CONSTANT_REJECTED_WEIGHT and RANDOM_REJECTED_WEIGHTS are alternative "
    "controls; set at most one")


def control_rng(original_idx):
    """Per-example RNG. An int seed, not hash((SEED, idx)): tuple hashing is
    salted per process, so a hash-based seed would differ across map() workers
    and across ranks."""
    return random.Random(SEED * 1_000_003 + original_idx)

# Olmo 3 paper, Table 48, 7B Instruct DPO column.
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 1.0e-5
NUM_EPOCHS = 1
BATCH_SIZE = 1
EFFECTIVE_BATCH_SIZE = 128
BETA = 0.002
WARMUP_RATIO = 0.1

# A cluster direction is only reproducible together with K and the dot rescale, so
# both go in the run name; other directions keep the existing naming untouched.
# The direction name already carries K, so only the dot rescale needs adding.
DIRECTION_TAG = (f"{GRADIENT_DIRECTION}-{CL.CLUSTER_DOT_SCALE}"
                 if IS_CLUSTER else GRADIENT_DIRECTION)

# Two more knobs change every weight but were invisible in the run name. They are
# appended ONLY when set away from their default, so names of existing runs are
# unchanged and a plain name still means "defaults".
METHOD_TAG = WEIGHT_METHOD
if CONSTANT_REJECTED_WEIGHT is not None:
    # The formula is not used at all, so naming the run after it would be a lie.
    METHOD_TAG = f"const{CONSTANT_REJECTED_WEIGHT:g}"
elif RANDOM_REJECTED_WEIGHTS is not None:
    lo, hi = RANDOM_REJECTED_WEIGHTS
    METHOD_TAG = f"rand{lo:g}-{hi:g}"
elif "sigmoid" in WEIGHT_FORMULA and SIGMOID_SLOPE != 1.0:
    METHOD_TAG += f"_k{SIGMOID_SLOPE:g}"
if not FORCE_CHOSEN_WEIGHT_ONE:
    METHOD_TAG += "_chosenweighted"

OUTPUT_DIR = f"/checkpoints/weighted-dpo/olmo3-7b-{WEIGHT_MODE}-dpo-full-{DIRECTION_TAG}-{METHOD_TAG}-{WEIGHT_SUM_MODE}_temp_{WEIGHT_TEMPERATURE}_beta_{BETA}_lr_{LEARNING_RATE}"

local_rank = int(os.environ.get("LOCAL_RANK", 0))
num_gpus = int(os.environ.get("WORLD_SIZE", 1))
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // (BATCH_SIZE * num_gpus)
assert EFFECTIVE_BATCH_SIZE % (BATCH_SIZE * num_gpus) == 0

run_name = OUTPUT_DIR.split("/")[-1]
if local_rank == 0:
    wandb_config = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "weight_mode": WEIGHT_MODE,
        "gradient_direction": GRADIENT_DIRECTION,
        "cluster_k": CLUSTER_SPEC[1] if IS_CLUSTER else None,
        "cluster_kind": CLUSTER_SPEC[0] if IS_CLUSTER else None,
        "cluster_dot_scale": CL.CLUSTER_DOT_SCALE if IS_CLUSTER else None,
        "cluster_weighting": CL.WEIGHTING if IS_CLUSTER else None,
        "centroids_subdir": CL.CENTROIDS_SUBDIR if IS_CLUSTER else None,
        "gradient_products_dir": GRADIENT_PRODUCTS_DIR,
        "weight_method": WEIGHT_METHOD,
        "weight_formula": WEIGHT_FORMULA,
        "weight_norm": WEIGHT_NORM,
        "weight_temperature": WEIGHT_TEMPERATURE,
        "sigmoid_slope": SIGMOID_SLOPE,
        "weight_sum_mode": WEIGHT_SUM_MODE,
        "force_chosen_weight_one": FORCE_CHOSEN_WEIGHT_ONE,
        "constant_rejected_weight": CONSTANT_REJECTED_WEIGHT,
        "random_rejected_weights": RANDOM_REJECTED_WEIGHTS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "beta": BETA,
        "warmup_ratio": WARMUP_RATIO,
        "num_gpus": num_gpus,
    }
    wandb.init(project="weighted-dpo", name=run_name, config=wandb_config)


def write_run_config(output_dir):
    """Dump the fully resolved config next to the checkpoint.

    A directory name can only carry so much before it stops being readable, and
    the things it omits are exactly the ones that are painful to reconstruct
    months later — which centroid set the cluster products were dotted against,
    for instance, is recorded nowhere else: the products FILENAME carries kind and
    K but not provenance, so `centroids_full` and `centroids_full_n3000` are
    indistinguishable after the fact. Read back from the sidecar here so the
    checkpoint is self-describing.
    """
    cfg = {
        "run_name": os.path.basename(output_dir),
        "model": MODEL_NAME, "dataset": DATASET_NAME,
        "gradient_products_dir": GRADIENT_PRODUCTS_DIR,
        "weight_mode": WEIGHT_MODE,
        "gradient_direction": GRADIENT_DIRECTION,
        "weight_method": WEIGHT_METHOD, "weight_formula": WEIGHT_FORMULA,
        "weight_norm": WEIGHT_NORM, "weight_temperature": WEIGHT_TEMPERATURE,
        "sigmoid_slope": SIGMOID_SLOPE, "weight_sum_mode": WEIGHT_SUM_MODE,
        "constant_rejected_weight": CONSTANT_REJECTED_WEIGHT,
        "random_rejected_weights": RANDOM_REJECTED_WEIGHTS,
        "force_chosen_weight_one": FORCE_CHOSEN_WEIGHT_ONE,
        "max_seq_length": MAX_SEQ_LENGTH, "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE, "beta": BETA,
        "warmup_ratio": WARMUP_RATIO, "seed": SEED,
        "cluster": None,
    }
    if IS_CLUSTER:
        kind, k, with_single = CLUSTER_SPEC
        cluster = {"kind": kind, "K": k, "includes_own_chosen": with_single,
                   "weighting": CL.WEIGHTING,
                   "dot_scale": CL.CLUSTER_DOT_SCALE,
                   "centroids_subdir": CL.CENTROIDS_SUBDIR,
                   "products_file": CL.products_filename(kind, k)}
        path = os.path.join(GRADIENT_PRODUCTS_DIR, CL.cluster_norms_filename(k))
        if os.path.isfile(path):
            with open(path) as f:
                info = json.load(f)
            # The authoritative provenance: which centroids these products were
            # actually computed against, as recorded by compute_cluster_products.
            cluster["centroids_dir"] = info.get("centroids_dir")
            cluster["cluster_norms"] = info.get("norms")
            cluster["cluster_sizes"] = info.get("counts")
        cfg["cluster"] = cluster
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote {os.path.join(output_dir, 'run_config.json')}")
    if IS_CLUSTER:
        print(f"  cluster provenance: K={cfg['cluster']['K']} "
              f"kind={cfg['cluster']['kind']} "
              f"dot_scale={cfg['cluster']['dot_scale']} "
              f"centroids={cfg['cluster'].get('centroids_dir')}")


# ---- Dataset preparation ----

def compute_token_weights(scores):
    """Map a TokenScores bundle (per-token c / p / pn) to per-token weights.

    Applies the WEIGHT_METHOD formula f(c, p, pn), then its per-sequence
    normalization ("none" / "softmax" / "l1"). This is the single source of truth
    for the score -> weight mapping and stays 1:1 with the live visualizer.
    """
    raw = WEIGHT_FORMULAS[WEIGHT_FORMULA](scores)
    if WEIGHT_NORM == "softmax":
        return torch.softmax(raw / WEIGHT_TEMPERATURE, dim=0)
    if WEIGHT_NORM == "l1":
        return raw / raw.sum().clamp(min=1e-12)
    return raw  # "none"


def base_directions(direction):
    """The saved base direction name(s) a direction resolves to.

    Two for a max_*; for a cluster direction, only the per-example single base it
    maxes against (the cluster products are a separate [K, T] file, handled by
    cluster_product_files).
    """
    if direction in MAX_DIRECTIONS:
        return list(MAX_DIRECTIONS[direction])
    spec = cluster_direction_spec(direction)
    if spec is not None:
        return [CLUSTER_SINGLE_BASE] if spec[2] else []
    return [direction]


def cluster_product_files(direction):
    """The [K, T] cluster products file a direction needs, if any."""
    spec = cluster_direction_spec(direction)
    if spec is None:
        return []
    kind, k, _ = spec
    return [CL.products_filename(kind, k)]


def base_dot_cosine(ex_dir, base_direction, norms, mean_norms):
    """(dot, cosine) of each rejected-token gradient vs one base chosen direction.

    dot    = products_<dir>                                   (raw, unnormalized)
    cosine = products_<dir> / (||grad_token|| * ||chosen_dir||)
    """
    products_file, norm_file, global_key = GRADIENT_DIRECTIONS[base_direction]
    products = torch.load(
        os.path.join(ex_dir, products_file), map_location="cpu", weights_only=True
    ).float()
    if norm_file is not None:
        chosen_norm = float(torch.load(
            os.path.join(ex_dir, norm_file), map_location="cpu", weights_only=True
        ))
    else:
        chosen_norm = mean_norms[global_key]
    return products, products / (norms.clamp(min=1e-12) * max(chosen_norm, 1e-12))


def cluster_dot_cosine(ex_dir, direction, norms, mean_norms, cluster_norms):
    """(dot, cosine) of each rejected-token gradient vs the BEST cluster centroid.

    The K dots live in one products_cluster_<kind>_<w>.pt of shape [K, T]. The max
    over k is taken by cluster_common (shared with 05_correlate.py), which applies
    CLUSTER_DOT_SCALE to the dots first so the max reflects alignment rather than
    ||c_k||. The cosine is scale-free and maxed as-is.
    """
    kind, k, _ = cluster_direction_spec(direction)
    products = torch.load(
        os.path.join(ex_dir, CL.products_filename(kind, k)),
        map_location="cpu", weights_only=True,
    ).float()                                                   # [K, T]
    dot, cos, _ = CL.cluster_dot_cos(products, norms, cluster_norms[kind],
                                     mean_norms[CLUSTER_AGG_NORM_KEY])
    return dot, cos


def direction_scores(ex_dir, norms, mean_norms, cluster_norms=None):
    """TokenScores (c, p, pn) for GRADIENT_DIRECTION.

    For a max_* direction the cosine and the dot are EACH the elementwise max of
    the two base directions' values, taken independently (mirrors the viz). A
    cluster direction maxes over the K centroids the same way, optionally against
    this example's own chosen gradient as well. pn is the per-example rescale
    p / max|p|, so it lands in [-1, 1] like the cosine.
    """
    if GRADIENT_DIRECTION in MAX_DIRECTIONS:
        a, b = MAX_DIRECTIONS[GRADIENT_DIRECTION]
        dot_a, cos_a = base_dot_cosine(ex_dir, a, norms, mean_norms)
        dot_b, cos_b = base_dot_cosine(ex_dir, b, norms, mean_norms)
        dot, cos = torch.maximum(dot_a, dot_b), torch.maximum(cos_a, cos_b)
    elif IS_CLUSTER:
        dot, cos = cluster_dot_cosine(ex_dir, GRADIENT_DIRECTION, norms,
                                      mean_norms, cluster_norms)
        if CLUSTER_SPEC[2]:
            dot_s, cos_s = base_dot_cosine(ex_dir, CLUSTER_SINGLE_BASE, norms,
                                           mean_norms)
            dot, cos = torch.maximum(dot, dot_s), torch.maximum(cos, cos_s)
    else:
        dot, cos = base_dot_cosine(ex_dir, GRADIENT_DIRECTION, norms, mean_norms)
    return TokenScores(c=cos, p=dot, pn=dot / dot.abs().max().clamp(min=1e-12))


def prepare_dataset(tokenizer):
    """
    Load the DPO dataset and attach per-token weights.

    When WEIGHT_MODE == "weighted", rejected weights are derived from each
    rejected-token gradient's alignment with the chosen direction selected by
    GRADIENT_DIRECTION -- either the raw dot product p = products_<dir> (also as
    pn = p / max|p| per example) or the cosine cos = p / (norms_rejected *
    ||chosen_dir||). Examples without that direction's products are skipped. The
    score -> weight mapping is selected by WEIGHT_METHOD (formula over c/p/pn +
    per-sequence normalization; see compute_token_weights and the WEIGHT_METHODS
    registry).
    When WEIGHT_MODE == "uniform", all weights are 1.0 (standard DPO).
    Chosen weights are always 1.0.
    """
    dataset = load_dataset(DATASET_NAME, split="train")

    # Base directions the selected GRADIENT_DIRECTION needs (two for a max_*),
    # and the product files that must be present for each example.
    needed_bases = base_directions(GRADIENT_DIRECTION)
    needed_products = ([GRADIENT_DIRECTIONS[b][0] for b in needed_bases]
                       + cluster_product_files(GRADIENT_DIRECTION))

    # Global mean-chosen norms (used by "aggregated_*" bases, and as the reference
    # scale for cluster dots) plus the per-cluster ||c_k||. Both are read once
    # here and closed over by format_example so they survive num_proc > 1;
    # "single_*" bases use a per-example norm file instead.
    mean_norms, cluster_norms = {}, {}
    if WEIGHT_MODE == "weighted":
        is_cluster = IS_CLUSTER
        if is_cluster or any(GRADIENT_DIRECTIONS[b][2] is not None for b in needed_bases):
            with open(os.path.join(GRADIENT_PRODUCTS_DIR, "mean_chosen_norms.json")) as f:
                mean_norms = {k: float(v) for k, v in json.load(f).items()
                              if isinstance(v, (int, float))}
        if is_cluster:
            kind, k, _ = CLUSTER_SPEC
            norms_path = os.path.join(GRADIENT_PRODUCTS_DIR,
                                      CL.cluster_norms_filename(k))
            if not os.path.isfile(norms_path):
                raise FileNotFoundError(
                    f"{norms_path} not found — run compute_cluster_products.py "
                    f"over {GRADIENT_PRODUCTS_DIR} before training on "
                    f"GRADIENT_DIRECTION={GRADIENT_DIRECTION!r}.")
            with open(norms_path) as f:
                info = json.load(f)
            cluster_norms = info["norms"]
            if kind not in cluster_norms:
                raise KeyError(
                    f"{norms_path} has no {kind!r} centroids (has "
                    f"{sorted(cluster_norms)}). Re-run steps 3-4 with {kind!r} "
                    f"in cluster_common.CENTROID_KINDS.")
            print(f"Cluster direction {GRADIENT_DIRECTION}: K={info['K']} "
                  f"kind={kind} centroids={info.get('centroids_dir')} "
                  f"dot_scale={CL.CLUSTER_DOT_SCALE!r} "
                  f"||c_k||={[round(v, 2) for v in cluster_norms[kind]]}")

        # Pre-scan which examples have all needed product files available
        available_indices = set()
        for idx in range(len(dataset)):
            ex_dir = os.path.join(GRADIENT_PRODUCTS_DIR, f"example_{idx:05d}")
            if all(os.path.exists(os.path.join(ex_dir, pf)) for pf in needed_products):
                available_indices.add(idx)
        print(f"Found gradient products for {len(available_indices)}/{len(dataset)} examples")

        # Filter to only examples with gradient products, and carry each
        # example's original dataset index as a column. This makes the
        # gradient-product lookup self-contained per row, so map() can run with
        # num_proc > 1 without relying on global ordering.
        kept = sorted(available_indices)
        dataset = dataset.select(kept)
        dataset = dataset.add_column("original_idx", kept)

    def normalize_message(msg):
        # Strip extra fields (reasoning_content, function_call, audio, ...) and
        # keep only what the Olmo-3 chat template consumes.
        return {"role": msg["role"], "content": msg["content"] or ""}

    # Boundary used by the gradient script's build_chat_ids (pissa_lora_common):
    # prompt ends right after the last "<|im_start|>assistant" header (no trailing
    # newline), so the completion starts with the "\n" token. Matching this here
    # keeps tokenizer.encode(completion) aligned with the saved products vector.
    HEADER_STR = "<|im_start|>assistant"

    def format_example(example, idx):
        chosen_messages = [normalize_message(m) for m in example["chosen"]]
        rejected_messages = [normalize_message(m) for m in example["rejected"]]

        chosen_full = tokenizer.apply_chat_template(
            chosen_messages, tokenize=False, add_generation_prompt=False,
        )
        rejected_full = tokenizer.apply_chat_template(
            rejected_messages, tokenize=False, add_generation_prompt=False,
        )

        chosen_pos = chosen_full.rfind(HEADER_STR)
        rejected_pos = rejected_full.rfind(HEADER_STR)
        if chosen_pos == -1 or rejected_pos == -1:
            raise RuntimeError(
                f"Could not locate '{HEADER_STR}' in example {idx}; "
                "chat template format may have changed."
            )
        prompt_text = chosen_full[: chosen_pos + len(HEADER_STR)]
        chosen_completion = chosen_full[chosen_pos + len(HEADER_STR):]
        rejected_completion = rejected_full[rejected_pos + len(HEADER_STR):]

        # --- Compute per-token weights ---
        chosen_ids = tokenizer.encode(chosen_completion, add_special_tokens=False)
        rejected_ids = tokenizer.encode(rejected_completion, add_special_tokens=False)

        chosen_weights = [1.0] * len(chosen_ids)

        if WEIGHT_MODE == "weighted" and CONSTANT_REJECTED_WEIGHT is not None:
            # Control: same examples, same chosen weights, flat rejected weights.
            rejected_weights = [CONSTANT_REJECTED_WEIGHT] * len(rejected_ids)
        elif WEIGHT_MODE == "weighted" and RANDOM_REJECTED_WEIGHTS is not None:
            # Control: same examples, same chosen weights, meaningless spread.
            rng = control_rng(example["original_idx"])
            lo, hi = RANDOM_REJECTED_WEIGHTS
            rejected_weights = [rng.uniform(lo, hi) for _ in rejected_ids]
        elif WEIGHT_MODE == "weighted":
            original_idx = example["original_idx"]
            ex_dir = os.path.join(GRADIENT_PRODUCTS_DIR, f"example_{original_idx:05d}")
            norms = torch.load(
                os.path.join(ex_dir, "norms_rejected.pt"), map_location="cpu", weights_only=True
            ).float()

            # Per-token scores vs the selected chosen direction: cosine c in
            # [-1, 1], raw dot p, and per-example normalized dot pn (the
            # elementwise max of single & aggregated for a max_* direction).
            # (1 - c) is high when a rejected token's gradient is DISSIMILAR to
            # the chosen direction, i.e. the tokens we want to penalize most.
            scores = direction_scores(ex_dir, norms, mean_norms, cluster_norms)

            # Scores -> per-token weight via the selected WEIGHT_METHOD.
            weights = compute_token_weights(scores)

            if len(weights) != len(rejected_ids):
                print(f"WARNING: example {original_idx} products length ({len(weights)}) != rejected_ids length ({len(rejected_ids)})")

            if len(weights) >= len(rejected_ids):
                rejected_weights = weights[:len(rejected_ids)].tolist()
            else:
                pad_val = weights.mean().item() if len(weights) > 0 else 1.0
                rejected_weights = weights.tolist() + [pad_val] * (len(rejected_ids) - len(weights))
        else:
            rejected_weights = [1.0] * len(rejected_ids)

        return {
            "prompt": prompt_text,
            "chosen": chosen_completion,
            "rejected": rejected_completion,
            "chosen_weights": chosen_weights,
            "rejected_weights": rejected_weights,
        }

    dataset = dataset.map(
        format_example,
        with_indices=True,
        num_proc=DATASET_NUM_PROC,
        desc="Formatting dataset with weights",
        remove_columns=dataset.column_names,
    )
    return dataset


# ---- Model loading ----

def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token

    # Olmo-3's config.json stores rope_parameters beta_fast/beta_slow as ints;
    # transformers now requires floats. Cast them in the loaded config.
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    rope_params = getattr(config, "rope_parameters", None)
    if isinstance(rope_params, dict):
        for k in ("beta_fast", "beta_slow"):
            if k in rope_params:
                rope_params[k] = float(rope_params[k])

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, config=config, dtype=torch.bfloat16, trust_remote_code=True,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, config=config, dtype=torch.bfloat16, trust_remote_code=True,
    )

    # Olmo-3's generation_config ships with temperature/top_p but do_sample=False,
    # which fails strict validation on checkpoint save in newer transformers.
    if model.generation_config is not None:
        model.generation_config.do_sample = True


    return model, ref_model, tokenizer


# ---- Training ----

def main():
    if local_rank == 0:
        write_run_config(OUTPUT_DIR)
    model, ref_model, tokenizer = load_models()
    dataset = prepare_dataset(tokenizer)

    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=BETA,
        max_length=MAX_SEQ_LENGTH,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="linear",
        warmup_ratio=WARMUP_RATIO,
        optim="adamw_8bit",
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, int(0.2 * len(dataset) // (BATCH_SIZE * GRAD_ACCUM_STEPS * num_gpus))),
        save_total_limit=2,
        report_to="wandb",
        run_name=run_name,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        loss_type="sigmoid",
        seed=SEED,
        data_seed=SEED,
    )

    trainer = WeightedDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        weight_sum_mode=WEIGHT_SUM_MODE,
        force_chosen_weight_one=FORCE_CHOSEN_WEIGHT_ONE,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
