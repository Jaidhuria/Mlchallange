"""Train and run an ML matcher for the business entity resolution challenge."""

from __future__ import annotations

import argparse
import json
import re
import sys
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "dataset" / "processed"
DEFAULT_OUTPUT = ROOT / "output"
MODEL_VERSION = 1
MAX_POSTINGS_PER_TOKEN = 2500
MAX_CANDIDATES_PER_ANCHOR = 250
RANDOM_STATE = 42
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_TOKENS = {
    "and", "at", "ave", "avenue", "building", "co", "company", "corp",
    "corporation", "for", "inc", "incorporated", "limited", "llc", "ltd",
    "of", "pvt", "road", "rd", "street", "st", "suite", "the", "unit",
}
FEATURE_NAMES = [
    "name_ratio", "name_token_sort", "name_token_set", "name_jaccard",
    "name_exact", "name_length_ratio", "name_shared_tokens",
    "address_ratio", "address_token_sort", "address_token_set",
    "address_jaccard", "address_exact", "address_length_ratio",
    "address_shared_tokens", "address_digit_jaccard", "country_exact",
    "name_char_3gram_jaccard", "address_char_3gram_jaccard",
    "name_partial_ratio", "address_partial_ratio",
    "digits_exact_match", "combined_token_jaccard",
]


def read_tsv(path: Path, columns: list[str], limit: int = 0) -> pd.DataFrame:
    options = {"sep": "\t", "dtype": str, "keep_default_na": False, "usecols": columns}
    if limit > 0:
        options["nrows"] = limit
    frame = pd.read_csv(path, **options)
    for column in ("business_name", "business_address", "country"):
        if column in frame.columns:
            frame[column] = frame[column].str.lower().str.strip()
    return frame


def tokens(value: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(value) if len(token) > 1]


def char_ngrams(value: str, n: int = 3) -> set[str]:
    clean = f" {value.strip()} "
    if len(clean) < n:
        return {clean}
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}


def index_tokens(value: str) -> list[str]:
    unique = {token for token in tokens(value) if token not in STOP_TOKENS and len(token) >= 2}
    return sorted(unique, key=lambda token: (-len(token), token))[:8]


def build_candidate_index(targets: pd.DataFrame) -> tuple[dict, list[np.ndarray]]:
    """Build bounded token postings; common keys are dropped to cap memory and fanout."""
    index: dict[str, dict[str, dict[str, array | None]]] = defaultdict(
        lambda: {"n": {}, "a": {}}
    )
    names = targets["business_name"].to_numpy(dtype=object)
    addresses = targets["business_address"].to_numpy(dtype=object)
    countries = targets["country"].to_numpy(dtype=object)

    for row_id, (name, address, country) in enumerate(zip(names, addresses, countries)):
        for field, value in (("n", name), ("a", address)):
            postings = index[country][field]
            for token in index_tokens(value):
                if token not in postings:
                    postings[token] = array("I")
                posting = postings[token]
                if posting is None:
                    continue
                if len(posting) >= MAX_POSTINGS_PER_TOKEN:
                    postings[token] = None
                else:
                    posting.append(row_id)
    target_columns = [names, addresses, countries, targets["entity_id"].to_numpy(dtype=object)]
    return index, target_columns


def generate_candidates(
    anchor: tuple[str, str, str, str], index: dict, target_columns: list[np.ndarray]
) -> list[int]:
    _, name, address, country = anchor
    country_index = index.get(country)
    if country_index is None:
        return []
    votes: dict[int, list[int]] = {}
    for field, value, vote_slot in (("n", name, 0), ("a", address, 1)):
        postings = country_index[field]
        for token in sorted(set(tokens(value))):
            posting = postings.get(token)
            if posting is None:
                continue
            for row_id in posting:
                counts = votes.get(row_id)
                if counts is None:
                    counts = [0, 0]
                    votes[row_id] = counts
                counts[vote_slot] += 1
    ranked = sorted(
        votes,
        key=lambda row_id: (
            target_columns[0][row_id] == name,
            votes[row_id][0],
            target_columns[1][row_id] == address,
            votes[row_id][1],
        ),
        reverse=True,
    )
    return ranked[:MAX_CANDIDATES_PER_ANCHOR]


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def comparison_features(
    name_a: str, address_a: str, country_a: str,
    name_b: str, address_b: str, country_b: str,
) -> list[float]:
    name_tokens_a, name_tokens_b = tokens(name_a), tokens(name_b)
    address_tokens_a, address_tokens_b = tokens(address_a), tokens(address_b)
    digits_a = set(re.findall(r"\d+", address_a))
    digits_b = set(re.findall(r"\d+", address_b))
    
    name_ngrams_a, name_ngrams_b = char_ngrams(name_a, 3), char_ngrams(name_b, 3)
    addr_ngrams_a, addr_ngrams_b = char_ngrams(address_a, 3), char_ngrams(address_b, 3)
    
    comb_a = set(name_tokens_a + address_tokens_a)
    comb_b = set(name_tokens_b + address_tokens_b)

    return [
        fuzz.ratio(name_a, name_b) / 100,
        fuzz.token_sort_ratio(name_a, name_b) / 100,
        fuzz.token_set_ratio(name_a, name_b) / 100,
        jaccard(name_tokens_a, name_tokens_b),
        float(name_a == name_b),
        min(len(name_a), len(name_b)) / max(len(name_a), len(name_b), 1),
        float(len(set(name_tokens_a) & set(name_tokens_b))),
        fuzz.ratio(address_a, address_b) / 100,
        fuzz.token_sort_ratio(address_a, address_b) / 100,
        fuzz.token_set_ratio(address_a, address_b) / 100,
        jaccard(address_tokens_a, address_tokens_b),
        float(address_a == address_b),
        min(len(address_a), len(address_b)) / max(len(address_a), len(address_b), 1),
        float(len(set(address_tokens_a) & set(address_tokens_b))),
        jaccard(digits_a, digits_b),
        float(country_a == country_b),
        jaccard(name_ngrams_a, name_ngrams_b),
        jaccard(addr_ngrams_a, addr_ngrams_b),
        fuzz.partial_ratio(name_a, name_b) / 100,
        fuzz.partial_ratio(address_a, address_b) / 100,
        float(len(digits_a) > 0 and digits_a == digits_b),
        jaccard(comb_a, comb_b),
    ]


def make_pair_data(
    anchors: pd.DataFrame,
    labels: dict[str, set[str]],
    index: dict,
    target_columns: list[np.ndarray],
    collect_candidate_ids: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[str]]]:
    anchor_rows = array("I")
    features = array("f")
    targets_y = bytearray()
    candidate_lists: list[list[str]] = []
    anchor_records = anchors[["entity_id", "business_name", "business_address", "country"]].itertuples(
        index=False, name=None
    )
    for anchor_index, anchor in enumerate(anchor_records):
        candidate_rows = generate_candidates(anchor, index, target_columns)
        if collect_candidate_ids:
            candidate_lists.append([str(target_columns[3][row_id]) for row_id in candidate_rows])
        truth = labels.get(anchor[0], set())
        for row_id in candidate_rows:
            features.extend(
                comparison_features(
                    anchor[1], anchor[2], anchor[3],
                    str(target_columns[0][row_id]), str(target_columns[1][row_id]),
                    str(target_columns[2][row_id]),
                )
            )
            anchor_rows.append(anchor_index)
            targets_y.append(int(target_columns[3][row_id] in truth))
    feature_array = np.frombuffer(features, dtype=np.float32).reshape((-1, len(FEATURE_NAMES)))
    return (
        np.frombuffer(anchor_rows, dtype=np.uint32),
        feature_array,
        np.frombuffer(targets_y, dtype=np.uint8),
        candidate_lists,
    )


def per_anchor_f05(
    anchor_rows: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    truth_counts: np.ndarray,
    threshold: float,
) -> float:
    true_positive = np.zeros(len(truth_counts), dtype=np.int32)
    false_positive = np.zeros(len(truth_counts), dtype=np.int32)
    selected = probabilities >= threshold
    np.add.at(true_positive, anchor_rows, selected & (y_true == 1))
    np.add.at(false_positive, anchor_rows, selected & (y_true == 0))
    scores = np.zeros(len(truth_counts), dtype=np.float64)
    for row in range(len(truth_counts)):
        tp, fp = true_positive[row], false_positive[row]
        fn = int(truth_counts[row]) - tp
        if tp == 0 and fp == 0 and fn == 0:
            scores[row] = 1.0
        elif tp:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            scores[row] = 1.25 * precision * recall / (0.25 * precision + recall)
    return float(scores.mean()) if len(truth_counts) else 0.0


def choose_threshold(
    anchor_rows: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    truth_counts: np.ndarray,
) -> tuple[float, float]:
    if len(probabilities) == 0:
        return 0.5, per_anchor_f05(anchor_rows, y_true, probabilities, truth_counts, 0.5)
    quantiles = np.quantile(probabilities, np.linspace(0.0, 1.0, 101))
    candidates = np.unique(np.concatenate((quantiles, np.linspace(0.0, 1.0, 201))))
    best_threshold, best_score = 0.5, -1.0
    for threshold in candidates:
        score = per_anchor_f05(anchor_rows, y_true, probabilities, truth_counts, float(threshold))
        if score > best_score:
            best_threshold, best_score = float(threshold), score
    return best_threshold, best_score


import optuna
from lightgbm import LGBMClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)


class EnsembleMatcher:
    def __init__(self, lgbm_params: dict | None = None):
        if lgbm_params is None:
            lgbm_params = {
                "n_estimators": 250,
                "learning_rate": 0.05,
                "num_leaves": 45,
                "min_child_samples": 40,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 1.0,
                "reg_lambda": 3.0,
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
                "verbose": -1,
            }
        self.lgbm_params = lgbm_params
        self.lgbm = LGBMClassifier(**lgbm_params)
        self.hgb = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=45,
            min_samples_leaf=40,
            l2_regularization=3.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.lgbm.fit(X, y)
        self.hgb.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_lgb = self.lgbm.predict_proba(X)
        p_hgb = self.hgb.predict_proba(X)
        return 0.6 * p_lgb + 0.4 * p_hgb


def tune_lgbm_with_optuna(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_rows: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    truth_counts: np.ndarray,
    n_trials: int = 15,
) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 350, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 60),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = LGBMClassifier(**params)
        model.fit(train_x, train_y)
        probs = model.predict_proba(valid_x)[:, 1]
        _, f05 = choose_threshold(valid_rows, valid_y, probs, truth_counts)
        return f05

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_params.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1})
    print(f"Optuna Best Macro F0.5: {study.best_value:.4f}", flush=True)
    return best_params


def make_classifier(lgbm_params: dict | None = None) -> EnsembleMatcher:
    return EnsembleMatcher(lgbm_params)


def load_data(data_dir: Path, max_anchors: int, max_targets: int):
    train_s1 = read_tsv(data_dir / "train_source1_clean.tsv", [
        "entity_id", "business_name", "business_address", "country",
    ])
    if max_anchors > 0 and len(train_s1) > max_anchors:
        train_s1 = train_s1.sample(n=max_anchors, random_state=RANDOM_STATE).reset_index(drop=True)
    train_targets = []
    for source in ("source2", "source3"):
        frame = read_tsv(data_dir / f"train_{source}_clean.tsv", [
            "entity_id", "business_name", "business_address", "country",
        ], max_targets)
        train_targets.append(frame)
    targets = pd.concat(train_targets, ignore_index=True)
    ground_truth = read_tsv(data_dir / "train_ground_truth_clean.tsv", [
        "source1_entity_id", "matched_entity_ids",
    ])
    selected_ids = set(train_s1.entity_id)
    ground_truth = ground_truth[ground_truth.source1_entity_id.isin(selected_ids)]
    labels = {
        row.source1_entity_id: set(row.matched_entity_ids.split(",")) - {""}
        for row in ground_truth.itertuples(index=False)
    }
    return train_s1, targets, labels


def sample_anchors(anchors: pd.DataFrame, validation_fraction: float):
    train_indices, valid_indices = train_test_split(
        np.arange(len(anchors)), test_size=validation_fraction,
        random_state=RANDOM_STATE,
    )
    return anchors.iloc[train_indices].reset_index(drop=True), anchors.iloc[valid_indices].reset_index(drop=True)


def train_and_evaluate(args: argparse.Namespace) -> None:
    anchors, targets, labels = load_data(args.data_dir, args.max_anchors, args.max_targets_per_source)
    if len(anchors) < 20:
        raise ValueError("Need at least 20 training anchors; increase --max-anchors.")
    print(f"Loaded anchors={len(anchors):,}; target records={len(targets):,}", flush=True)
    index, target_columns = build_candidate_index(targets)
    print("Candidate index built.", flush=True)
    train_anchors, valid_anchors = sample_anchors(anchors, args.validation_fraction)
    _, train_x, train_y, _ = make_pair_data(train_anchors, labels, index, target_columns)
    valid_rows, valid_x, valid_y, valid_candidates = make_pair_data(
        valid_anchors, labels, index, target_columns, collect_candidate_ids=True
    )
    if len(train_y) == 0 or len(np.unique(train_y)) < 2:
        raise ValueError("Candidate pairs need both matching and non-matching examples. Use more anchors/targets.")
    print(
        f"Pairs: train={len(train_y):,} ({int(train_y.sum()):,} positive); "
        f"validation={len(valid_y):,} ({int(valid_y.sum()):,} positive)", flush=True,
    )
    
    truth_sets = [labels.get(entity_id, set()) for entity_id in valid_anchors.entity_id]
    truth_counts = np.asarray([len(truth) for truth in truth_sets], dtype=np.int32)
    
    lgbm_params = None
    if getattr(args, "tune", False):
        print("Running Optuna hyperparameter optimization...", flush=True)
        lgbm_params = tune_lgbm_with_optuna(train_x, train_y, valid_rows, valid_x, valid_y, truth_counts)

    model = make_classifier(lgbm_params)
    model.fit(train_x, train_y)
    valid_probabilities = model.predict_proba(valid_x)[:, 1] if len(valid_y) else np.array([])
    threshold, macro_f05 = choose_threshold(valid_rows, valid_y, valid_probabilities, truth_counts)
    predicted = valid_probabilities >= threshold
    tp = int(np.sum(predicted & (valid_y == 1)))
    fp = int(np.sum(predicted & (valid_y == 0)))
    candidate_fn = int(np.sum((~predicted) & (valid_y == 1)))
    candidate_recall = sum(len(truth & set(candidates)) for truth, candidates in zip(
        truth_sets, valid_candidates
    )) / max(1, int(truth_counts.sum()))
    exact_anchor = 0
    offset = 0
    for truth, candidates in zip(truth_sets, valid_candidates):
        selected_ids = {
            candidate for candidate, is_selected in zip(
                candidates, predicted[offset:offset + len(candidates)]
            ) if is_selected
        }
        exact_anchor += int(selected_ids == truth)
        offset += len(candidates)
    precision = tp / max(1, tp + fp)
    conditional_recall = tp / max(1, tp + candidate_fn)
    pair_accuracy = float(np.mean(predicted == valid_y)) if len(valid_y) else 0.0
    report = {
        "macro_f0_5": macro_f05,
        "exact_anchor_accuracy": exact_anchor / max(1, len(valid_anchors)),
        "candidate_recall": candidate_recall,
        "candidate_pair_precision": precision,
        "candidate_pair_recall_given_candidate": conditional_recall,
        "candidate_pair_accuracy": pair_accuracy,
        "threshold": threshold,
        "validation_anchors": len(valid_anchors),
    }
    print(json.dumps(report, indent=2), flush=True)

    # Refit on all selected anchors after threshold selection; validation labels are never features.
    _, all_x, all_y, _ = make_pair_data(anchors, labels, index, target_columns)
    if len(np.unique(all_y)) < 2:
        raise ValueError("Full selected candidate set does not contain both classes.")
    final_model = make_classifier(lgbm_params)
    final_model.fit(all_x, all_y)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "version": MODEL_VERSION,
        "classifier": final_model,
        "threshold": threshold,
        "feature_names": FEATURE_NAMES,
        "metrics": report,
    }, args.model_path)
    print(f"Saved model to {args.model_path}", flush=True)


def predict(args: argparse.Namespace) -> None:
    artifact = joblib.load(args.model_path)
    if artifact.get("version") != MODEL_VERSION or artifact.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Model artifact is incompatible with this pipeline version.")
    test_s1 = read_tsv(args.data_dir / "test_source1_clean.tsv", [
        "entity_id", "business_name", "business_address", "country",
    ], args.max_anchors)
    targets = pd.concat([
        read_tsv(args.data_dir / f"test_{source}_clean.tsv", [
            "entity_id", "business_name", "business_address", "country",
        ], args.max_targets_per_source)
        for source in ("source2", "source3")
    ], ignore_index=True)
    index, target_columns = build_candidate_index(targets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate_pairs.tsv"
    matching_path = args.output_dir / "matching_results.tsv"
    classifier = artifact["classifier"]
    threshold = float(artifact["threshold"])
    candidate_count = matched_count = 0
    with candidate_path.open("w", encoding="utf-8", newline="") as candidates_file, matching_path.open(
        "w", encoding="utf-8", newline=""
    ) as matching_file:
        candidates_file.write("source1_entity_id\tcandidate_entity_ids\n")
        matching_file.write("source1_entity_id\tmatched_entity_ids\n")
        for anchor in test_s1[["entity_id", "business_name", "business_address", "country"]].itertuples(
            index=False, name=None
        ):
            candidate_rows = generate_candidates(anchor, index, target_columns)
            candidate_ids = [str(target_columns[3][row_id]) for row_id in candidate_rows]
            if candidate_rows:
                features = np.asarray([
                    comparison_features(
                        anchor[1], anchor[2], anchor[3],
                        str(target_columns[0][row_id]), str(target_columns[1][row_id]),
                        str(target_columns[2][row_id]),
                    )
                    for row_id in candidate_rows
                ], dtype=np.float32)
                probabilities = classifier.predict_proba(features)[:, 1]
                matches = [candidate_ids[i] for i, score in enumerate(probabilities) if score >= threshold]
            else:
                matches = []
            candidates_file.write(f"{anchor[0]}\t{','.join(candidate_ids)}\n")
            matching_file.write(f"{anchor[0]}\t{','.join(matches)}\n")
            candidate_count += len(candidate_ids)
            matched_count += len(matches)
    print(f"Wrote {len(test_s1):,} anchors; candidates={candidate_count:,}; matches={matched_count:,}.")
    print(f"Candidate file: {candidate_path}")
    print(f"Matching file: {matching_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("evaluate", "predict"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_OUTPUT / "entity_matcher.joblib")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-anchors", type=int, default=0,
                        help="Optional reproducible random anchor sample for evaluation; 0 uses all anchors.")
    parser.add_argument("--max-targets-per-source", type=int, default=0,
                        help="Optional first-N target cap per source for smoke tests; 0 uses all.")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--tune", action="store_true", help="Enable Optuna hyperparameter optimization.")
    return parser.parse_args()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.validation_fraction <= 0 or args.validation_fraction >= 0.5:
        raise ValueError("--validation-fraction must be greater than 0 and less than 0.5")
    if args.mode == "evaluate":
        train_and_evaluate(args)
    else:
        predict(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())