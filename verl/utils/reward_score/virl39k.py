# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Reward scoring for the TIGER-Lab/ViRL39K dataset.

Every ViRL39K question is a single-answer, rule-verifiable math/reasoning
problem. Ground-truth strings are (mostly) wrapped as ``\\boxed{...}``.
Model outputs are expected to follow the same ``<think>…</think>…\\boxed{…}``
format as VLAA-Thinking.

Scoring components:
  * format reward — 1.0 iff the output contains a <think>…</think>
    block followed by \\boxed{…}. Same as VLAA.
  * accuracy reward — extract the predicted answer from \\boxed{…} and
    compare against the ground-truth answer via (in order):
      1. math_verify (preferred when the library is available) —
         symbolic/LaTeX equivalence
      2. mathruler grade_answer — numeric / simple-expression grading
      3. normalised string equality — strip whitespace, punctuation,
         lowercase, final fallback for free-text answers.

Final reward = (1 - format_score) * acc + format_score * fmt, with
format_score default 0.1 matching the VLAA convention.
"""
from __future__ import annotations

import re
import string

from mathruler.grader import extract_boxed_content, grade_answer

try:
    from math_verify import parse as _mv_parse, verify as _mv_verify   # type: ignore
    _HAS_MATH_VERIFY = True
except Exception:  # pragma: no cover — optional dependency
    _HAS_MATH_VERIFY = False


# ── helpers ──────────────────────────────────────────────────────────────────

_FORMAT_RE = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)


def _format_reward(predict_str: str) -> float:
    return 1.0 if _FORMAT_RE.fullmatch(predict_str) else 0.0


def _extract_pred(predict_str: str) -> str:
    return extract_boxed_content(predict_str)


def _strip_boxed(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{"):-1].strip()
    return s


_PUNCT_TABLE = str.maketrans("", "", string.punctuation + " \t\n\r")


def _norm_eq(a: str, b: str) -> bool:
    return a.lower().translate(_PUNCT_TABLE) == b.lower().translate(_PUNCT_TABLE)


def _math_verify_eq(pred: str, gt: str) -> bool:
    if not _HAS_MATH_VERIFY:
        return False
    try:
        return bool(_mv_verify(_mv_parse(gt), _mv_parse(pred)))
    except Exception:
        return False


# ── public API ───────────────────────────────────────────────────────────────


def compute_score(
    predict_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    format_score: float = 0.1,
) -> float:
    fmt = _format_reward(predict_str)

    pred = _extract_pred(predict_str)
    if not pred:
        return format_score * fmt

    gt = _strip_boxed(ground_truth)

    if _math_verify_eq(pred, gt):
        acc = 1.0
    elif grade_answer(pred, gt):
        acc = 1.0
    elif _norm_eq(pred, gt):
        acc = 1.0
    else:
        acc = 0.0

    return (1.0 - format_score) * acc + format_score * fmt
