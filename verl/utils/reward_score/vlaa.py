# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reward scoring for the UCSC-VLAA/VLAA-Thinking dataset.

The dataset contains five verifier_type values, each requiring different scoring:

  math        – Mathematical expression. GT may be raw or wrapped in \\[\\boxed{...}\\].
                Extract \\boxed{} from model output and use mathruler.grade_answer.
  digit       – Single integer. Same pipeline as math.
  mcq         – Single letter choice (A/B/C/D). Same pipeline as math.
  open_ended  – Free-form text / document QA. Extract \\boxed{} from model output,
                then do case-insensitive substring containment check against GT.
  iou         – Grounding; GT is a normalised [x1,y1,x2,y2] bbox string.
                Extract bbox from \\boxed{} of model output and compute IoU.
                Reward = IoU score (continuous 0-1) so the model learns to
                tighten its predicted box.

All types share the same *format* reward: the output must contain
<think>...</think> followed by \\boxed{...}.
"""

import ast
import re

from mathruler.grader import extract_boxed_content, grade_answer

# ── helpers ──────────────────────────────────────────────────────────────────


def _format_reward(predict_str: str) -> float:
    """Return 1.0 if the output follows <think>…</think>…\\boxed{…} format."""
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, predict_str) else 0.0


def _extract_pred(predict_str: str) -> str:
    """Extract the content of the first \\boxed{} in the prediction."""
    return extract_boxed_content(predict_str)


# ── per-type accuracy rewards ─────────────────────────────────────────────────


def _math_acc(predict_str: str, ground_truth: str) -> float:
    """math / digit / mcq: grade extracted prediction against GT."""
    answer = _extract_pred(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def _open_ended_acc(predict_str: str, ground_truth: str) -> float:
    """Open-ended QA: case-insensitive substring containment check.

    We extract from \\boxed{} first (model is instructed to do so).
    Fall back to the full prediction string if no \\boxed{} found.
    """
    extracted = _extract_pred(predict_str).strip()
    candidate = extracted if extracted else predict_str.strip()

    # Normalise whitespace and lowercase both sides
    candidate_norm = " ".join(candidate.lower().split())
    gt_norm = " ".join(ground_truth.lower().split())

    # Accept if GT is contained in prediction or prediction is contained in GT
    if gt_norm in candidate_norm or candidate_norm in gt_norm:
        return 1.0

    # Fallback: try grade_answer (handles numeric / short answers embedded in text)
    return 1.0 if grade_answer(candidate, ground_truth) else 0.0


def _parse_bbox(s: str) -> list[float] | None:
    """Parse a bounding-box string such as '[0.62, 0.57, 0.74, 0.94]'."""
    s = s.strip()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)) and len(val) == 4:
            return [float(v) for v in val]
    except Exception:
        pass
    # Try regex fallback for formats like "0.62, 0.57, 0.74, 0.94"
    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    if len(nums) == 4:
        return [float(n) for n in nums]
    return None


def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes (values in [0,1])."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iou_acc(predict_str: str, ground_truth: str) -> float:
    """IoU-based reward: continuous score in [0, 1]."""
    extracted = _extract_pred(predict_str).strip()
    if not extracted:
        return 0.0
    pred_box = _parse_bbox(extracted)
    gt_box = _parse_bbox(ground_truth)
    if pred_box is None or gt_box is None:
        return 0.0
    return _compute_iou(pred_box, gt_box)


# ── public API ────────────────────────────────────────────────────────────────


def compute_score(
    predict_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    format_score: float = 0.1,
) -> float:
    """Compute the VLAA reward score.

    Args:
        predict_str:   Model's generated text.
        ground_truth:  GT string from reward_model.ground_truth.
        extra_info:    Row's extra_info dict; must contain 'verifier_type'.
        format_score:  Weight assigned to the format component (default 0.1).

    Returns:
        Scalar reward in [0, 1].
    """
    verifier_type = (extra_info or {}).get("verifier_type", "math")
    fmt = _format_reward(predict_str)

    if verifier_type in ("math", "digit", "mcq"):
        acc = _math_acc(predict_str, ground_truth)
    elif verifier_type == "open_ended":
        acc = _open_ended_acc(predict_str, ground_truth)
    elif verifier_type == "iou":
        # For IoU, format reward still checks <think>/\boxed{} structure
        acc = _iou_acc(predict_str, ground_truth)
    else:
        # Unknown type: fall back to math grader
        acc = _math_acc(predict_str, ground_truth)

    return (1.0 - format_score) * acc + format_score * fmt
