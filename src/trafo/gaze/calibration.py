"""Mapping from gaze features to a point on the (multi-display) virtual desktop.

Ridge regression on an expanded feature vector: the 15 base features plus
pairwise products of the pose/iris terms. Targets are global virtual-desktop
coordinates, so a calibration that visited several displays maps gaze onto all
of them with a single model.

Two deliberate choices keep the mapping stable across posture changes:

- Features are scaled by FIXED physical constants, never by the calibration
  data's own std. The user holds their head still while calibrating, so head
  pose has near-zero variance there; dividing by that tiny std would blow the
  features up and turn every later posture shift into a large screen offset.
- Per-feature ridge penalties make iris movement cheap and head/face position
  expensive: the model is forced to explain gaze with the eyes first, and may
  use head pose only as a mild corrector. Eyes are reliable across sessions;
  face position is not.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..config import CALIBRATION_PATH, ensure_config_dir
from .features import FEATURE_NAMES

# Indices (into FEATURE_NAMES) whose pairwise products enter the expansion:
# pitch, yaw, and the four iris offsets — the terms gaze actually depends on
# nonlinearly. Face position/size stay linear (translation compensation).
_PAIR_IDX = [0, 1, 3, 4, 5, 6]

# Typical magnitudes per feature: degrees for pose, normalized units for the
# rest, cm-ish for head translation. Fixed so that posture-induced variance
# can never be amplified.
_FIXED_SCALE = np.array([
    10.0, 10.0, 10.0,            # pitch, yaw, roll (deg)
    0.04, 0.04, 0.04, 0.04,      # iris offsets
    0.08, 0.08, 0.03,            # face_cx, face_cy, eye_dist
    5.0, 5.0, 5.0,               # head_tx, head_ty, head_tz (cm-ish)
    0.05, 0.3,                   # cheek_dz, oval_ratio
])

# Relative ridge penalty per base feature (same order as FEATURE_NAMES):
# iris offsets cheap (1), pitch/yaw moderate (head turns toward a far monitor
# are real signal), roll / face position / eye distance expensive. Where head
# pose and iris are collinear (they always are in calibration data), ridge
# shifts the credit onto the cheap iris terms — exactly the "trust eyes over
# face" behavior we want. The head-state tail (translation + asymmetry) exists
# for the screen classifier; the within-screen regression must not lean on it.
_BASE_PENALTY = np.array([
    15.0, 15.0, 75.0,
    1.0, 1.0, 1.0, 1.0,
    75.0, 75.0, 75.0,
    75.0, 75.0, 75.0,
    75.0, 75.0,
])


def _expand(xs: np.ndarray) -> np.ndarray:
    """(n, 15) scaled features -> (n, 37) design matrix with bias."""
    n = xs.shape[0]
    pairs = [xs[:, i] * xs[:, j] for k, i in enumerate(_PAIR_IDX) for j in _PAIR_IDX[k:]]
    return np.column_stack([np.ones(n), xs, *pairs])


def _penalties() -> np.ndarray:
    """Per-column ridge penalty for the expanded design matrix (bias unpenalized)."""
    pair_pen = [
        np.sqrt(_BASE_PENALTY[i] * _BASE_PENALTY[j])
        for k, i in enumerate(_PAIR_IDX)
        for j in _PAIR_IDX[k:]
    ]
    return np.concatenate([[0.0], _BASE_PENALTY, pair_pen])


class GazeMapper:
    def __init__(self, alpha: float = 0.02):
        self.alpha = alpha
        self.mean: np.ndarray | None = None
        self.weights: np.ndarray | None = None  # (37, 2)

    @property
    def is_fitted(self) -> bool:
        return self.weights is not None

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """Fit on (n, 15) features and (n, 2) global screen targets; returns RMS px error."""
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        self.mean = x.mean(axis=0)

        phi = _expand((x - self.mean) / _FIXED_SCALE)
        yw = y
        n_eff = len(x)
        if sample_weight is not None:
            sw = np.sqrt(np.asarray(sample_weight, dtype=float))[:, None]
            phi = phi * sw
            yw = y * sw
            n_eff = float(np.sum(sample_weight))
        # Scale the penalty with the (effective) sample count so regularization
        # strength is independent of how many samples were collected.
        reg = self.alpha * n_eff * np.diag(_penalties())
        self.weights = np.linalg.solve(phi.T @ phi + reg, phi.T @ yw)
        return self.rms(x, y)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """(15,) feature vector -> (2,) global screen point."""
        if not self.is_fitted:
            raise RuntimeError("GazeMapper is not fitted")
        xs = (np.atleast_2d(features) - self.mean) / _FIXED_SCALE
        out = _expand(xs) @ self.weights
        return out[0]

    def rms(self, features: np.ndarray, targets: np.ndarray) -> float:
        xs = (np.asarray(features, dtype=float) - self.mean) / _FIXED_SCALE
        err = _expand(xs) @ self.weights - targets
        return float(np.sqrt((err**2).sum(axis=1).mean()))

    # -- persistence ---------------------------------------------------------

    def save(self, path: Path = CALIBRATION_PATH) -> None:
        ensure_config_dir()
        path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "feature_names": FEATURE_NAMES,
                    "alpha": self.alpha,
                    "mean": self.mean.tolist(),
                    "weights": self.weights.tolist(),
                }
            )
        )

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> "GazeMapper | None":
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        # Reject calibrations from older model layouts; recalibration required.
        if data.get("version") != 2 or data.get("feature_names") != FEATURE_NAMES:
            return None
        mapper = cls(alpha=data["alpha"])
        mapper.mean = np.array(data["mean"])
        mapper.weights = np.array(data["weights"])
        return mapper

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "mean": self.mean.tolist(), "weights": self.weights.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "GazeMapper":
        mapper = cls(alpha=data["alpha"])
        mapper.mean = np.array(data["mean"])
        mapper.weights = np.array(data["weights"])
        return mapper


def _reject_outliers(features: np.ndarray, keep_mad: float = 3.5) -> np.ndarray:
    """Boolean mask keeping samples near the per-point median feature vector.

    Calibration samples are collected while the eye may still be saccading
    toward the dot (or half-blinking); those land far from the point's median
    in feature space and would drag the fit.
    """
    xs = features / _FIXED_SCALE
    med = np.median(xs, axis=0)
    dist = np.linalg.norm(xs - med, axis=1)
    mad = np.median(np.abs(dist - np.median(dist))) + 1e-9
    return dist <= np.median(dist) + keep_mad * mad


class _SoftmaxClassifier:
    """L2-regularized multinomial logistic regression for screen classification.

    Tiny and dependency-free: full-batch gradient descent on a few thousand
    scaled samples runs in milliseconds and is deterministic (zero init).
    Unlike the nearest-centroid fallback it learns which features actually
    separate the screens — head yaw/position for head-turn switches, iris
    terms for eyes-only switches — instead of weighting everything equally.
    """

    def __init__(self, l2: float = 1e-2, iters: int = 300, lr: float = 0.5):
        self.l2, self.iters, self.lr = l2, iters, lr
        self.mean: np.ndarray | None = None
        self.w: np.ndarray | None = None  # (n_classes, n_features + 1)

    @property
    def is_fitted(self) -> bool:
        return self.w is not None

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def _design(self, features: np.ndarray) -> np.ndarray:
        xs = (np.atleast_2d(features) - self.mean) / _FIXED_SCALE
        return np.column_stack([np.ones(len(xs)), xs])

    def fit(self, features: np.ndarray, labels: np.ndarray, n_classes: int) -> None:
        x = np.asarray(features, dtype=float)
        self.mean = x.mean(axis=0)
        phi = self._design(x)
        y = np.eye(n_classes)[np.asarray(labels, dtype=int)]
        w = np.zeros((n_classes, phi.shape[1]))
        for _ in range(self.iters):
            p = self._softmax(phi @ w.T)
            grad = (p - y).T @ phi / len(phi) + self.l2 * w
            grad[:, 0] -= self.l2 * w[:, 0]  # bias unpenalized
            w -= self.lr * grad
        self.w = w

    def proba(self, features: np.ndarray) -> np.ndarray:
        return self._softmax(self._design(features) @ self.w.T)[0]

    def to_dict(self) -> dict:
        return {"l2": self.l2, "mean": self.mean.tolist(), "w": self.w.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "_SoftmaxClassifier":
        clf = cls(l2=data.get("l2", 1e-2))
        clf.mean = np.array(data["mean"])
        clf.w = np.array(data["w"])
        return clf


class ScreenLockedMapper:
    """Two-stage gaze mapping: classify the target screen, then map within it.

    Stage 1 ("which screen"): a learned softmax classifier over the scaled
    feature vector (nearest-centroid as fallback), with confidence-aware
    hysteresis — the lock moves after `lock_n` consecutive confident samples
    prefer another screen, or after FAST_N very confident ones (an obvious
    head turn switches snappily; ambiguity near shared edges stays sticky).
    Stage 2 ("where on it"): a per-screen GazeMapper trained only on that
    screen's calibration samples — a narrow head-pose regime, so the local
    model is driven almost entirely by iris movement. Predictions are clamped
    to the locked screen's rect.
    """

    CLICK_BUFFER = 200  # rolling click samples kept per screen
    CLICK_WEIGHT = 3.0  # fresh ground truth counts more than old calibration
    REFIT_EVERY = 5  # refit a screen's mapper after this many new clicks
    CONF_SWITCH = 0.6  # min average confidence for a lock_n challenge to win
    CONF_FAST = 0.9  # confidence of an "obvious" switch
    FAST_N = 3  # consecutive obvious frames that switch immediately

    def __init__(self, lock_n: int = 8):
        self.lock_n = lock_n
        self.rects: list[tuple[float, float, float, float]] = []
        self.centroids: list[np.ndarray] = []
        self.mappers: list[GazeMapper] = []
        self.classifier: _SoftmaxClassifier | None = None
        # Training data is kept so click samples can be folded in later:
        # the calibration set anchors the model, clicks correct drift.
        self._cal_x: list[np.ndarray] = []
        self._cal_y: list[np.ndarray] = []
        self._click_x: list[list[np.ndarray]] = []
        self._click_y: list[list[np.ndarray]] = []
        self._clicks_since_refit: list[int] = []
        self._locked: int | None = None
        self._challenger: int | None = None
        self._challenger_count = 0
        self._conf_sum = 0.0
        self._fast_count = 0

    @property
    def is_fitted(self) -> bool:
        return bool(self.mappers)

    @property
    def locked_screen(self) -> int | None:
        return self._locked

    def fit(
        self,
        samples: list[tuple[np.ndarray, np.ndarray, int]],
        screen_rects: list[tuple[float, float, float, float]],
    ) -> dict:
        """samples: (features, global target, screen index). Returns per-screen stats."""
        self.rects = [tuple(r) for r in screen_rects]
        self.centroids, self.mappers = [], []
        self._cal_x, self._cal_y = [], []
        self._click_x = [[] for _ in screen_rects]
        self._click_y = [[] for _ in screen_rects]
        self._clicks_since_refit = [0] * len(screen_rects)
        stats = {}
        for si in range(len(screen_rects)):
            group = [(f, t) for f, t, s in samples if s == si]
            x = np.stack([f for f, _ in group])
            y = np.stack([t for _, t in group])

            # Per-point outlier rejection (points identified by their target).
            # A no-op for smooth-pursuit data, where every target is unique.
            keep = np.zeros(len(x), dtype=bool)
            for target in np.unique(y, axis=0):
                idx = np.where((y == target).all(axis=1))[0]
                keep[idx] = _reject_outliers(x[idx])
            dropped = int((~keep).sum())
            x, y = x[keep], y[keep]

            mapper = GazeMapper()
            mapper.fit(x, y)
            # Residual-trimmed refit: catch-up saccades during pursuit (and any
            # bad frames the per-point gate cannot see) are transient samples
            # the model cannot explain — drop them and fit again.
            xs = (x - mapper.mean) / _FIXED_SCALE
            res = np.linalg.norm(_expand(xs) @ mapper.weights - y, axis=1)
            mad = np.median(np.abs(res - np.median(res))) + 1e-9
            keep2 = res <= np.median(res) + 3.5 * mad
            dropped += int((~keep2).sum())
            x, y = x[keep2], y[keep2]
            rms = mapper.fit(x, y)

            self.centroids.append(x.mean(axis=0))
            self.mappers.append(mapper)
            self._cal_x.append(x)
            self._cal_y.append(y)
            stats[si] = {"rms": rms, "kept": int(len(x)), "dropped": dropped}
        self._fit_classifier()
        return stats

    def _fit_classifier(self) -> None:
        """(Re)train the screen classifier on calibration + click samples."""
        if len(self.rects) < 2:
            self.classifier = None  # single screen: nothing to classify
            return
        xs, labels = [], []
        for si in range(len(self.rects)):
            group = [self._cal_x[si]]
            if self._click_x[si]:
                group.append(np.stack(self._click_x[si]))
            x = np.concatenate(group)
            xs.append(x)
            labels.append(np.full(len(x), si))
        try:
            clf = _SoftmaxClassifier()
            clf.fit(np.concatenate(xs), np.concatenate(labels), len(self.rects))
            self.classifier = clf
        except Exception:
            self.classifier = None  # degenerate data: centroid fallback

    # -- continuous recalibration from clicks --------------------------------

    @property
    def click_count(self) -> int:
        return sum(len(c) for c in self._click_x)

    def add_click_sample(self, features: np.ndarray, point: tuple[float, float]) -> bool:
        """Fold in a ground-truth (features -> click position) sample.

        The screen is identified from the click position. Returns True if the
        sample triggered a refit of that screen's mapper.
        """
        si = next(
            (i for i, r in enumerate(self.rects) if r[0] <= point[0] < r[0] + r[2]
             and r[1] <= point[1] < r[1] + r[3]),
            None,
        )
        if si is None:
            return False
        self._click_x[si].append(np.asarray(features, dtype=float))
        self._click_y[si].append(np.array(point, dtype=float))
        if len(self._click_x[si]) > self.CLICK_BUFFER:
            self._click_x[si].pop(0)
            self._click_y[si].pop(0)
        self._clicks_since_refit[si] += 1
        if self._clicks_since_refit[si] >= self.REFIT_EVERY:
            self._refit_screen(si)
            self._clicks_since_refit[si] = 0
            return True
        return False

    def _refit_screen(self, si: int) -> None:
        x = np.concatenate([self._cal_x[si], np.stack(self._click_x[si])])
        y = np.concatenate([self._cal_y[si], np.stack(self._click_y[si])])
        w = np.concatenate(
            [np.ones(len(self._cal_x[si])), np.full(len(self._click_x[si]), self.CLICK_WEIGHT)]
        )
        self.mappers[si].fit(x, y, sample_weight=w)
        # Clicks are screen-labeled ground truth, so they also keep the screen
        # classifier tracking posture drift. Centroids stay calibration-defined
        # (they are only the fallback).
        self._fit_classifier()

    def classify_proba(self, features: np.ndarray) -> np.ndarray:
        """Per-screen probabilities (learned classifier, centroid fallback)."""
        if self.classifier is not None and self.classifier.is_fitted:
            return self.classifier.proba(features)
        scaled = np.asarray(features, dtype=float) / _FIXED_SCALE
        dists = np.array(
            [np.linalg.norm(scaled - c / _FIXED_SCALE) for c in self.centroids]
        )
        return _SoftmaxClassifier._softmax(-dists[None, :])[0]

    def classify(self, features: np.ndarray) -> int:
        return int(np.argmax(self.classify_proba(features)))

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Global screen point, clamped to the currently locked screen."""
        if not self.is_fitted:
            raise RuntimeError("ScreenLockedMapper is not fitted")
        proba = self.classify_proba(features)
        best = int(np.argmax(proba))
        conf = float(proba[best])

        if self._locked is None:
            self._locked = best
        elif best != self._locked:
            if best == self._challenger:
                self._challenger_count += 1
                self._conf_sum += conf
                self._fast_count = self._fast_count + 1 if conf >= self.CONF_FAST else 0
            else:
                self._challenger, self._challenger_count = best, 1
                self._conf_sum = conf
                self._fast_count = 1 if conf >= self.CONF_FAST else 0
            confident_challenge = (
                self._challenger_count >= self.lock_n
                and self._conf_sum / self._challenger_count >= self.CONF_SWITCH
            )
            if self._fast_count >= self.FAST_N or confident_challenge:
                self._locked = best
                self._reset_challenge()
        else:
            self._reset_challenge()

        point = self.mappers[self._locked].predict(features)
        rx, ry, rw, rh = self.rects[self._locked]
        return np.array(
            [min(max(point[0], rx), rx + rw - 1), min(max(point[1], ry), ry + rh - 1)]
        )

    def _reset_challenge(self) -> None:
        self._challenger, self._challenger_count = None, 0
        self._conf_sum, self._fast_count = 0.0, 0

    def reset_lock(self) -> None:
        self._locked = None
        self._reset_challenge()

    # -- persistence ---------------------------------------------------------

    def save(self, path: Path = CALIBRATION_PATH) -> None:
        ensure_config_dir()
        screens = []
        for si in range(len(self.rects)):
            screens.append(
                {
                    "rect": list(self.rects[si]),
                    "centroid": self.centroids[si].tolist(),
                    "mapper": self.mappers[si].to_dict(),
                    "cal_x": np.round(self._cal_x[si], 5).tolist(),
                    "cal_y": self._cal_y[si].tolist(),
                    "click_x": [np.round(f, 5).tolist() for f in self._click_x[si]],
                    "click_y": [p.tolist() for p in self._click_y[si]],
                }
            )
        path.write_text(
            json.dumps(
                {
                    "version": 5,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "feature_names": FEATURE_NAMES,
                    "lock_n": self.lock_n,
                    "classifier": self.classifier.to_dict() if self.classifier else None,
                    "screens": screens,
                }
            )
        )

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> "ScreenLockedMapper | None":
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if data.get("version") != 5 or data.get("feature_names") != FEATURE_NAMES:
            return None
        mapper = cls(lock_n=data.get("lock_n", 8))
        if data.get("classifier"):
            mapper.classifier = _SoftmaxClassifier.from_dict(data["classifier"])
        for s in data["screens"]:
            mapper.rects.append(tuple(s["rect"]))
            mapper.centroids.append(np.array(s["centroid"]))
            mapper.mappers.append(GazeMapper.from_dict(s["mapper"]))
            mapper._cal_x.append(np.array(s["cal_x"]))
            mapper._cal_y.append(np.array(s["cal_y"]))
            mapper._click_x.append([np.array(f) for f in s["click_x"]])
            mapper._click_y.append([np.array(p) for p in s["click_y"]])
            mapper._clicks_since_refit.append(0)
        return mapper
