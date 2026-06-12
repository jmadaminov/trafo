import numpy as np

from trafo.gaze import features
from trafo.gaze.landmarks import FaceResult


def synthetic_face(yaw_deg: float = 0.0) -> FaceResult:
    lm = np.zeros((478, 3), dtype=np.float32)
    # Eye corners / lids / irises in plausible normalized positions.
    lm[33] = (0.40, 0.45, 0.0)   # right outer
    lm[133] = (0.46, 0.45, 0.0)  # right inner
    lm[263] = (0.60, 0.45, 0.0)  # left outer
    lm[362] = (0.54, 0.45, 0.0)  # left inner
    lm[159], lm[145] = (0.43, 0.43, 0.0), (0.43, 0.47, 0.0)  # right lids
    lm[386], lm[374] = (0.57, 0.43, 0.0), (0.57, 0.47, 0.0)  # left lids
    lm[468] = (0.43, 0.45, 0.0)  # right iris
    lm[473] = (0.57, 0.45, 0.0)  # left iris
    lm[1] = (0.50, 0.55, -0.02)  # nose tip
    lm[234] = (0.36, 0.50, 0.01 + 0.002 * yaw_deg)  # right oval (tragion)
    lm[454] = (0.64, 0.50, 0.01 - 0.002 * yaw_deg)  # left oval

    a = np.radians(yaw_deg)
    transform = np.eye(4)
    transform[:3, :3] = [
        [np.cos(a), 0, np.sin(a)],
        [0, 1, 0],
        [-np.sin(a), 0, np.cos(a)],
    ]
    transform[:3, 3] = (1.5, -2.0, -45.0)
    return FaceResult(landmarks=lm, transform=transform)


def test_extract_matches_feature_names():
    f = features.extract(synthetic_face())
    assert f is not None
    assert f.shape == (len(features.FEATURE_NAMES),)
    assert np.isfinite(f).all()
    names = dict(zip(features.FEATURE_NAMES, f))
    assert abs(names["head_tx"] - 1.5) < 1e-5
    assert abs(names["head_tz"] - (-45.0)) < 1e-3
    assert abs(names["oval_ratio"] - 1.0) < 0.05  # symmetric face, frontal


def test_asymmetry_features_track_yaw():
    frontal = features.extract(synthetic_face(0.0))
    turned = features.extract(synthetic_face(20.0))
    names_f = dict(zip(features.FEATURE_NAMES, frontal))
    names_t = dict(zip(features.FEATURE_NAMES, turned))
    assert abs(names_t["yaw"] - 20.0) < 1.0
    assert names_t["cheek_dz"] != names_f["cheek_dz"]  # asymmetry responds


def test_extract_none_without_transform():
    face = synthetic_face()
    face.transform = None
    assert features.extract(face) is None
