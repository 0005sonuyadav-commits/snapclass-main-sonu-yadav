import numpy as np

import src.pipelines.face_pipeline as face_pipeline


def test_predict_attendance_rejects_similar_but_wrong_face(monkeypatch):
    known_face_a = np.full(128, 0.40, dtype=np.float64)
    known_face_b = np.full(128, 0.45, dtype=np.float64)
    impostor = np.full(128, 0.52, dtype=np.float64)

    monkeypatch.setattr(face_pipeline, "get_face_embedding", lambda _img: [impostor])
    monkeypatch.setattr(
        face_pipeline,
        "get_trained_model",
        lambda: {
            "x": np.array([known_face_a, known_face_b], dtype=np.float64),
            "y": np.array([10, 20], dtype=np.int64),
            "clf": None,
        },
    )

    detected, _, _ = face_pipeline.predict_attendance(np.zeros((100, 100, 3), dtype=np.uint8))

    assert detected == {}
