from pathlib import Path

import dlib
import numpy as np
from sklearn.svm import SVC
import streamlit as st

try:
    import face_recognition_models
except ModuleNotFoundError:
    face_recognition_models = None

from src.database.db import get_all_students


def _resolve_model_path(model_name):
    candidates = []

    if face_recognition_models is not None:
        if model_name == "pose":
            candidates.append(face_recognition_models.pose_predictor_model_location())
        else:
            candidates.append(face_recognition_models.face_recognition_model_location())

    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models"

    if model_name == "pose":
        candidates.extend([
            model_dir / "shape_predictor_5_face_landmarks.dat",
            model_dir / "shape_predictor_68_face_landmarks.dat",
            Path("/app/models/shape_predictor_5_face_landmarks.dat"),
        ])
    else:
        candidates.extend([
            model_dir / "dlib_face_recognition_resnet_model_v1.dat",
            Path("/app/models/dlib_face_recognition_resnet_model_v1.dat"),
        ])

    for candidate in candidates:
        if candidate and Path(str(candidate)).exists():
            return str(candidate)

    return None


@st.cache_resource
def load_dlib_model():
    pose_model = _resolve_model_path("pose")
    face_model = _resolve_model_path("face")

    if pose_model is None or face_model is None:
        st.warning(
            "Face recognition models are missing. Add the dlib model files under the project models/ directory or install them in the deployment environment."
        )
        return None, None, None

    detector = dlib.get_frontal_face_detector()

    sp = dlib.shape_predictor(pose_model)
    facerec = dlib.face_recognition_model_v1(face_model)

    return detector, sp, facerec


def get_face_embedding(image_np):
    detector, sp, facerec = load_dlib_model()

    if detector is None or sp is None or facerec is None:
        return []

    faces = detector(image_np, 1)

    encodings = []

    for face in faces:
        shape = sp(image_np, face)

        face_descriptor = facerec.compute_face_descriptor(
            image_np,
            shape,
            1
        )

        encoding = np.array(
            face_descriptor,
            dtype=np.float64
        )

        encodings.append(encoding)

    return encodings


@st.cache_resource
def get_trained_model():
    x = []
    y = []

    student_db = get_all_students()

    if not student_db:
        return None

    for student in student_db:
        embedding = student.get("face_embedding")
        student_id = student.get("student_id")

        if embedding is None or student_id is None:
            continue

        try:
            embedding_array = np.array(
                embedding,
                dtype=np.float64
            )

            if embedding_array.shape[0] != 128:
                continue

            x.append(embedding_array)
            y.append(int(student_id))

        except (ValueError, TypeError):
            continue

    if len(x) == 0:
        return None

    x = np.array(x)
    y = np.array(y)

    unique_students = np.unique(y)

    clf = None

    if len(unique_students) >= 2:
        clf = SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced"
        )

        try:
            clf.fit(x, y)
        except ValueError:
            return None

    return {
        "clf": clf,
        "x": x,
        "y": y
    }


def train_classifier():
    st.cache_resource.clear()

    model_data = get_trained_model()

    return model_data is not None


def _best_match_for_encoding(encoding, x_train, y_train):
    nearest_id = None
    nearest_distance = None
    second_distance = None

    for student_id in sorted(set(y_train.tolist())):
        student_indices = np.where(y_train == student_id)[0]

        for index in student_indices:
            stored_embedding = x_train[index]
            distance = float(np.linalg.norm(stored_embedding - encoding))

            if nearest_distance is None or distance < nearest_distance:
                second_distance = nearest_distance
                nearest_distance = distance
                nearest_id = int(student_id)
            elif second_distance is None or distance < second_distance:
                second_distance = distance

    return nearest_id, nearest_distance, second_distance


def predict_attendance(class_image_np):
    encodings = get_face_embedding(class_image_np)

    detected_students = {}

    if not encodings:
        return detected_students, [], 0

    model_data = get_trained_model()

    if not model_data:
        return detected_students, [], len(encodings)

    x_train = model_data["x"]
    y_train = model_data["y"]

    all_students = sorted(
        list(set(y_train))
    )

    if len(all_students) == 0:
        return detected_students, [], len(encodings)

    for encoding in encodings:
        nearest_id, nearest_distance, second_distance = _best_match_for_encoding(
            encoding,
            x_train,
            y_train,
        )

        resemblance_threshold = 0.6
        confidence_margin = 0.08

        if nearest_id is not None and nearest_distance is not None:
            if nearest_distance <= resemblance_threshold:
                if second_distance is None or (second_distance - nearest_distance) >= confidence_margin:
                    detected_students[nearest_id] = True

    return detected_students, all_students, len(encodings)