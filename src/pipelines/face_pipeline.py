import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st

from src.database.db import get_all_students


@st.cache_resource
def load_dlib_model():
    detector = dlib.get_frontal_face_detector()

    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec


def get_face_embedding(image_np):
    detector, sp, facerec = load_dlib_model()

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


def predict_attendance(class_image_np):
    encodings = get_face_embedding(class_image_np)

    detected_students = {}

    model_data = get_trained_model()

    if not model_data:
        return detected_students, [], len(encodings)

    clf = model_data["clf"]
    x_train = model_data["x"]
    y_train = model_data["y"]

    all_students = sorted(
        list(set(y_train))
    )

    if len(all_students) == 0:
        return detected_students, [], len(encodings)

    for encoding in encodings:

        if len(all_students) >= 2:
            predicted_id = int(
                clf.predict([encoding])[0]
            )
        else:
            predicted_id = int(
                all_students[0]
            )

        student_indices = np.where(
            y_train == predicted_id
        )[0]

        if len(student_indices) == 0:
            continue

        distances = []

        for index in student_indices:
            stored_embedding = x_train[index]

            distance = np.linalg.norm(
                stored_embedding - encoding
            )

            distances.append(distance)

        best_match_score = min(distances)

        resemblance_threshold = 0.6

        if best_match_score <= resemblance_threshold:
            detected_students[predicted_id] = True

    return detected_students, all_students, len(encodings)