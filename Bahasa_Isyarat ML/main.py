"""Train and run a real-time hand-sign classifier."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence
from urllib.error import URLError
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "Datafull train.csv"
DEFAULT_MODEL = BASE_DIR / "hand_sign_model.joblib"
HAND_LANDMARKER_MODEL = BASE_DIR / "hand_landmarker.task"
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FORMAT_VERSION = 1
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    f"{landmark}{axis}"
    for landmark in range(21)
    for axis in ("x", "y")
]


class HandSignError(Exception):
    """An expected application error that can be shown directly to the user."""


def normalize_landmarks(values: Sequence[float]) -> np.ndarray:
    """Normalize 21 x/y landmarks relative to the wrist and hand size."""
    coordinates = np.asarray(values, dtype=np.float64)
    if coordinates.shape != (len(FEATURE_COLUMNS),):
        raise ValueError(
            f"Fitur harus berjumlah {len(FEATURE_COLUMNS)}, "
            f"bukan {coordinates.size}."
        )
    if not np.isfinite(coordinates).all():
        raise ValueError("Fitur landmark mengandung NaN atau nilai tak hingga.")

    points = coordinates.reshape(21, 2)
    relative_points = points - points[0]
    scale = float(np.max(np.abs(relative_points)))
    if scale <= np.finfo(np.float64).eps:
        raise ValueError("Landmark tidak valid: semua titik bertumpuk.")

    return (relative_points / scale).reshape(-1)


def load_dataset(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series, int]:
    if not dataset_path.is_file():
        raise HandSignError(
            f"Dataset tidak ditemukan: {dataset_path}\n"
            "Pastikan file CSV tersedia atau gunakan --data untuk menentukan lokasinya."
        )

    try:
        dataframe = pd.read_csv(dataset_path, sep=";")
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise HandSignError(f"Gagal membaca dataset '{dataset_path}': {exc}") from exc

    if dataframe.columns.has_duplicates:
        raise HandSignError("Dataset memiliki nama kolom duplikat.")

    required_columns = [*FEATURE_COLUMNS, "char"]
    missing_columns = [name for name in required_columns if name not in dataframe.columns]
    if missing_columns:
        raise HandSignError(
            "Kolom dataset tidak lengkap: " + ", ".join(missing_columns)
        )

    features = dataframe.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    labels = dataframe["char"].astype("string").str.strip()
    valid_labels = labels.notna() & labels.ne("")
    finite_features = np.isfinite(features.to_numpy(dtype=np.float64)).all(axis=1)
    valid_rows = valid_labels.to_numpy() & finite_features
    removed_rows = int((~valid_rows).sum())

    features = features.loc[valid_rows].reset_index(drop=True)
    labels = labels.loc[valid_rows].reset_index(drop=True)
    if features.empty:
        raise HandSignError(
            "Tidak ada baris dataset valid setelah membersihkan label dan fitur."
        )

    normalized = np.vstack(
        [normalize_landmarks(row) for row in features.to_numpy(dtype=np.float64)]
    )
    normalized_features = pd.DataFrame(normalized, columns=FEATURE_COLUMNS)
    return normalized_features, labels, removed_rows


def create_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=150,
                    max_leaf_nodes=31,
                    learning_rate=0.1,
                    l2_regularization=1.0,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def save_model(model: Pipeline, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "model": model,
    }
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=model_path.parent,
            prefix=f".{model_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
        joblib.dump(payload, temporary_path)
        os.replace(temporary_path, model_path)
    except OSError as exc:
        raise HandSignError(f"Gagal menyimpan model ke '{model_path}': {exc}") from exc
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_model(model_path: Path) -> Pipeline:
    if not model_path.is_file():
        raise HandSignError(
            f"Model tidak ditemukan: {model_path}\n"
            "Jalankan 'python main.py train' terlebih dahulu."
        )
    try:
        payload = joblib.load(model_path)
    except (OSError, EOFError, ValueError, KeyError, ImportError) as exc:
        raise HandSignError(f"Gagal memuat model '{model_path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise HandSignError("Format file model tidak dikenal.")
    if payload.get("format_version") != MODEL_FORMAT_VERSION:
        raise HandSignError(
            "Versi format model tidak kompatibel. Latih ulang model dengan "
            "'python main.py train'."
        )
    if payload.get("feature_columns") != FEATURE_COLUMNS:
        raise HandSignError(
            "Urutan fitur pada model berbeda dari urutan fitur aplikasi. "
            "Latih ulang model sebelum menjalankan webcam."
        )

    model = payload.get("model")
    if not isinstance(model, Pipeline) or not hasattr(model, "predict_proba"):
        raise HandSignError("Isi model tidak valid atau tidak mendukung probabilitas.")
    if len(model.feature_names_in_) != len(FEATURE_COLUMNS):
        raise HandSignError(
            f"Model mengharapkan {len(model.feature_names_in_)} fitur, "
            f"seharusnya {len(FEATURE_COLUMNS)}."
        )
    if list(model.feature_names_in_) != FEATURE_COLUMNS:
        raise HandSignError("Nama atau urutan fitur model tidak sesuai.")
    return model


def ensure_hand_landmarker_model() -> Path:
    if HAND_LANDMARKER_MODEL.is_file():
        return HAND_LANDMARKER_MODEL

    temporary_path: str | None = None
    try:
        with urlopen(HAND_LANDMARKER_MODEL_URL, timeout=30) as response:
            with tempfile.NamedTemporaryFile(
                dir=BASE_DIR,
                prefix=".hand_landmarker.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = temporary_file.name
                shutil.copyfileobj(response, temporary_file)

        if os.path.getsize(temporary_path) < 1_000_000:
            raise HandSignError(
                "File model Hand Landmarker yang diunduh tidak valid. "
                "Hapus file sementara dan coba kembali."
            )
        os.replace(temporary_path, HAND_LANDMARKER_MODEL)
        return HAND_LANDMARKER_MODEL
    except (OSError, URLError) as exc:
        raise HandSignError(
            "Model Hand Landmarker tidak tersedia dan gagal diunduh. "
            f"Periksa koneksi internet, atau unduh file model dari "
            f"{HAND_LANDMARKER_MODEL_URL} dan simpan sebagai "
            f"'{HAND_LANDMARKER_MODEL}'. Detail: {exc}"
        ) from exc
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def draw_task_landmarks(
    frame: np.ndarray,
    landmarks: Sequence[object],
    connections: Sequence[object],
    cv2: object,
) -> None:
    height, width = frame.shape[:2]
    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in landmarks
    ]
    for connection in connections:
        start, end = connection.start, connection.end
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    for point in points:
        cv2.circle(frame, point, 3, (0, 0, 255), -1)


def train(args: argparse.Namespace) -> None:
    features, labels, removed_rows = load_dataset(args.data)
    class_counts = labels.value_counts()
    if len(class_counts) < 2:
        raise HandSignError("Training memerlukan sedikitnya dua kelas label.")
    if int(class_counts.min()) < 2:
        raise HandSignError(
            "Setiap kelas harus memiliki sedikitnya 2 baris agar pembagian "
            "training dan evaluasi dapat distratifikasi."
        )

    test_rows = math.ceil(len(labels) * args.test_size)
    if test_rows < len(class_counts) or len(labels) - test_rows < len(class_counts):
        raise HandSignError(
            "Data terlalu sedikit untuk test_size dan jumlah kelas yang tersedia."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    model = create_classifier()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(f"Baris valid: {len(labels):,}")
    print(f"Baris dibuang (label/fitur kosong atau tidak valid): {removed_rows:,}")
    print(f"Jumlah kelas: {len(class_counts)}")
    print(f"Akurasi holdout: {accuracy_score(y_test, predictions):.4f}")
    print(
        classification_report(
            y_test,
            predictions,
            labels=model.named_steps["classifier"].classes_,
            zero_division=0,
        )
    )

    save_model(model, args.model)
    print(f"Model tersimpan: {args.model}")


def run_webcam(args: argparse.Namespace) -> None:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise HandSignError(
            "Library webcam belum terpasang. Jalankan: "
            "python -m pip install -r requirements.txt"
        ) from exc

    model = load_model(args.model)
    legacy_api = getattr(mp, "solutions", None)
    task_api = None
    if legacy_api is None:
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise HandSignError(
                f"MediaPipe {getattr(mp, '__version__', 'terpasang')} tidak "
                "menyediakan API solutions maupun Tasks yang diperlukan. "
                "Pasang ulang dengan: python -m pip install -r requirements.txt"
            ) from exc
        task_api = (python, vision)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        capture.release()
        raise HandSignError(
            f"Tidak dapat membuka webcam index {args.camera}. "
            "Periksa koneksi kamera atau pilih index lain dengan --camera."
        )

    previous_time = time.perf_counter()
    fps = 0.0

    try:
        with ExitStack() as stack:
            if legacy_api is not None:
                mp_hands = legacy_api.hands
                detector = stack.enter_context(
                    mp_hands.Hands(
                        static_image_mode=False,
                        max_num_hands=1,
                        model_complexity=0,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6,
                    )
                )

                def detect_hands(frame_rgb: np.ndarray, timestamp_ms: int):
                    del timestamp_ms
                    results = detector.process(frame_rgb)
                    return results.multi_hand_landmarks or []

                def draw_landmarks(frame: np.ndarray, landmarks: object) -> None:
                    legacy_api.drawing_utils.draw_landmarks(
                        frame, landmarks, mp_hands.HAND_CONNECTIONS
                    )

            else:
                python, vision = task_api
                asset_path = ensure_hand_landmarker_model()
                options = vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(asset_path)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=1,
                    min_hand_detection_confidence=0.6,
                    min_hand_presence_confidence=0.6,
                    min_tracking_confidence=0.6,
                )
                detector = stack.enter_context(
                    vision.HandLandmarker.create_from_options(options)
                )

                def detect_hands(frame_rgb: np.ndarray, timestamp_ms: int):
                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=frame_rgb
                    )
                    results = detector.detect_for_video(image, timestamp_ms)
                    return results.hand_landmarks or []

                def draw_landmarks(frame: np.ndarray, landmarks: object) -> None:
                    draw_task_landmarks(
                        frame,
                        landmarks,
                        vision.HandLandmarksConnections.HAND_CONNECTIONS,
                        cv2,
                    )

            while True:
                success, frame = capture.read()
                if not success or frame is None:
                    raise HandSignError(
                        "Gagal membaca frame dari webcam. Periksa koneksi kamera."
                    )
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                frame_rgb.flags.writeable = False
                timestamp_ms = int(time.perf_counter() * 1000)
                detected_hands = detect_hands(frame_rgb, timestamp_ms)
                frame_rgb.flags.writeable = True

                status = "Tangan tidak terdeteksi"
                if detected_hands:
                    hand_landmarks = detected_hands[0]
                    draw_landmarks(frame, hand_landmarks)
                    landmark_points = getattr(
                        hand_landmarks, "landmark", hand_landmarks
                    )
                    raw_features = [
                        coordinate
                        for landmark in landmark_points
                        for coordinate in (landmark.x, landmark.y)
                    ]
                    normalized_features = normalize_landmarks(raw_features)
                    input_frame = pd.DataFrame(
                        [normalized_features], columns=FEATURE_COLUMNS
                    )
                    probabilities = model.predict_proba(input_frame)[0]
                    best_index = int(np.argmax(probabilities))
                    label = model.classes_[best_index]
                    confidence = float(probabilities[best_index])
                    status = f"Prediksi: {label} ({confidence:.1%})"

                now = time.perf_counter()
                elapsed = now - previous_time
                previous_time = now
                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    fps = instant_fps if fps == 0 else 0.85 * fps + 0.15 * instant_fps

                cv2.putText(
                    frame,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0) if detected_hands else (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "Tekan Q untuk keluar",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("Hand Sign Recognition", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Training dan deteksi bahasa isyarat real-time berbasis landmark."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Latih model dari dataset CSV.")
    train_parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATASET, help="Lokasi file CSV."
    )
    train_parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL, help="Lokasi file model keluaran."
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi data evaluasi (default: 0.2).",
    )

    webcam_parser = commands.add_parser(
        "webcam", help="Jalankan prediksi dari kamera secara real-time."
    )
    webcam_parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL, help="Lokasi file model."
    )
    webcam_parser.add_argument(
        "--camera", type=int, default=0, help="Index kamera (default: 0)."
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        if not 0 < args.test_size < 1:
            parser.error("--test-size harus lebih besar dari 0 dan lebih kecil dari 1.")
        train(args)
    elif args.command == "webcam":
        if args.camera < 0:
            parser.error("--camera tidak boleh bernilai negatif.")
        run_webcam(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HandSignError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
