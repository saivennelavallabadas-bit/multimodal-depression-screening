
import sys
import json
from pathlib import Path

import plotly.express as px
import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_processing import extract_audio_features
from src.video_processing import extract_video_features
from src.text_processing import extract_text_features

ARTIFACTS = PROJECT_ROOT / "artifacts" / "lmvd_baseline"
MODELS_DIR = ARTIFACTS / "models"
METRICS_PATH = ARTIFACTS / "metrics.json"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed_lmvd.npz"

st.set_page_config(
    page_title="Multimodal Depression Research",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {
    padding: 1.5rem;
    border-radius: 16px;
    background: linear-gradient(120deg, #14243b, #273e5c);
    color: white;
}
.hero h1 {color: white;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🧠 Research Framework")
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Speech Analysis",
        "Facial Video Analysis",
        "Text Analysis",
        "Multimodal Fusion",
        "Explainability",
        "Evaluation",
    ],
)

st.sidebar.caption("Research prototype — not a diagnostic tool.")
st.title("Multimodal Explainable Deep Learning Framework")
st.caption(
    "Research using speech, facial/visual features, and text analysis"
)
st.warning(
    "Research use only. This application is not a medical diagnostic "
    "tool. Do not use its outputs to make clinical decisions."
)


@st.cache_resource
def load_models():
    models = {}
    for name in ["audio", "visual", "fusion"]:
        path = MODELS_DIR / f"{name}_model.joblib"
        if path.exists():
            models[name] = joblib.load(path)
    return models


def load_metrics():
    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    return None


def show_upload(label, types, key):
    uploaded = st.file_uploader(label, type=types, key=key)
    if uploaded is not None:
        st.success(f"File selected: {uploaded.name}")
        st.caption(f"Size: {uploaded.size / (1024 * 1024):.2f} MB")
    return uploaded


def show_saved_evaluation(metrics):
    if not isinstance(metrics, dict):
        st.info("No saved evaluation metrics were found.")
        return

    for model_name, result in metrics.items():
        if not isinstance(result, dict):
            continue

        st.subheader(model_name.replace("_", " ").title())

        # Support the metrics structure produced by train_lmvd.py.
        for split_name in ["validation", "test"]:
            split_result = result.get(split_name)
            if isinstance(split_result, dict):
                st.markdown(f"**{split_name.title()}**")
                cols = st.columns(5)
                for col, metric_name in zip(
                    cols,
                    ["accuracy", "precision", "recall", "f1", "roc_auc"],
                ):
                    value = split_result.get(metric_name)
                    if isinstance(value, (int, float)):
                        col.metric(
                            metric_name.upper(),
                            f"{value:.3f}",
                        )

                cm = split_result.get("confusion_matrix")
                if cm is not None:
                    st.write("Confusion matrix (actual rows, predicted columns)")
                    st.dataframe(
                        pd.DataFrame(
                            cm,
                            index=["Actual Normal", "Actual Depression"],
                            columns=["Predicted Normal", "Predicted Depression"],
                        )
                    )

        # Also support flat metrics dictionaries, if present.
        if "test" not in result and "accuracy" in result:
            st.json(result)


try:
    models = load_models()
except Exception as exc:
    models = {}
    st.error(f"Could not load saved models: {exc}")

metrics_data = load_metrics()

if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>Research Dashboard</h1>
        <p>Speech and visual baselines with an early feature-fusion model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Project status")
    c1, c2, c3 = st.columns(3)
    c1.metric("Saved baseline models", len(models))
    c2.metric("Processed samples", "1,823" if PROCESSED_PATH.exists() else "Not found")
    c3.metric("Saved evaluation", "Available" if metrics_data else "Not found")

    
    st.write("**Available audio-visual models:**", ", ".join(models.keys()) or "None")

    st.subheader("English Text Model")
    text_metrics_path = (
        PROJECT_ROOT / "artifacts" / "english_text" / "metrics.json"
    )

    if text_metrics_path.exists():
        try:
            with text_metrics_path.open("r", encoding="utf-8") as file:
                text_metrics = json.load(file)

            st.caption(
                f"Dataset: {text_metrics.get('dataset', 'Not specified')} | "
                f"Method: {text_metrics.get('method', 'Not specified')}"
            )

            metric_cols = st.columns(4)
            for col, metric_name, label in zip(
                metric_cols,
                ["accuracy", "precision", "recall", "f1"],
                ["Accuracy", "Precision", "Recall", "F1-score"],
            ):
                value = text_metrics.get(metric_name)
                if isinstance(value, (int, float)):
                    col.metric(label, f"{value:.2%}")

            st.caption(
                f"Training samples: {text_metrics.get('training_samples', 'N/A')} | "
                f"Test samples: {text_metrics.get('test_samples', 'N/A')}"
            )

            st.info(
                "The English text model is trained separately from the "
                "audio-visual models. Its scores are not directly comparable "
                "to the audio-visual results because the datasets differ."
            )
        except Exception as exc:
            st.error(f"Could not load text evaluation metrics: {exc}")
    else:
        st.warning("English text evaluation metrics were not found.")

    st.warning(
        "Research prototype only. The English text labels are dataset-provided, "
        "not clinical diagnoses. Current audio-visual evaluation uses "
        "sample-level splits; participant-independent generalization has "
        "not been established."
    )

    st.info(
        "Current evaluation used sample-level splits. Participant-independent "
        "generalization has not been established."
    )

elif page == "Speech Analysis":
    import tempfile
    import os

    st.header("Speech Analysis")
    st.write("Upload audio to extract acoustic features.")

    audio = show_upload(
        "Choose an audio file", ["wav", "flac", "ogg"], "speech_file"
    )

    if audio is not None:
        st.audio(audio)
        if st.button("Extract audio features"):
            try:
                features = extract_audio_features(audio.getvalue())
                st.success("Feature extraction completed.")
                st.json(features)
                st.caption(
                    "Feature extraction alone does not produce a validated "
                    "depression prediction."
                )
            except Exception as exc:
                st.error(f"Could not process this audio: {exc}")

elif page == "Facial Video Analysis":
    import tempfile
    import os

    st.header("Facial Video Analysis")
    st.write("Upload a video to extract basic visual statistics.")

    video = show_upload(
        "Choose a video file", ["mp4", "mov", "avi", "mkv"], "video_file"
    )

    if video is not None:
        st.video(video)
        if st.button("Extract video features"):
            temp_path = None
            try:
                suffix = "." + video.name.rsplit(".", 1)[-1].lower()
                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                ) as temp:
                    temp.write(video.getvalue())
                    temp_path = temp.name

                features = extract_video_features(temp_path)
                st.success("Video processing completed.")
                st.json(features)
                st.caption(
                    "These measurements are not a validated depression "
                    "prediction or a reliable emotion assessment."
                )
            except Exception as exc:
                st.error(f"Could not process this video: {exc}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)




elif page == "Text Analysis":
    st.header("Text Analysis")
    st.write(
        "Text-only research baseline using the English Reddit dataset."
    )

    text_model_path = (
        PROJECT_ROOT
        / "artifacts"
        / "english_text"
        / "english_text_model.joblib"
    )

    metrics_path = (
        PROJECT_ROOT
        / "artifacts"
        / "english_text"
        / "metrics.json"
    )

    text_input = st.text_area(
        "Enter text for research experimentation",
        height=200,
        placeholder="Enter text here...",
    )

    if st.button("Run Text Analysis"):
        if not text_input.strip():
            st.warning("Please enter some text first.")

        elif not text_model_path.exists():
            st.error(
                "Text model not found. Check that "
                "artifacts/english_text/english_text_model.joblib exists."
            )

        else:
            try:
                # Load the trained model.
                text_model = joblib.load(text_model_path)

                # Generate the predicted dataset label.
                prediction = int(text_model.predict([text_input])[0])

                # Calculate estimated probabilities for both classes.
                probabilities = text_model.predict_proba([text_input])[0]
                class_labels = list(text_model.classes_)

                predicted_index = class_labels.index(prediction)
                predicted_probability = float(
                    probabilities[predicted_index]
                )

                # Display the prediction.
                st.subheader("Experimental Model Output")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Predicted dataset label",
                        prediction,
                    )

                with col2:
                    st.metric(
                        "Estimated model probability",
                        f"{predicted_probability:.2%}",
                    )

                st.progress(predicted_probability)

                st.caption(
                    "This probability is the model's estimated probability "
                    "for the predicted dataset label. It is not the "
                    "probability that a person has depression."
                )

                # Display probabilities for both classes.
                st.write("**Class probabilities**")

                probability_data = pd.DataFrame({
                    "Dataset label": class_labels,
                    "Estimated probability": [
                        float(value) for value in probabilities
                    ],
                })

                st.dataframe(
                    probability_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.write(
                    f"The model predicted dataset label {prediction}."
                )

                # Explain words influencing the prediction.
                
                # Explain words influencing the prediction.
                st.subheader("Text Explainability")

                try:
                    vectorizer = text_model.named_steps["tfidf"]
                    classifier = text_model.named_steps["classifier"]

                    word_values = vectorizer.transform([text_input])
                    feature_names = vectorizer.get_feature_names_out()
                    coefficients = classifier.coef_[0]

                    contributions = (
                        word_values.toarray()[0] * coefficients
                    )

                    explanation = pd.DataFrame({
                        "Word or phrase": feature_names,
                        "Contribution": contributions,
                    })

                    explanation = explanation[
                        explanation["Contribution"] != 0
                    ].copy()

                    if explanation.empty:
                        st.info(
                            "No words from the model vocabulary were "
                            "found to explain this prediction."
                        )
                    else:
                        positive = explanation[
                            explanation["Contribution"] > 0
                        ].copy()

                        negative = explanation[
                            explanation["Contribution"] < 0
                        ].copy()

                        positive["Strength"] = positive["Contribution"]
                        negative["Strength"] = negative["Contribution"].abs()

                        st.caption(
                            "These values show how words influence the "
                            "linear model's decision. They are not clinical "
                            "indicators or explanations of a person's mental state."
                        )

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Pushes toward label 1**")
                            if positive.empty:
                                st.info("No positive contributions found.")
                            else:
                                st.dataframe(
                                    positive[
                                        ["Word or phrase", "Contribution"]
                                    ]
                                    .sort_values(
                                        "Contribution",
                                        ascending=False,
                                    )
                                    .head(10),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                        with col2:
                            st.markdown("**Pushes toward label 0**")
                            if negative.empty:
                                st.info("No negative contributions found.")
                            else:
                                st.dataframe(
                                    negative[
                                        ["Word or phrase", "Contribution"]
                                    ]
                                    .sort_values(
                                        "Contribution",
                                        ascending=True,
                                    )
                                    .head(10),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                except Exception as exc:
                    st.info(
                        f"Text explanation is unavailable: {exc}"
                    )

                st.warning(
                    "Research use only. This model was trained on English "
                    "Reddit data. Predictions reflect dataset labels, "
                    "not clinical diagnoses."
                )

            except Exception as exc:
                st.error(f"Text model error: {exc}")

    if metrics_path.exists():
        with st.expander("View text model evaluation results"):
            try:
                with metrics_path.open("r", encoding="utf-8") as file:
                    st.json(json.load(file))
            except Exception as exc:
                st.error(
                    f"Could not load evaluation results: {exc}"
                )


elif page == "Multimodal Fusion":
    st.header("Multimodal Fusion")
    st.write(
        "This page describes the saved early-fusion baseline. "
        "The current trained fusion model uses concatenated audio and "
        "visual feature vectors; it does not include a trained text branch."
    )

    if "fusion" in models:
        st.success("Saved audio-visual fusion model loaded successfully.")
        st.info(
            "Raw uploaded audio/video cannot be passed directly into this "
            "model until their extraction pipelines are aligned with the "
            "training feature representation."
        )
    else:
        st.warning("Fusion model file was not found.")


elif page == "Explainability":
    st.header("Explainability")
    st.write(
        "Inspect the learned coefficients of the saved audio baseline. "
        "These coefficients describe the model's behavior; they do not "
        "establish clinical causes or validated depression indicators."
    )

    try:
        models = load_models()
        audio_model = models.get("audio")

        if audio_model is None:
            st.warning("The saved audio model could not be loaded.")
        else:
            # Find the fitted estimator inside the saved pipeline.
            estimator = audio_model
            if hasattr(audio_model, "named_steps"):
                candidates = list(audio_model.named_steps.values())
                fitted = [
                    step for step in candidates
                    if hasattr(step, "coef_")
                ]
                if fitted:
                    estimator = fitted[-1]

            if not hasattr(estimator, "coef_"):
                st.warning(
                    "This saved estimator does not expose linear "
                    "coefficients. A model-specific explanation method "
                    "is required for this model."
                )
            else:
                coefficients = np.asarray(estimator.coef_).ravel()

                importance = pd.DataFrame({
                    "Feature index": [
                        f"Audio feature {i}"
                        for i in range(len(coefficients))
                    ],
                    "Coefficient": coefficients,
                    "Absolute coefficient": np.abs(coefficients),
                }).sort_values(
                    "Absolute coefficient", ascending=False
                )

                st.subheader("Largest learned coefficients")
                st.caption(
                    "Feature indices are used because the original "
                    "feature names are not available here. Positive and "
                    "negative signs indicate the model's learned direction."
                )

                st.dataframe(
                    importance.head(20),
                    use_container_width=True,
                    hide_index=True,
                )

                chart_data = (
                    importance.head(20)
                    .sort_values("Coefficient")
                    .set_index("Feature index")["Coefficient"]
                )
                st.bar_chart(chart_data)

                st.info(
                    "These are model coefficients, not proof that a "
                    "feature causes depression or reliably identifies it. "
                    "Validate explanations before making research claims."
                )

    except Exception as exc:
        st.error(f"Could not generate the explanation: {exc}")




elif page == "Evaluation":
    st.header("Model Evaluation")
    st.caption(
        "Performance comparison of the saved audio-only, "
        "visual-only, and audio-visual fusion baseline models."
    )

    metrics_data = load_metrics()

    if not metrics_data:
        st.warning("Saved evaluation metrics were not found.")
    else:
        models = metrics_data.get("models", {})
        preferred = ["audio", "visual", "fusion"]

        test_results = {
            name: models[name].get("test", {})
            for name in preferred
            if isinstance(models.get(name), dict)
            and models[name].get("test")
        }

        if not test_results:
            st.warning("No test-set model metrics were found.")
        else:
            available = list(test_results.keys())

            # Key performance indicators
            st.subheader("Test-set performance")

            best_accuracy = max(
                available,
                key=lambda n: test_results[n].get("accuracy", -1),
            )
            best_f1 = max(
                available,
                key=lambda n: test_results[n].get("f1", -1),
            )
            best_auc = max(
                available,
                key=lambda n: test_results[n].get("roc_auc", -1),
            )

            cols = st.columns(3)

            cols[0].metric(
                "Highest accuracy",
                best_accuracy.title(),
                f"{test_results[best_accuracy]['accuracy']:.2%}",
            )
            cols[1].metric(
                "Highest F1-score",
                best_f1.title(),
                f"{test_results[best_f1]['f1']:.3f}",
            )
            cols[2].metric(
                "Highest ROC-AUC",
                best_auc.title(),
                f"{test_results[best_auc]['roc_auc']:.3f}",
            )

            # Comparison table
            rows = []
            for name in available:
                result = test_results[name]
                rows.append({
                    "Model": name.title(),
                    "Accuracy": result.get("accuracy"),
                    "Precision": result.get("precision"),
                    "Recall": result.get("recall"),
                    "F1-score": result.get("f1"),
                    "ROC-AUC": result.get("roc_auc"),
                })

            comparison = pd.DataFrame(rows).set_index("Model")

            st.subheader("Detailed metric comparison")
            st.dataframe(
                comparison.style.format("{:.3f}", na_rep="N/A"),
                use_container_width=True,
            )

            # Compare all available metrics
            
            # Compare all available metrics
            st.subheader("Performance comparison chart")
            st.caption(
                "Each bar represents an individual model score. "
                "All metrics range from 0 to 1."
            )

            chart_data = comparison.reset_index().melt(
                id_vars="Model",
                value_vars=[
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1-score",
                    "ROC-AUC",
                ],
                var_name="Metric",
                value_name="Score",
            )

            fig = px.bar(
                chart_data,
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                text_auto=".3f",
                range_y=[0, 1],
                title="Audio vs Visual vs Audio-Visual Fusion",
            )

            fig.update_traces(
                textposition="outside",
                cliponaxis=False,
            )

            fig.update_layout(
                yaxis_title="Score",
                xaxis_title="Evaluation metric",
                legend_title="Model",
                yaxis=dict(range=[0, 1.08]),
                height=500,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            # Confusion matrices
            st.subheader("Confusion matrices")
            st.caption(
                "Rows represent actual dataset labels; columns represent "
                "predicted dataset labels."
            )

            matrix_cols = st.columns(len(available))

            for col, name in zip(matrix_cols, available):
                with col:
                    st.markdown(f"**{name.title()} model**")
                    matrix = test_results[name].get("confusion_matrix")

                    if matrix is not None:
                        st.dataframe(
                            pd.DataFrame(
                                matrix,
                                index=["Actual 0", "Actual 1"],
                                columns=["Predicted 0", "Predicted 1"],
                            ),
                            use_container_width=True,
                        )
                    else:
                        st.info("Confusion matrix not saved.")

            # Dataset split information
            counts = metrics_data.get("split_counts", {})
            st.subheader("Dataset split")

            split_cols = st.columns(3)
            split_cols[0].metric(
                "Training samples", counts.get("train", "N/A")
            )
            split_cols[1].metric(
                "Validation samples", counts.get("validation", "N/A")
            )
            split_cols[2].metric(
                "Test samples", counts.get("test", "N/A")
            )

            st.info(
                "These are baseline results on the saved test split. "
                "The highest score does not establish clinical validity. "
                "The audio-visual fusion model combines audio and visual "
                "features; the English text model is evaluated separately "
                "on a different dataset."
            )

        st.warning(
            metrics_data.get(
                "warning",
                "Research prototype only. Dataset labels are not clinical "
                "diagnoses, and participant-independent generalization "
                "has not been established."
            )
        )

        with st.expander("View complete saved metrics (JSON)"):
            st.json(metrics_data)
