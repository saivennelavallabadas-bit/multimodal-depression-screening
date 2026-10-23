"""Train a compact multimodal neural baseline on participant-level splits.

The modality inputs are fixed audio/video descriptors and TF-IDF text vectors;
the learned components are the modality MLPs and fusion head.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features import audio_features, video_features, read_transcript, resolve_path


class MultimodalMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.audio = nn.Sequential(nn.Linear(dims[0], 64), nn.ReLU(), nn.Dropout(.25), nn.Linear(64, 32), nn.ReLU())
        self.video = nn.Sequential(nn.Linear(dims[1], 64), nn.ReLU(), nn.Dropout(.25), nn.Linear(64, 32), nn.ReLU())
        self.text = nn.Sequential(nn.Linear(dims[2], 128), nn.ReLU(), nn.Dropout(.25), nn.Linear(128, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.Dropout(.25), nn.Linear(48, 1))

    def forward(self, audio, video, text):
        joined = torch.cat([self.audio(audio), self.video(video), self.text(text)], dim=1)
        return self.fusion(joined).squeeze(1)


def load_part(path):
    frame = pd.read_csv(path, dtype={"participant_id": str})
    required = {"participant_id", "audio_path", "video_path", "transcript_path", "label"}
    if not required.issubset(frame):
        raise ValueError(f"{path} missing {sorted(required - set(frame.columns))}")
    if frame.participant_id.duplicated().any():
        raise ValueError(f"Duplicate participant in {path}")
    return frame


def extract(frame, root):
    audio, video, texts = [], [], []
    for row in frame.itertuples(index=False):
        try:
            audio.append(audio_features(resolve_path(row.audio_path, root)))
            video.append(video_features(resolve_path(row.video_path, root)))
            texts.append(read_transcript(resolve_path(row.transcript_path, root)))
        except Exception as exc:
            raise RuntimeError(f"Feature extraction failed for {row.participant_id}: {exc}") from exc
    return [np.vstack(audio), np.vstack(video), texts]


def score(y, probs):
    pred = (probs >= .5).astype(int)
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else None,
            "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out-dir", default="artifacts/deep_run_001")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-source", choices=["authorized", "synthetic"], required=True,
                    help="Explicit provenance tag saved with the experiment output")
    args = ap.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(False)
    parts = {k: load_part(Path(args.splits_dir) / f"{k}.csv") for k in ("train", "val", "test")}
    idsets = [set(p.participant_id) for p in parts.values()]
    if any(idsets[i] & idsets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Participant leakage across splits")
    manifest = pd.read_csv(args.manifest, dtype={"participant_id": str})
    manifest_ids = set(manifest.participant_id)
    split_ids = set.union(*idsets)
    if split_ids != manifest_ids:
        raise ValueError("Train/validation/test participants must exactly match the manifest")
    if args.data_source == "authorized" and any(str(pid).startswith("DEMO") for pid in manifest_ids):
        raise ValueError("DEMO participant IDs require --data-source synthetic")
    for part in parts.values():
        if part.label.isna().any() or not set(part.label.astype(int)).issubset({0, 1}):
            raise ValueError("Labels must be binary 0/1")
    root = Path(args.manifest).resolve().parent.parent
    raw = {k: extract(v, root) for k, v in parts.items()}
    processors = []
    arrays = {k: [] for k in parts}
    for modality in (0, 1):
        imp, scale = SimpleImputer(strategy="median"), StandardScaler()
        train_x = scale.fit_transform(imp.fit_transform(raw["train"][modality]))
        processors.append((imp, scale))
        for key in parts:
            arrays[key].append(scale.transform(imp.transform(raw[key][modality])).astype("float32"))
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    arrays["train"].append(vectorizer.fit_transform(raw["train"][2]).toarray().astype("float32"))
    for key in ("val", "test"):
        arrays[key].append(vectorizer.transform(raw[key][2]).toarray().astype("float32"))
    y = {k: v.label.astype(int).to_numpy(dtype="float32") for k, v in parts.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalMLP([x.shape[1] for x in arrays["train"]]).to(device)
    positive = float(y["train"].sum())
    negative = len(y["train"]) - positive
    if not positive or not negative:
        raise ValueError("Training split must include both label classes")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = TensorDataset(*(torch.from_numpy(x) for x in arrays["train"]), torch.from_numpy(y["train"]))
    loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)
    best_loss, best_state, patience = float("inf"), None, 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        for *batch, target in loader:
            batch = [x.to(device) for x in batch]
            target = target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(*batch), target)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_inputs = [torch.from_numpy(x).to(device) for x in arrays["val"]]
            val_loss = criterion(model(*val_inputs), torch.from_numpy(y["val"]).to(device)).item()
        history.append({"epoch": epoch + 1, "validation_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, patience = val_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 8:
                break
    model.load_state_dict(best_state)
    model.eval()
    def predict(xs):
        with torch.no_grad():
            inputs = [torch.from_numpy(x).to(device) for x in xs]
            return torch.sigmoid(model(*inputs)).cpu().numpy()
    val_prob, test_prob = predict(arrays["val"]), predict(arrays["test"])
    # Sensitivity analysis: zero one standardized modality at a time. This is a
    # model-dependence measure, not a causal or clinical explanation.
    importance = {}
    base_f1 = score(y["test"], test_prob)["f1"]
    for i, name in enumerate(("audio", "video", "text")):
        ablated = [x.copy() for x in arrays["test"]]
        ablated[i][:] = 0
        ablated_f1 = score(y["test"], predict(ablated))["f1"]
        importance[name] = {"f1_after_zeroing_modality": ablated_f1,
                            "f1_change_from_full_model": float(base_f1 - ablated_f1)}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"audio_video_processors": processors, "text_vectorizer": vectorizer}, out / "preprocessing.joblib")
    torch.save({"state_dict": model.state_dict(), "input_dims": [x.shape[1] for x in arrays["train"]]}, out / "model.pt")
    result = {"research_only": True, "data_source": args.data_source,
              "warning": "Synthetic data metrics are software checks only; never report them as research results.",
              "seed": args.seed, "epochs_trained": len(history),
              "counts": {k: len(v) for k, v in parts.items()},
              "validation_metrics": score(y["val"], val_prob), "test_metrics": score(y["test"], test_prob),
              "explainability": {"method": "test-set modality zeroing sensitivity", "baseline_test_f1": base_f1,
                                 "modality_ablation": importance,
                                 "caveat": "Model dependence only; not causal, clinical, or individual-level evidence."},
              "training_history": history}
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "output": str(out.resolve()), "test_metrics": result["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
