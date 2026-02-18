"""
Detection metrics using Scikit-Learn, Pandas and Matplotlib.

Features:
- IoU calculation (torchvision.ops.box_iou)
- Metrics Report (sklearn + pandas) -> output/metrics.csv
- Confusion Matrix (sklearn + matplotlib) -> output/confusion_matrix.png
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.config import (
    CLASSES,
    VISUALIZATION_THRESHOLDS,
    VISUALIZATION_DISABLED_CLASSES,
)

# Backend no interactivo: necesario en Docker (sin pantalla)
plt.switch_backend("Agg")


def _filter_predictions(annotations: List[Dict]) -> List[Dict]:
    """Apply visualization thresholds and disabled-class filtering."""
    filtered = []
    for ann in annotations:
        cls_id = ann["category_id"]
        # Omite clases desactivadas (ej. FA DC)
        if cls_id in VISUALIZATION_DISABLED_CLASSES:
            continue
        # Omite predicciones por debajo del umbral de visualización
        threshold = VISUALIZATION_THRESHOLDS.get(cls_id, 0.5)
        if ann.get("score", 1.0) < threshold:
            continue
        filtered.append(ann)
    return filtered


def _match_boxes(
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    iou_threshold: float
) -> List[Tuple[int, int, float]]:
    """
    Match GT and Pred boxes using greedy IoU.
    
    Returns:
        List of (gt_idx, pred_idx, iou) tuples for matched pairs.
    """
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return []

    # Calcula la matriz IoU completa entre todos los GT y todas las predicciones
    iou_matrix = box_iou(gt_boxes, pred_boxes)

    # Ordena todos los pares por IoU descendente para hacer matching greedy
    values, indices = torch.sort(iou_matrix.flatten(), descending=True)
    
    matched_gt = set()
    matched_pred = set()
    matched_pairs = []

    for idx, val in zip(indices, values):
        if val < iou_threshold:
            break  # El resto tendrá IoU aún menor, podemos parar
            
        gt_idx = (idx // iou_matrix.shape[1]).item()
        pred_idx = (idx % iou_matrix.shape[1]).item()

        # Cada GT y cada predicción solo puede emparejarse una vez
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue

        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        matched_pairs.append((gt_idx, pred_idx, val.item()))

    return matched_pairs


def _build_classification_data(
    gt_json_path: Path,
    results_json_path: Path,
    iou_threshold: float = 0.4,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Match GT vs Predictions and build y_true/y_pred alignment lists.

    Returns:
        (y_true, y_pred, labels) where labels excludes 'Background'.
    """
    with open(gt_json_path, "r") as f:
        gt_data = json.load(f)
    with open(results_json_path, "r") as f:
        pred_data = json.load(f)

    # Solo evalúa imágenes que aparezcan en ambos JSONs
    gt_file_to_id = {img["file_name"]: img["id"] for img in gt_data["images"]}
    pred_file_to_id = {img["file_name"]: img["id"] for img in pred_data["images"]}
    common_files = set(gt_file_to_id) & set(pred_file_to_id)

    # Agrupa GT por imagen, excluyendo clases desactivadas
    gt_by_img: Dict[int, List[Dict]] = {}
    for ann in gt_data["annotations"]:
        img_id = ann["image_id"]
        fname = next((k for k, v in gt_file_to_id.items() if v == img_id), None)
        if fname not in common_files:
            continue
        if ann["category_id"] in VISUALIZATION_DISABLED_CLASSES:
            continue
        gt_by_img.setdefault(img_id, []).append(ann)

    # Filtra predicciones con los mismos criterios que la visualización
    filtered_preds = _filter_predictions(pred_data.get("annotations", []))
    pred_by_img: Dict[int, List[Dict]] = {}
    for ann in filtered_preds:
        pred_by_img.setdefault(ann["image_id"], []).append(ann)

    y_true: List[str] = []
    y_pred: List[str] = []

    all_img_ids = set(gt_by_img.keys()) | set(pred_by_img.keys())
    for img_id in all_img_ids:
        gts = gt_by_img.get(img_id, [])
        preds = pred_by_img.get(img_id, [])

        # Convierte bboxes [x,y,w,h] → [x1,y1,x2,y2] para torchvision.box_iou
        g_boxes = (
            torch.tensor([
                [a["bbox"][0], a["bbox"][1],
                 a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                for a in gts
            ], dtype=torch.float32) if gts
            else torch.empty((0, 4), dtype=torch.float32)
        )
        p_boxes = (
            torch.tensor([
                [a["bbox"][0], a["bbox"][1],
                 a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                for a in preds
            ], dtype=torch.float32) if preds
            else torch.empty((0, 4), dtype=torch.float32)
        )

        matched_pairs = _match_boxes(g_boxes, p_boxes, iou_threshold)
        used_g: Set[int] = set()
        used_p: Set[int] = set()

        # Pares emparejados: verdaderos positivos (o errores de clase)
        for gi, pi, _ in matched_pairs:
            used_g.add(gi)
            used_p.add(pi)
            y_true.append(CLASSES.get(gts[gi]["category_id"], str(gts[gi]["category_id"])))
            y_pred.append(CLASSES.get(preds[pi]["category_id"], str(preds[pi]["category_id"])))

        # GT sin match → falso negativo (el modelo no lo detectó)
        for i, ann in enumerate(gts):
            if i not in used_g:
                y_true.append(CLASSES.get(ann["category_id"], str(ann["category_id"])))
                y_pred.append("Background")

        # Predicción sin match → falso positivo (el modelo detectó algo que no existe)
        for i, ann in enumerate(preds):
            if i not in used_p:
                y_true.append("Background")
                y_pred.append(CLASSES.get(ann["category_id"], str(ann["category_id"])))

    labels = sorted([c for c in set(y_true) | set(y_pred) if c != "Background"])
    return y_true, y_pred, labels


def compute_metrics(
    gt_json_path: Path,
    results_json_path: Path,
    iou_threshold: float = 0.4,
) -> Dict[str, Any]:
    """Compute metrics and save CSV + PNG to disk (for CLI)."""
    output_dir = results_json_path.parent

    y_true, y_pred, labels = _build_classification_data(
        gt_json_path, results_json_path, iou_threshold,
    )
    cm_labels = labels + ["Background"]

    print(f"\n{'=' * 80}")
    print(f"  Generating Metrics Reports (IoU > {iou_threshold})")
    print(f"{'=' * 80}")

    # Genera el informe de clasificación (precision, recall, f1, support) por clase
    report_dict = classification_report(
        y_true, y_pred, labels=labels,
        output_dict=True, zero_division=0.0,
    )
    df = pd.DataFrame(report_dict).transpose()
    csv_path = output_dir / "metrics.csv"
    df.to_csv(csv_path)
    print(f"Saved metrics CSV -> {csv_path}")

    # Genera y guarda la matriz de confusión como imagen PNG
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title(f"Confusion Matrix (IoU > {iou_threshold})")
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix -> {cm_path}")
    print(f"{'=' * 80}\n")

    return {}


def compute_metrics_for_ui(
    gt_json_path: Path,
    results_json_path: Path,
    iou_threshold: float = 0.4,
) -> Tuple["pd.DataFrame", "plt.Figure"]:
    """
    Compute metrics and return Gradio-compatible objects.

    Returns:
        (metrics_dataframe, confusion_matrix_figure)
        - DataFrame: rows = classes, cols = precision/recall/f1/support
        - Figure: matplotlib confusion matrix figure for gr.Plot
    """
    y_true, y_pred, labels = _build_classification_data(
        gt_json_path, results_json_path, iou_threshold,
    )
    cm_labels = labels + ["Background"]

    # DataFrame de métricas para mostrar en la tabla de Gradio
    report_dict = classification_report(
        y_true, y_pred, labels=labels,
        output_dict=True, zero_division=0.0,
    )
    df = pd.DataFrame(report_dict).transpose()

    # Redondea para mejor legibilidad en la UI
    for col in ["precision", "recall", "f1-score"]:
        if col in df.columns:
            df[col] = df[col].round(2)
    if "support" in df.columns:
        df["support"] = df["support"].astype(int)

    # Convierte el índice en columna para que gr.Dataframe lo muestre correctamente
    df = df.reset_index().rename(columns={"index": "Class"})

    # También guarda en disco
    output_dir = results_json_path.parent
    df.to_csv(output_dir / "metrics.csv")

    # Figura de la matriz de confusión para gr.Plot
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix (IoU > {iou_threshold})", fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout(pad=0.5)

    # También guarda en disco
    fig.savefig(output_dir / "confusion_matrix.png")

    return df, fig
