import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report, balanced_accuracy_score, f1_score
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset
from model import MNIST_CNN
from data_loader import load_medical_partitions
from config import FLConfig


def _risk_tier(precision, recall):
    if precision >= 0.90 and recall >= 0.90:
        return "Low"
    if precision >= 0.75 and recall >= 0.75:
        return "Moderate"
    return "High"


def _treatment_suggestion(class_name, risk_tier):
    if class_name.startswith("CIFAR"):
        return "Non-clinical label group; do not use for treatment decisions."

    if risk_tier == "Low":
        return "Use as supportive evidence; correlate with history, labs, and imaging findings."
    if risk_tier == "Moderate":
        return "Request secondary radiology review before final treatment planning."
    return "Escalate to specialist review and confirm with additional imaging/tests."


def generate_combined_report():
    # 1. Setup
    config = FLConfig()
    device = torch.device("cpu")

    print("Loading datasets for evaluation...")
    train_loaders, test_loader, names = load_medical_partitions(config)
    num_classes = len(names)
    print(f"Detected {num_classes} classes: {names}")

    # 2. Load the Saved Global Model (infer num_classes from checkpoint)
    model_path = "models/global_fedavg.pth"
    try:
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)
        ckpt_classes = checkpoint["fc2.weight"].shape[0]
        model = MNIST_CNN(num_classes=ckpt_classes).to(device)
        model.load_state_dict(checkpoint)
        print(f"Successfully loaded '{model_path}' ({ckpt_classes} classes)")
    except FileNotFoundError:
        print("Error: Model file not found. Please run main.py first.")
        return

    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []
    misclassified_examples = []
    max_misclassified_examples = getattr(config, "max_misclassified_examples", 16)
    eval_max_samples = getattr(config, "eval_max_samples", 12000)

    # Build a random evaluation loader (avoids class bias from sequential truncation).
    eval_dataset = test_loader.dataset
    total_eval_size = len(eval_dataset)
    if eval_max_samples is not None and total_eval_size > eval_max_samples:
        rand_idx = torch.randperm(total_eval_size)[:eval_max_samples].tolist()
        eval_dataset = Subset(eval_dataset, rand_idx)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=test_loader.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 3. Collect Evaluation Data
    print("Collecting predictions...")
    with torch.no_grad():
        for images, labels in eval_loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)      # shape [B, num_classes]
            preds = torch.argmax(outputs, dim=1)

            # Keep a bounded sample of failure cases for qualitative review.
            if len(misclassified_examples) < max_misclassified_examples:
                for i in range(images.size(0)):
                    if len(misclassified_examples) >= max_misclassified_examples:
                        break
                    true_label = int(labels[i].item())
                    pred_label = int(preds[i].item())
                    if true_label >= probs.shape[1] or pred_label >= probs.shape[1]:
                        continue
                    if pred_label != true_label:
                        image_np = images[i].cpu().numpy().squeeze()
                        image_np = np.clip((image_np * 0.5) + 0.5, 0.0, 1.0)
                        pred_conf = float(probs[i, pred_label].item())
                        misclassified_examples.append((image_np, true_label, pred_label, pred_conf))

            all_probs.extend(probs.numpy())
            all_preds.extend(preds.numpy())
            all_targets.extend(labels.numpy())

    print(f"Evaluated samples: {len(all_targets)} (from total {total_eval_size})")

    all_probs = np.array(all_probs)      # [N, ckpt_classes]
    all_preds = np.array(all_preds)      # [N]
    all_targets = np.array(all_targets)  # [N]

    # Align eval set to classes that exist in the loaded checkpoint.
    eval_classes = min(num_classes, all_probs.shape[1])
    eval_names = names[:eval_classes]
    valid_mask = all_targets < eval_classes
    if not np.all(valid_mask):
        dropped = int((~valid_mask).sum())
        print(
            f"Warning: {dropped} samples have target labels outside checkpoint class range; "
            "excluding them from visualization metrics."
        )
    all_probs = all_probs[valid_mask, :eval_classes]
    all_preds = all_preds[valid_mask]
    all_targets = all_targets[valid_mask]

    if all_targets.size == 0:
        print("Error: No valid samples remain after class-range alignment.")
        return

    # Binarize targets for one-vs-rest ROC / PR (handles any num_classes)
    classes_range = list(range(eval_classes))
    targets_bin = label_binarize(all_targets, classes=classes_range)
    if eval_classes == 2:
        # label_binarize returns shape [N,1] for binary; expand to [N,2]
        targets_bin = np.hstack([1 - targets_bin, targets_bin])

    prob_cols = all_probs.shape[1]

    # Additional summary metrics
    top1_acc = (all_preds == all_targets).mean() * 100.0
    bal_acc = balanced_accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted")

    max_conf = all_probs.max(axis=1)
    confidence_mean = float(np.mean(max_conf) * 100.0)
    confidence_p90 = float(np.percentile(max_conf, 90) * 100.0)

    report_text = classification_report(
        all_targets,
        all_preds,
        labels=list(range(eval_classes)),
        target_names=eval_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        all_targets,
        all_preds,
        labels=list(range(eval_classes)),
        target_names=eval_names,
        output_dict=True,
        zero_division=0,
    )
    with open("classification_report.txt", "w", encoding="utf-8") as report_file:
        report_file.write(report_text)

    with open("treatment_support_table.csv", "w", encoding="utf-8") as support_csv:
        support_csv.write(
            "class,precision,recall,f1_score,support,risk_tier,suggested_action\n"
        )
        for class_name in eval_names:
            class_stats = report_dict.get(class_name, {})
            precision = float(class_stats.get("precision", 0.0))
            recall = float(class_stats.get("recall", 0.0))
            f1 = float(class_stats.get("f1-score", 0.0))
            support = int(class_stats.get("support", 0))
            tier = _risk_tier(precision, recall)
            action = _treatment_suggestion(class_name, tier)
            action = action.replace(",", ";")
            support_csv.write(
                f"{class_name},{precision:.4f},{recall:.4f},{f1:.4f},{support},{tier},{action}\n"
            )

    with open("treatment_support_table.txt", "w", encoding="utf-8") as support_txt:
        support_txt.write("Treatment Support Table (AI-assisted)\n")
        support_txt.write("-----------------------------------\n")
        for class_name in eval_names:
            class_stats = report_dict.get(class_name, {})
            precision = float(class_stats.get("precision", 0.0))
            recall = float(class_stats.get("recall", 0.0))
            f1 = float(class_stats.get("f1-score", 0.0))
            tier = _risk_tier(precision, recall)
            action = _treatment_suggestion(class_name, tier)
            support_txt.write(
                f"{class_name}: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}, "
                f"Risk={tier} | {action}\n"
            )

    # Top confusion pairs
    cm_counts = confusion_matrix(all_targets, all_preds, labels=list(range(eval_classes)))
    confusion_pairs = []
    for i in range(eval_classes):
        for j in range(eval_classes):
            if i != j and cm_counts[i, j] > 0:
                confusion_pairs.append((int(cm_counts[i, j]), eval_names[i], eval_names[j]))
    confusion_pairs.sort(reverse=True)
    with open("top_confusions.txt", "w", encoding="utf-8") as confusion_file:
        confusion_file.write("count,true_class,predicted_class\n")
        for count, true_cls, pred_cls in confusion_pairs[:20]:
            confusion_file.write(f"{count},{true_cls},{pred_cls}\n")

    # 4. Advanced dashboard layout
    sns.set_theme(style="whitegrid")
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 3, figsize=(30, 14))

    # --- GRAPH 1: Normalized Confusion Matrix ---
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(eval_classes)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    sns.heatmap(
        cm_norm,
        annot=True if eval_classes <= 12 else False,
        fmt='.2f',
        cmap='Blues',
        ax=axes[0, 0],
        xticklabels=eval_names,
        yticklabels=eval_names
    )
    axes[0, 0].set_title('Normalized Confusion Matrix', fontsize=15, fontweight='bold')
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].tick_params(axis='x', rotation=30)
    axes[0, 0].tick_params(axis='y', rotation=0)

    # --- GRAPH 2: Per-class ROC Curves (one-vs-rest, valid classes only) ---
    roc_aucs = []
    roc_plotted = 0
    for i in range(prob_cols):
        y_true = targets_bin[:, i]
        positives = y_true.sum()
        negatives = len(y_true) - positives
        if positives == 0 or negatives == 0:
            continue

        fpr, tpr, _ = roc_curve(y_true, all_probs[:, i])
        cls_auc = auc(fpr, tpr)
        roc_aucs.append(cls_auc)
        roc_plotted += 1
        axes[0, 1].plot(fpr, tpr, lw=2, color=colors[i % 10],
                        label=f'{eval_names[i]} (AUC={cls_auc:.2f})')

    macro_auc = float(np.mean(roc_aucs)) if roc_aucs else float("nan")
    axes[0, 1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0, 1].set_title(
        f'ROC Curves | Macro AUC = {macro_auc:.2f} | Plotted={roc_plotted}/{prob_cols}',
                         fontsize=15, fontweight='bold')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].legend(loc="lower right", fontsize=8)

    # --- GRAPH 3: Reliability Diagram (confidence calibration) ---
    correctness = (all_preds == all_targets).astype(int)
    frac_pos, mean_pred = calibration_curve(correctness, max_conf, n_bins=10, strategy="uniform")
    axes[0, 2].plot(mean_pred, frac_pos, marker='o', lw=2, label='Model')
    axes[0, 2].plot([0, 1], [0, 1], 'k--', lw=1, label='Perfectly calibrated')
    ece = float(np.mean(np.abs(frac_pos - mean_pred))) if len(frac_pos) else 0.0
    axes[0, 2].set_title(f'Reliability Diagram | ECE={ece:.4f}', fontsize=15, fontweight='bold')
    axes[0, 2].set_xlabel('Mean Predicted Confidence')
    axes[0, 2].set_ylabel('Empirical Accuracy')
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].legend(loc="lower right", fontsize=8)

    # --- GRAPH 4: Per-class Precision-Recall Curves (valid classes only) ---
    pr_plotted = 0
    pr_aps = []
    for i in range(prob_cols):
        y_true = targets_bin[:, i]
        if y_true.sum() == 0:
            continue

        precision, recall, _ = precision_recall_curve(y_true, all_probs[:, i])
        ap = average_precision_score(y_true, all_probs[:, i])
        pr_plotted += 1
        pr_aps.append(ap)
        axes[1, 0].plot(recall, precision, lw=2, color=colors[i % 10],
                        label=f'{eval_names[i]} (AP={ap:.2f})')
    macro_ap = float(np.mean(pr_aps)) if pr_aps else float("nan")
    axes[1, 0].set_title(
        f'Precision-Recall Curves | Macro AP={macro_ap:.2f} | Plotted={pr_plotted}/{prob_cols}',
        fontsize=15,
        fontweight='bold'
    )
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend(loc="upper right", fontsize=8)

    # --- GRAPH 5: Confidence Distribution + key summary metrics ---
    axes[1, 1].hist(max_conf * 100.0, bins=20, color='teal', alpha=0.85)
    axes[1, 1].set_title('Prediction Confidence Distribution', fontsize=15, fontweight='bold')
    axes[1, 1].set_xlabel('Max Predicted Probability (%)')
    axes[1, 1].set_ylabel('Sample Count')

    metrics_text = (
        f"Top-1 Accuracy: {top1_acc:.2f}%\n"
        f"Balanced Accuracy: {bal_acc:.2f}%\n"
        f"Macro F1: {macro_f1:.4f}\n"
        f"Weighted F1: {weighted_f1:.4f}\n"
        f"Macro AUC (valid): {macro_auc:.4f}\n"
        f"Macro AP (valid): {macro_ap:.4f}\n"
        f"ECE: {ece:.4f}\n"
        f"Mean Confidence: {confidence_mean:.2f}%\n"
        f"P90 Confidence: {confidence_p90:.2f}%"
    )
    axes[1, 1].text(
        0.97,
        0.97,
        metrics_text,
        transform=axes[1, 1].transAxes,
        va='top',
        ha='right',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray')
    )

    # --- GRAPH 6: Class Support and F1 ---
    class_support = np.array([int(report_dict[name]["support"]) for name in eval_names], dtype=float)
    class_f1 = np.array([float(report_dict[name]["f1-score"]) for name in eval_names], dtype=float)
    order = np.argsort(-class_support)
    ordered_names = [eval_names[i] for i in order]
    ordered_support = class_support[order]
    ordered_f1 = class_f1[order]

    ax_bar = axes[1, 2]
    ax_bar.bar(np.arange(len(ordered_names)), ordered_support, color='steelblue', alpha=0.8, label='Support')
    ax_bar.set_ylabel('Support (samples)')
    ax_bar.set_xlabel('Class')
    ax_bar.set_title('Class Support + F1 Score', fontsize=15, fontweight='bold')
    ax_bar.set_xticks(np.arange(len(ordered_names)))
    ax_bar.set_xticklabels(ordered_names, rotation=40, ha='right')

    ax_f1 = ax_bar.twinx()
    ax_f1.plot(np.arange(len(ordered_names)), ordered_f1, color='darkorange', marker='o', lw=2, label='F1')
    ax_f1.set_ylabel('F1 Score')
    ax_f1.set_ylim(0, 1)

    plt.suptitle(
        f"Federated Learning — {config.algorithm.upper()} | {eval_classes} Evaluated Classes",
        fontsize=18, y=1.02
    )
    plt.tight_layout()

    report_name = "federated_medical_report.png"
    plt.savefig(report_name, dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS: Report saved to '{report_name}'")
    print("Saved per-class report: 'classification_report.txt'")
    print("Saved top confusion pairs: 'top_confusions.txt'")
    print("Saved treatment support table: 'treatment_support_table.csv' and 'treatment_support_table.txt'")

    # Save a dedicated panel of top misclassified examples.
    if misclassified_examples:
        cols = 4
        rows = int(np.ceil(len(misclassified_examples) / cols))
        fig_mis, axes_mis = plt.subplots(rows, cols, figsize=(4 * cols, 3.8 * rows))
        axes_mis = np.array(axes_mis).reshape(rows, cols)

        for ax in axes_mis.ravel():
            ax.axis("off")

        for idx, (img, true_idx, pred_idx, conf) in enumerate(misclassified_examples):
            r, c = divmod(idx, cols)
            ax = axes_mis[r, c]
            ax.imshow(img, cmap="gray")
            true_name = eval_names[true_idx] if true_idx < len(eval_names) else str(true_idx)
            pred_name = eval_names[pred_idx] if pred_idx < len(eval_names) else str(pred_idx)
            ax.set_title(
                f"True: {true_name}\nPred: {pred_name} ({conf * 100:.1f}%)",
                fontsize=9,
            )
            ax.axis("off")

        fig_mis.suptitle("Top Misclassified Examples", fontsize=16, y=0.995)
        fig_mis.tight_layout()
        mis_name = "misclassified_examples.png"
        fig_mis.savefig(mis_name, dpi=250, bbox_inches='tight')
        print(f"Saved misclassified samples panel: '{mis_name}'")

    plt.show()


if __name__ == "__main__":
    generate_combined_report()