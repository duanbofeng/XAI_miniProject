from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from xai_miniproject.config import Config
from xai_miniproject.data import build_graph_data, build_split_tensors, relation_edge_groups
from xai_miniproject.metrics import accuracy, macro_f1
from xai_miniproject.model import RGCNClassifier, build_tensor_graph
from xai_miniproject.utils import ensure_dir, seed_everything, short_uri, write_json


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def run_training(config: Config) -> dict[str, object]:
    seed_everything(config.project.seed)
    output_dir = ensure_dir(config.project.artifacts_dir)
    data = build_graph_data(config.dataset)
    splits = build_split_tensors(data, config.dataset)
    device = choose_device(config.model.device)
    tensor_graph = build_tensor_graph(
        relation_edge_groups(data),
        num_nodes=data.num_nodes,
        device=device,
    )

    train_idx = torch.tensor(splits["train_idx"], dtype=torch.long, device=device)
    train_labels = torch.tensor(splits["train_labels"], dtype=torch.long, device=device)
    test_idx = torch.tensor(splits["test_idx"], dtype=torch.long, device=device)
    test_labels = torch.tensor(splits["test_labels"], dtype=torch.long, device=device)

    model = RGCNClassifier(
        num_nodes=data.num_nodes,
        num_relations=data.num_relations,
        num_classes=data.num_classes,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
    )

    history: list[dict[str, float]] = []

    for epoch in range(1, config.model.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(tensor_graph)
        loss = F.cross_entropy(logits[train_idx], train_labels)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(tensor_graph)
            train_pred = logits[train_idx].argmax(dim=1).cpu().tolist()
            test_pred = logits[test_idx].argmax(dim=1).cpu().tolist()
            train_true = train_labels.cpu().tolist()
            test_true = test_labels.cpu().tolist()
            train_acc = accuracy(train_true, train_pred)
            test_acc = accuracy(test_true, test_pred)
            train_f1 = macro_f1(train_true, train_pred, list(range(data.num_classes)))
            test_f1 = macro_f1(test_true, test_pred, list(range(data.num_classes)))

        if epoch == 1 or epoch % config.model.log_every == 0 or epoch == config.model.epochs:
            row = {
                "epoch": float(epoch),
                "loss": float(loss.item()),
                "train_accuracy": train_acc,
                "train_macro_f1": train_f1,
                "test_accuracy": test_acc,
                "test_macro_f1": test_f1,
            }
            history.append(row)
            print(
                f"epoch={epoch:03d} loss={loss.item():.4f} "
                f"train_acc={train_acc:.3f} test_acc={test_acc:.3f} "
                f"train_f1={train_f1:.3f} test_f1={test_f1:.3f}"
            )

    model.eval()
    with torch.no_grad():
        logits = model(tensor_graph)
        probabilities = torch.softmax(logits, dim=1)
        all_target_idx = torch.tensor(splits["all_idx"], dtype=torch.long, device=device)
        all_pred_ids = probabilities[all_target_idx].argmax(dim=1).cpu().tolist()
        all_conf = probabilities[all_target_idx].max(dim=1).values.cpu().tolist()
        train_pred = logits[train_idx].argmax(dim=1).cpu().tolist()
        test_pred = logits[test_idx].argmax(dim=1).cpu().tolist()

    train_true = train_labels.cpu().tolist()
    test_true = test_labels.cpu().tolist()
    metrics = {
        "dataset": config.dataset.name,
        "device": str(device),
        "num_nodes": data.num_nodes,
        "num_edges": len(data.edges),
        "num_relations": data.num_relations,
        "num_classes": data.num_classes,
        "train_accuracy": accuracy(train_true, train_pred),
        "train_macro_f1": macro_f1(train_true, train_pred, list(range(data.num_classes))),
        "test_accuracy": accuracy(test_true, test_pred),
        "test_macro_f1": macro_f1(test_true, test_pred, list(range(data.num_classes))),
        "history": history,
    }

    predictions = _prediction_frame(config, data, all_pred_ids, all_conf)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    write_json(output_dir / "metrics.json", metrics)
    write_json(
        output_dir / "label_mapping.json",
        {
            "id_to_label": data.id_to_label,
            "label_to_id": data.label_to_id,
            "label_names": config.dataset.label_names,
        },
    )
    write_json(
        output_dir / "graph_metadata.json",
        {
            "id_to_node": data.id_to_node,
            "id_to_relation": data.id_to_relation,
            "excluded_predicates": sorted(config.dataset.exclude_predicates),
        },
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config_path": str(config.path),
            "metrics": metrics,
        },
        output_dir / "model.pt",
    )
    print(f"Saved artifacts to {output_dir}")
    return metrics


def _prediction_frame(
    config: Config,
    data,
    pred_ids: list[int],
    confidences: list[float],
) -> pd.DataFrame:
    split_by_entity: dict[str, str] = {
        row[config.dataset.entity_column]: "train" for _, row in data.train_df.iterrows()
    }
    split_by_entity.update(
        {row[config.dataset.entity_column]: "test" for _, row in data.test_df.iterrows()}
    )

    rows = []
    for offset, (_, row) in enumerate(data.all_labels_df.iterrows()):
        entity = row[config.dataset.entity_column]
        true_label = row[config.dataset.label_column]
        pred_label = data.id_to_label[pred_ids[offset]]
        rows.append(
            {
                "entity_uri": entity,
                "entity_name": short_uri(entity),
                "true_label_uri": true_label,
                "true_label_name": config.dataset.label_names.get(true_label, short_uri(true_label)),
                "pred_label_uri": pred_label,
                "pred_label_name": config.dataset.label_names.get(pred_label, short_uri(pred_label)),
                "confidence": confidences[offset],
                "split": split_by_entity.get(entity, "unknown"),
                "correct": true_label == pred_label,
            }
        )
    return pd.DataFrame(rows)


def load_metrics(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)
