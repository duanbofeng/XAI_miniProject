from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TensorGraph:
    relation_edges: list[torch.Tensor]
    num_nodes: int
    num_relations: int

    def to(self, device: torch.device) -> "TensorGraph":
        return TensorGraph(
            relation_edges=[edge_index.to(device) for edge_index in self.relation_edges],
            num_nodes=self.num_nodes,
            num_relations=self.num_relations,
        )


def build_tensor_graph(
    relation_groups: list[list[tuple[int, int]]],
    num_nodes: int,
    device: torch.device,
) -> TensorGraph:
    edge_tensors = []
    for group in relation_groups:
        if group:
            edge_tensors.append(torch.tensor(group, dtype=torch.long, device=device).t().contiguous())
        else:
            edge_tensors.append(torch.empty((2, 0), dtype=torch.long, device=device))
    return TensorGraph(
        relation_edges=edge_tensors,
        num_nodes=num_nodes,
        num_relations=len(relation_groups),
    )


class RGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int, dropout: float) -> None:
        super().__init__()
        self.relation_weights = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_weights)
        nn.init.xavier_uniform_(self.self_loop.weight)

    def forward(self, x: torch.Tensor, graph: TensorGraph) -> torch.Tensor:
        out = self.self_loop(x)
        for relation_id, edge_index in enumerate(graph.relation_edges):
            if edge_index.numel() == 0:
                continue
            src, dst = edge_index
            messages = x[src] @ self.relation_weights[relation_id]
            degree = torch.bincount(dst, minlength=graph.num_nodes).clamp(min=1).to(messages.dtype)
            messages = messages / degree[dst].unsqueeze(1)
            out.index_add_(0, dst, messages)
        out = out + self.bias
        return self.dropout(F.relu(out))


class RGCNClassifier(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        num_classes: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.layer1 = RGCNLayer(embedding_dim, hidden_dim, num_relations, dropout)
        self.layer2 = RGCNLayer(hidden_dim, hidden_dim, num_relations, dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.xavier_uniform_(self.node_embeddings.weight)

    def forward(self, graph: TensorGraph) -> torch.Tensor:
        node_ids = torch.arange(graph.num_nodes, device=self.node_embeddings.weight.device)
        x = self.node_embeddings(node_ids)
        x = self.layer1(x, graph)
        x = self.layer2(x, graph)
        return self.classifier(x)
