import torch


def compute_accuracy(outputs, targets):
    preds = torch.argmax(outputs, dim=1)
    return (preds == targets).float().mean().item()


def compute_recall_at_k(embeddings, targets, k=5):
    n = embeddings.shape[0]

    if n <= 1:
        return 1.0

    sim = torch.matmul(embeddings, embeddings.T)

    actual_k = min(k + 1, n)
    topk = torch.topk(sim, k=actual_k, dim=1).indices

    # 去掉自己
    if actual_k > 1:
        topk = topk[:, 1:]
    else:
        return 1.0

    hits = 0
    for i in range(len(targets)):
        retrieved_targets = targets[topk[i]]
        if targets[i] in retrieved_targets:
            hits += 1

    return hits / len(targets)