import torch

def precision_at_k(G, dataset, k=10, n_users=50):
    G.eval()
    hits = 0
    with torch.no_grad():
        for uid in range(min(n_users, dataset.n_users)):
            user_tensor = torch.tensor([uid])
            scores = G(user_tensor).squeeze()
            top_k = scores.topk(k).indices.tolist()
            actual = dataset.items[dataset.users == uid].tolist()
            hits += len(set(top_k) & set(actual))
    return hits / (n_users * k)

def recommend(G, user_id, dataset, k=10):
    G.eval()
    with torch.no_grad():
        scores = G(torch.tensor([user_id])).squeeze()
        top_k = scores.topk(k).indices.tolist()
    return top_k