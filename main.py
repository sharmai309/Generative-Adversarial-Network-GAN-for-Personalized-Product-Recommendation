from src.train import train
from src.evaluate import precision_at_k, recommend

if __name__ == '__main__':
    G, D, dataset = train(epochs=30, batch_size=64, embed_dim=128)
    
    p = precision_at_k(G, dataset, k=10)
    print(f"\nPrecision@10: {p:.4f}")
    
    recs = recommend(G, user_id=0, dataset=dataset, k=10)
    print(f"Top-10 recommendations for user 0: {recs}")