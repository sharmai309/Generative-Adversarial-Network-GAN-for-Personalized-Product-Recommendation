# Generative Adversarial Network (GAN) for Personalized Product Recommendation

A deep learning recommendation system using GANs trained on MovieLens 100K dataset.

## Overview
Traditional recommender systems struggle with data sparsity and popularity bias. This project uses adversarial learning where a **Generator** proposes recommendations and a **Discriminator** evaluates them against real user behavior.

## Architecture
| Component | Description |
|---|---|
| Generator | User embedding + noise → item preference vector |
| Discriminator | Classifies real vs. generated user-item pairs |
| Dataset | MovieLens 100K (943 users, 1,682 movies) |

## Project Structure

├── src/
│   ├── dataset.py       # Data loading
│   ├── generator.py     # Generator network
│   ├── discriminator.py # Discriminator network
│   ├── train.py         # Training loop
│   └── evaluate.py      # Metrics
├── data/                # MovieLens dataset
├── models/              # Saved weights
├── main.py              # Entry point
└── requirements.txt

## Setup
```bash
git clone https://github.com/sharmai309/Generative-Adversarial-Network-GAN-for-Personalized-Product-Recommendation.git
cd Generative-Adversarial-Network-GAN-for-Personalized-Product-Recommendation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

## Results
| Metric | Score |
|---|---|
| Precision@10 | 0.0700 |
| Diversity | High |

## Technologies
Python · PyTorch · Pandas · NumPy · Scikit-learn

## License
MIT
