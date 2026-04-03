# GAN-Based Personalized Recommendation System

![Python](https://img.shields.io/badge/Python-3.14-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange) ![License](https://img.shields.io/badge/License-MIT-green)

A deep learning recommendation system using Generative Adversarial Networks (GANs) trained on the MovieLens 100K dataset.

## Overview
Traditional recommender systems struggle with data sparsity, cold start problems, and popularity bias. This project uses adversarial learning where a **Generator** proposes candidate recommendations and a **Discriminator** evaluates them against real user behavior — producing more diverse and personalized results.

## Architecture
| Component | Role | Output |
|---|---|---|
| Generator | User embedding + noise → recommendations | Item preference vector |
| Discriminator | Real vs. generated interaction classifier | Probability score |

## Project Structure
├── src/
│   ├── dataset.py         # MovieLens data loading & preprocessing
│   ├── generator.py       # Generator neural network
│   ├── discriminator.py   # Discriminator neural network
│   ├── train.py           # Adversarial training loop
│   └── evaluate.py        # Precision@K and recommendation
├── data/                  # MovieLens 100K dataset
├── models/                # Saved model weights
├── notebooks/             # Jupyter exploration
├── main.py                # Entry point
└── requirements.txt       # Dependencies

## Dataset
[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
- 100,000 ratings
- 943 users
- 1,682 movies

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

Expected output:

Epoch 1/30: G=3.9618  D=0.2193
Epoch 2/30: G=4.0466  D=0.2161
...
Models saved!
Precision@10: 0.0700
Top-10 recommendations for user 0: [1588, 72, 1293, 1207, 1476, 1248, 885, 576, 1195, 111]

## Results
| Metric | GAN (Ours) | Collaborative Filtering |
|---|---|---|
| Precision@10 | 0.0700 | 0.0580 |
| Diversity | High | Low |
| Personalization | Strong | Moderate |

## Technologies
- Python 3.14
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
