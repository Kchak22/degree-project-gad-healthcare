# degree-project-gad-healthcare
Code for the degree project by Karim Chakroun

# Bipartite Graph Anomaly Detection

A comprehensive framework for detecting anomalies in bipartite graphs with node attributes. This project implements and compares various anomaly detection models ranging from simple baselines to sophisticated graph neural network approaches.

## 📋 Overview

This project focuses on detecting anomalies in healthcare claims data represented as a bipartite graph between providers and members. It includes:

- Data loading and preprocessing utilities
- Multiple baseline and advanced anomaly detection models
- Evaluation tools and metrics
- Interactive notebooks for model comparison

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bipartite-graph-anomaly-detection.git
cd bipartite-graph-anomaly-detection

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.7+
- PyTorch 1.9+
- PyTorch Geometric
- scikit-learn
- pandas
- numpy
- matplotlib

## 🚀 Quick Start

1. **Prepare your data**: Place your data files in the `data/` directory:
   - `final_df.pickle`: Provider features
   - `final_members_df.pickle`: Member features
   - `df_descriptions.pickle`: Claims data linking providers and members

2. **Run the comparison notebook**:
   ```bash
   jupyter notebook notebooks/model_comparison.ipynb
   ```

3. **Train a specific model**:
   ```python
   from src.data.dataloader import load_member_features, load_provider_features, load_claims_data, prepare_hetero_data
   from src.models.main_model import BipartiteGraphAutoEncoder
   from src.utils.train_utils import train_model
   
   # Load data
   df_member_features, members_dataset = load_member_features()
   df_provider_features, providers_dataset = load_provider_features()
   df_edges = load_claims_data(members_dataset=members_dataset, providers_dataset=providers_dataset)
   
   # Prepare graph data
   data = prepare_hetero_data(df_member_features, df_provider_features, df_edges)
   
   # Initialize and train model
   model = BipartiteGraphAutoEncoder(
       in_dim_member=data['member'].x.size(1),
       in_dim_provider=data['provider'].x.size(1),
       edge_dim=1,
       hidden_dim_member=64,
       hidden_dim_provider=64,
       out_dim=32,
       num_layers=2
   )
   
   train_model(model, data, num_epochs=100)
   ```

## 💻 Project Structure

```
anomaly_detection/
│
├── data/                   # Data files
│   ├── final_df.pickle
│   ├── final_members_df.pickle
│   └── df_descriptions.pickle
│
├── src/                    # Source code
│   ├── data/
│   │   └── dataloader.py   # Data loading utilities
│   │
│   ├── models/
│   │   ├── main_model.py   # Main bipartite graph autoencoder
│   │   └── baseline_models.py # Baseline models
│   │
│   └── utils/
│       ├── eval_utils.py   # Evaluation metrics and plotting
│       └── train_utils.py  # Training functions
│
├── notebooks/             # Jupyter notebooks
│   └── model_comparison.ipynb # Model comparison notebook
│
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## 📚 Implemented Models

1. **Non-Graph Methods**
   - MLP Autoencoder: Simple autoencoder on node features
   - Isolation Forest: Outlier detection algorithm
   - PCA: Principal Component Analysis with reconstruction error

2. **Graph-Based Methods**
   - GCN Autoencoder: Graph Convolutional Network
   - GAT Autoencoder: Graph Attention Network
   - GraphSAGE Autoencoder: Graph SAmple and aggreGatE
   - Bipartite Graph Autoencoder (Our model): Specialized for bipartite graphs with node attributes

## 📊 Evaluation

The models are evaluated based on:

- Area Under ROC Curve (AUC)
- Average Precision (AP)
- Precision-Recall curves
- Anomaly score distributions

## 🛠️ Customization

### Adding New Models

To add a new model:

1. Create a new class in `src/models/baseline_models.py` or in a new file
2. Implement the required methods: `forward()` and `compute_anomaly_scores()`
3. Add your model to the comparison in `notebooks/model_comparison.ipynb`

### Using Your Own Data

To use your own data:

1. Prepare your data in a similar format to the provided pickle files
2. Modify the paths in `dataloader.py` or provide paths when calling the loading functions
3. Ensure your data has the necessary node features and edge information


## 👥 Contributors

- Karim Chakroun (@kchak22)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📬 Contact

For any questions or feedback, please open an issue or contact:

Karim Chakroun - chakroun@kth.se, karimchakroun2212@yahoo.fr

---
