# kchak22-degree-project-gad-healthcare

This repository contains the code for a degree project focused on detecting anomalies in healthcare claims data using bipartite graphs with node attributes. The framework implements a variety of anomaly detection techniques—from simple baselines to advanced graph neural network models—along with utilities for data loading, anomaly injection, evaluation, training, and interactive visualization.

## 📋 Overview

The project is centered on detecting anomalies in a healthcare setting by modeling claims data as a bipartite graph between providers and members. Key components include:

- **Data Handling & Preprocessing**: Load and preprocess healthcare data from pickle files.
- **Anomaly Injection**: Inject different types of anomalies (structural, feature-based, and healthcare-specific) into the graph with detailed tracking.
- **Models**: Implementations range from non-graph (e.g., MLP autoencoder, Isolation Forest, PCA) to graph-based methods (e.g., GCN, GAT, GraphSAGE) and a custom Bipartite Graph Autoencoder tailored for the problem.
- **Evaluation**: Tools for computing metrics such as ROC AUC, Average Precision, and for plotting precision-recall curves and anomaly score distributions.
- **Visualization**: Interactive visualizations of the graph and anomaly scores.
- **Notebooks**: Jupyter notebooks for graph visualization and model comparison.

## 🔧 Installation

Clone the repository and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/kchak22-degree-project-gad-healthcare.git
cd kchak22-degree-project-gad-healthcare

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Requirements

    Python 3.7+
    PyTorch 1.9+
    PyTorch Geometric (>=2.0.0)
    PyTorch Lightning (>=1.5.0)
    scikit-learn
    pandas
    numpy
    matplotlib
    jupyter
    networkx
    tqdm

🚀 Quick Start

    Prepare Your Data
    Place your data files (e.g., final_df.pickle, final_members_df.pickle, df_descriptions.pickle) in a folder (e.g., a data/ directory located outside or alongside the repository). The data loading functions in src/data/dataloader.py expect paths like ../../data/final_df.pickle.

    Explore Notebooks
    Use the provided notebooks to visualize graphs and compare models:
        Graph Visualization & Anomalies
        Model Comparison

    Launch Jupyter Notebook:

jupyter notebook notebooks/model_comparison.ipynb

Train a Model
Below is an example of loading data, preparing the graph, and training the custom Bipartite Graph Autoencoder:

    from src.data.dataloader import load_member_features, load_provider_features, load_claims_data, prepare_hetero_data
    from src.models.main_model import BipartiteGraphAutoEncoder
    from src.utils.train_utils import train_model

    # Load data (ensure file paths are correct)
    df_member_features, members_dataset = load_member_features("../../data/final_members_df.pickle")
    df_provider_features, providers_dataset = load_provider_features("../../data/final_df.pickle")
    df_edges = load_claims_data("../../data/df_descriptions.pickle", members_dataset, providers_dataset)

    # Prepare the HeteroData graph
    data = prepare_hetero_data(df_member_features, df_provider_features, df_edges)

    # Initialize and train the Bipartite Graph Autoencoder
    model = BipartiteGraphAutoEncoder(
        in_dim_member=data['member'].x.size(1),
        in_dim_provider=data['provider'].x.size(1),
        edge_dim=1,
        hidden_dim_member=64,
        hidden_dim_provider=64,
        out_dim=32,
        num_layers=2,
        dropout=0.5
    )

    train_model(model, data, num_epochs=100)

💻 Project Structure

kchak22-degree-project-gad-healthcare/
├── requirements.txt           # Project dependencies
├── setup.py                   # Package setup (if needed)
├── docs/                      # Project documentation (design, etc.)
│   └── design.md
├── notebooks/                 # Jupyter notebooks for experiments and visualization
│   ├── graph_vizualization_anomalies.ipynb
│   └── model_comparison.ipynb
├── src/                       # Source code
│   ├── __init__.py
│   ├── data/                  # Data utilities and anomaly injection functions
│   │   ├── __init__.py
│   │   ├── anomaly_injection.py
│   │   └── dataloader.py
│   ├── evaluation/            # Evaluation metrics and functions
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── models/                # Model implementations
│   │   ├── __init__.py
│   │   ├── baseline_models.py # Baseline models and non-graph approaches
│   │   └── main_model.py      # Bipartite Graph Autoencoder and custom layers
│   └── utils/                 # Utility functions for training and visualization
│       ├── __init__.py
│       ├── eval_utils.py      # Evaluation plots and comparison functions
│       ├── train_utils.py     # Training routines
│       └── vizualize.py       # Graph and anomaly visualization tools
└── tests/                     # Unit tests
    └── __init__.py

📚 Implemented Models
Non-Graph Methods

    MLP Autoencoder: A simple autoencoder operating solely on node attributes.
    Sklearn Baselines: Isolation Forest, One-Class SVM, and PCA-based anomaly detection.

Graph-Based Methods

    GCN Autoencoder: Uses Graph Convolutional Networks for feature reconstruction.
    GAT Autoencoder: Leverages attention mechanisms.
    GraphSAGE Autoencoder: Aggregates neighborhood information using GraphSAGE.
    Bipartite Graph Autoencoder (Custom): A specialized model that uses bipartite attention layers and is designed for heterogeneous graphs with node attributes.

📊 Evaluation

Anomaly detection models are evaluated using:

    ROC AUC (Area Under the ROC Curve)
    Average Precision (AP)
    Precision-Recall Curves
    Anomaly Score Distributions

Evaluation routines are available in src/evaluation/metrics.py and src/utils/eval_utils.py.
🛠️ Customization
Extending the Framework

    Adding New Models:
    Create a new model class in the src/models/ directory (or extend the baseline models in baseline_models.py) by implementing the forward() and compute_anomaly_scores() methods. Update the notebooks for comparison if needed.

    Data and Anomaly Injection:
    The data loading functions and anomaly injection methods (with tracking) are found in src/data/dataloader.py and src/data/anomaly_injection.py. To use your own data, modify these functions or update the file paths accordingly.

    Visualizations and Evaluation:
    Customize evaluation metrics, ROC/PR curves, and graph visualizations via the scripts in src/utils/.

👥 Contributors

    Karim Chakroun (@kchak22)

🤝 Contributing

Contributions are welcome! To contribute:

    Fork the repository.
    Create a feature branch:
    git checkout -b feature/your-feature
    Commit your changes:
    git commit -m "Add some feature"
    Push to the branch:
    git push origin feature/your-feature
    Open a Pull Request.

📬 Contact

For questions or feedback, please open an issue or contact:

    Karim Chakroun
    Email: chakroun@kth.se, karimchakroun2212@yahoo.fr