# 🧠 Temporal Link Prediction with Graph Neural Networks

This project focuses on solving the **Temporal Link Prediction** problem using **Graph Neural Networks (GNNs)**. We implemented and compared multiple GNN models to predict whether an edge (connection) will exist between two nodes within a specified time window.

---

## 📌 Project Description

In temporal graphs, the structure evolves over time. Traditional link prediction ignores time dynamics, whereas **Temporal Link Prediction** incorporates them. This is essential for real-world applications such as:

- Predicting future interactions in social networks
- Recommending items in e-commerce
- Modeling temporal user behavior

---

## 📂 Dataset Description

Data files are accessed from Google Drive and consist of:

- `node_features_sampled.csv`: Feature vectors of nodes
- `edges_train_A.csv`: Temporal edge connections for training
- `input_A.csv`: Test edges for link prediction with labels
- *(Optionally used)* `input_A_final.csv`: Test edges without labels for inference only

---

## 🏗️ Project Pipeline

### ✅ Step-by-step Workflow in the Notebook:

1. **Load & preprocess data** from CSV files
2. **Normalize node features** and handle missing data
3. **Map original node IDs** to new indices
4. **Create PyTorch Geometric Data object**
5. **Define two GNN models**: `GraphSAGE` and `GCN`
6. **Train both models** on the same data
7. **Optimize threshold** for binary classification using F1-score
8. **Evaluate models** using AUC, Accuracy, and F1-score
9. **Visualize model performance** via bar chart
10. **Export predictions** to `.csv` files

---

## 🧪 Models Implemented

| Model      | Description                                 |
|------------|---------------------------------------------|
| **GraphSAGE** | Samples neighbors and aggregates features |
| **GCN**       | Applies normalized graph convolution      |

Each model is trained for 100 epochs and evaluated under identical conditions.

---

## 📈 Evaluation Metrics

- **AUC (Area Under Curve)**: Measures quality of probability estimates
- **Accuracy**: Correct predictions over total predictions
- **F1 Score**: Harmonic mean of Precision and Recall
- **Threshold Optimization**: Best threshold selected by maximizing F1

---

## 📊 Results Visualization

A bar chart compares AUC and Accuracy of both models:

- GraphSAGE vs GCN
- Visual insight into model performance
- Easy to interpret and report

---

## 📁 Output Files

- `output_A_sage.csv`: Predictions by GraphSAGE
- `output_A_gcn.csv`: Predictions by GCN

Each CSV contains:
- `src`, `dst`, `edge_type`, `start`, `end`, and `predicted_probability`

---

## 🚀 Requirements

- Python 3.8+
- Google Colab (preferred)
- Libraries: `torch`, `torch_geometric`, `sklearn`, `pandas`, `matplotlib`, `numpy`

> All libraries are loaded within the notebook. If running locally, ensure dependencies are installed.

---

## 👨‍💻 Author

**Yağmur Sina**  
Computer Engineering Student  
This project was developed as part of the course **CSE 474 / CSE 5074 – Social Network Analysis**.

---

## 📌 Notes

- All steps are written in a modular and explainable Jupyter Notebook format.
- Each section is labeled (Step 1, Step 2, ...) with a detailed Markdown explanation.
