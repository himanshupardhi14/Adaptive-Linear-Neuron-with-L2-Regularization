#  Adaptive Linear Neuron with L2 RegularizationAdaline Classifier with Regularization

This project implements the **Adaline (Adaptive Linear Neuron)** algorithm from scratch using NumPy. It includes L2 regularization, visualizes training loss, and evaluates performance on a synthetic binary classification dataset.

##  What is Adaline?

Adaline is a simple neural network model that uses linear activation and is trained using gradient descent. It's one of the foundational algorithms in machine learning and is ideal for binary classification problems.

---

##  Features

- Custom Adaline implementation
- L2 regularization support
- Synthetic dataset generation using `scikit-learn`
- Standardization of input features
- Accuracy evaluation and metrics report
- Loss visualization over epochs
- Accuracy bar chart

---

##  Dataset

The dataset is synthetically generated using `make_classification()` from scikit-learn:

- **Samples**: 1000
- **Features**: 2 (both informative)
- **Classes**: 2 (binary classification)

---

##   Model Details

###  Hyperparameters

- **Learning Rate**: `0.01`
- **Epochs**: `100`
- **Regularization Strength**: `0.01` (L2)

###  Loss Function

- Mean Squared Error (MSE) with L2 regularization

---

##  Visualizations

- **Loss Curve**: Tracks loss across training epochs
- **Accuracy Bar Chart**: Displays final model accuracy on test data

---

##  Evaluation Metrics

- **Accuracy**
- **Confusion Matrix**
- **Precision, Recall, F1-Score** (via `classification_report`)

---

Install dependencies:

```bash
pip install numpy matplotlib scikit-learn


