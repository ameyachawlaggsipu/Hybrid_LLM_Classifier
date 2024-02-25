# Hybrid Text Classification Model

The Hybrid Text Classification Model combines the powerful sentence embedding capabilities of Sentence Transformers with the simplicity and effectiveness of a Logistic Regression classifier from scikit-learn. This package is designed to offer an easy-to-use interface for training and predicting text classifications, ideal for a wide range of NLP tasks.

## Features

- **Sentence Embedding**: Utilizes Sentence Transformers for generating rich text embeddings.
- **Classification**: Incorporates a Logistic Regression classifier for efficient text classification.
- **Easy Integration**: Designed for straightforward integration into existing Python projects.
- **Logging**: Includes logging for transparent monitoring of the training and prediction processes.
- **Customizable**: Supports customization of the underlying Sentence Transformer model and Logistic Regression classifier.

## Installation

Ensure you have Python 3.6+ installed on your system. You can then install the package using the following steps:

1. Clone the repository:

```bash
git clone https://github.com/ameyachawlaggsipu/Hybrid_LLM_Classifier/tree/main
```

2. Navigate to the cloned directory:

```bash
cd hybrid_model
```

3. Install the package:

```bash
pip install .
```

This will install the Hybrid Model package along with its dependencies.

## Quick Start

Here's how to get started with the Hybrid Model package:

### Initializing the Model

First, import and initialize the HybridModel with a pre-trained Sentence Transformer model ID.

```python
from hybrid_model.model import HybridModel

model_id = "all-MiniLM-L6-v2"  # Example Sentence Transformer model ID
hybrid_model = HybridModel(model_id=model_id)
```

### Training the Model

To train the model, provide a list of text samples and their corresponding labels.

```python
text_list = ["This is a positive example.", "This is a negative example."]
label_list = [1, 0]  # Example binary labels

# Train the model
hybrid_model.train(text_list, label_list)
```

### Making Predictions

Once trained, you can use the model to predict the classification of new text samples.

```python
predictions = hybrid_model.predict(["This is a test."])
print(predictions)
```

## Contributing

Contributions to improve the Hybrid Model package are welcome. Please feel free to fork the repository, make changes, and submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
