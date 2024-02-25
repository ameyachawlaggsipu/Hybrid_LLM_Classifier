from typing import List
import logging
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

class HybridModel:
    def __init__(self, model_id: str, random_state: int = 24):
        """
        Initializes the hybrid model with a SentenceTransformer(based on provided model_id) and Logistic Regression classifier.
        
        :param model_id: ID for the pre-trained SentenceTransformer model.
        :param random_state: Random state for reproducibility in the logistic regression classifier.
        """
        try:
            self.model = SentenceTransformer(model_id)
        except Exception as e:
            logging.error(f"Failed to load sentence transformer model with ID '{model_id}': {str(e)}")
            raise
        self.classifier = LogisticRegression(random_state=random_state)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def _prepare_training_data(self, text_list: List[str], label_list: List[int]) -> DataLoader:
        """
        Prepares and returns a DataLoader for the training data.
        
        :param text_list: List of text samples.
        :param label_list: Corresponding labels for the text samples.
        :return: DataLoader with the training examples.
        """
        train_examples = [InputExample(texts=[text], label=label) for text, label in zip(text_list, label_list)]
        return DataLoader(train_examples, shuffle=True, batch_size=10)
    
    def train(self, text_list: List[str], label_list: List[int], epochs: int = 10) -> None:
        """
        Trains the hybrid model using the provided texts and labels.
        
        :param text_list: List of text samples for training.
        :param label_list: Corresponding labels for the training samples.
        :param epochs: Number of epochs to train the model.
        """
        try:
            logging.info("Preparing training data...")
            train_dataloader = self._prepare_training_data(text_list, label_list)
            train_loss = losses.BatchAllTripletLoss(model=self.model)
            
            logging.info("Training sentence transformer model...")
            self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, optimizer_class=torch.optim.AdamW)
            
            logging.info("Encoding training data...")
            train_x = self.model.encode(text_list)
            train_y = np.array(label_list)
            
            logging.info("Training logistic regression classifier...")
            self.classifier.fit(train_x, train_y)
            logging.info("Model training completed successfully.")
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise

    def predict(self, text_list: List[str]) -> np.ndarray:
        """
        Predicts labels for a given list of texts.
        
        :param text_list: List of texts to predict.
        :return: Predicted labels for the input texts.
        """
        try:
            test_x = self.model.encode(text_list)
            predictions = self.classifier.predict(test_x)
            return predictions
        except NotFittedError:
            logging.error("The model has not been trained yet. Please train the model before prediction.")
            raise
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise

# Example usage
if __name__ == "__main__":
    model_id = "all-MiniLM-L6-v2"  # Example model ID
    hybrid_model = HybridModel(model_id=model_id)
    
    # Example data - Replace these with your actual data
    text_list = ["This is a positive example.", "This is a negative example."]
    label_list = [1, 0]  # Example binary labels
    
    # Train the model
    hybrid_model.train(text_list, label_list)
    
    # Predict with the model
    predictions = hybrid_model.predict(["This is a test."])
    print(predictions)
