from hybrid_model.model import HybridModel
import pytest

def test_initialization():
    model_id = "all-MiniLM-L6-v2"
    model = HybridModel(model_id=model_id)
    assert model is not None, "Failed to initialize HybridModel"

def test_training():
    model_id = "all-MiniLM-L6-v2"
    model = HybridModel(model_id=model_id)
    text_list = ["This is a positive example.", "This is a negative example."]
    label_list = [1, 0]
    try:
        model.train(text_list, label_list, epochs=1)
    except Exception as e:
        pytest.fail(f"Training failed with exception: {e}")

def test_prediction():
    model_id = "all-MiniLM-L6-v2"
    model = HybridModel(model_id=model_id)
    text_list = ["This is a positive example.", "This is a negative example."]
    label_list = [1, 0]
    model.train(text_list, label_list, epochs=1)
    try:
        predictions = model.predict(["This is a test."])
        assert len(predictions) == 1, "Prediction failed or returned unexpected number of results"
    except Exception as e:
        pytest.fail(f"Prediction failed with exception: {e}")

def test_prediction_without_training():
    model_id = "all-MiniLM-L6-v2"
    model = HybridModel(model_id=model_id)
    with pytest.raises(Exception):
        _ = model.predict(["This should fail."])
