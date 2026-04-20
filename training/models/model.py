from models.trocr_model import TrOCRModel
from models.retrieval_model import RetrievalModel


def get_model(model_type):
    if model_type == "htr":
        return TrOCRModel()
    elif model_type == "retrieval":
        return RetrievalModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")