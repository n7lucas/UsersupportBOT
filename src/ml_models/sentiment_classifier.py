from src.ml_models.base_classifier import BaseLogisticClassifier


class SentimentClassifier(BaseLogisticClassifier):

    def __init__(self, **kwargs):
        """Model responsible for classify user sentiment"""
        super().__init__(**kwargs)