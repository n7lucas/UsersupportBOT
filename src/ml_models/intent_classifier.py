
import numpy as np
from src.ml_models.base_classifier import BaseLogisticClassifier
class IntentClassifier(BaseLogisticClassifier):

    def __init__(self):
        """ Model responsible for classify User Intent"""
        super().__init__()
        