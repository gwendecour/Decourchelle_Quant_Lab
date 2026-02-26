from abc import ABC, abstractmethod
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

class FinancialInstrument(ABC):
    """
    Abstract base class. 
    Forces all options (Call, Phoenix...) to implement the same methods.
    """
    def __init__(self, **params):
        self.params = params
        
    @abstractmethod
    def price(self) -> float:
        pass

    @abstractmethod
    def greeks(self) -> dict:
        """Returns a dictionary e.g.: {'delta': 0.5, 'gamma': 0.02, ...}"""
        pass


class InstrumentFactory:
    """
    Factory class that creates the correct object based on user choice.
    """
    @staticmethod
    def create_instrument(instrument_type, **kwargs):
        from src.derivatives.pricing_model import EuropeanOption
        from src.derivatives.structured_products import PhoenixStructure
        
        if instrument_type in ["Call", "Put"]:
            return EuropeanOption(option_type=instrument_type.lower(), **kwargs)
        
        elif instrument_type == "Phoenix Autocall":
            return PhoenixStructure(**kwargs)
        
        else:
            raise ValueError(f"Unknown instrument: {instrument_type}")