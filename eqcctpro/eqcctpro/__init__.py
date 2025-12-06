# eqcctpro/__init__.py
from .functionality import (
    RunEQCCTPro,
    EvaluateSystem,
    OptimalCPUConfigurationFinder,
    OptimalGPUConfigurationFinder,
)
__all__ = [
    "RunEQCCTPro",
    "EvaluateSystem",
    "OptimalCPUConfigurationFinder",
    "OptimalGPUConfigurationFinder",
]
__version__ = "0.6.7"
