"""Q-FAR: fairness- and risk-aware microfinance allocation as a QUBO (synthetic simulation study)."""
__version__ = "2.0.0"

from .pipeline import AllocationResult, run_allocation

__all__ = ["__version__", "AllocationResult", "run_allocation"]
