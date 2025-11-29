"""
Derivatives Features Module.

Features baseadas em dados de derivativos da Binance Futures:
- Funding Rate
- Open Interest
"""
from v2.src.features.derivatives.funding_features import FundingRateFeatures
from v2.src.features.derivatives.open_interest_features import OpenInterestFeatures

__all__ = ['FundingRateFeatures', 'OpenInterestFeatures']
