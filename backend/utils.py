import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any




def safe_convert_to_python(value):
    """Convert pandas/numpy types to native Python types for JSON serialization."""
    if value is None:
        return None
    
    # Handle numpy arrays and pandas Series BEFORE pd.isna()
    if isinstance(value, np.ndarray):
        return value.tolist()
    
    if isinstance(value, pd.Series):
        return value.tolist()
    
    # Handle NaN for scalars
    if isinstance(value, float) and np.isnan(value):
        return None
    
    # Handle pandas NA for scalars
    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass
    
    # Handle numpy scalar types
    if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    
    # Handle lists recursively
    if isinstance(value, list):
        return [safe_convert_to_python(item) for item in value]
    
    # Handle dicts recursively
    if isinstance(value, dict):
        return {k: safe_convert_to_python(v) for k, v in value.items()}
    
    return value