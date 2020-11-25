import numpy as np
import pandas as pd

def match(x,y):
    """
    x = model output
    y = golden
    """
    x = set(range(x[0], x[1]+1))
    y = set(range(y[0], y[1]+1))
    matches = x.intersection(y)
    if matches:
        return True
    return False