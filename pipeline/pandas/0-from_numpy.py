#!/usr/bin/env python3
"""
documented
"""
import pandas as pd


def from_numpy(array):
    """
    Docstring for from_numpy

    :param array: Description
    """
    num_columns = array.shape[1]
    columns = [chr(ord('A') + i) for i in range(num_columns)]
    df = pd.DataFrame(array, columns=columns)
    return df
