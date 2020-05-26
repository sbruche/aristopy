#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
** The Series class **

* Last edited: 2020-06-01
* Created by: Stefan Bruche (TU Berlin)
"""
import pandas as pd
import numpy as np


class Series:
    def __init__(self, name, data, weighting_factor=1.0):
        """
        The Series class is used to add time series data to the model.
        Time series data can be required in the Source class to implement data
        for keyword arguments 'commodity_rate_min',´'commodity_rate_max',
        'commodity_rate_fix'. Additionally, they might be needed to a add time-
        dependent commodity cost or revenues, or generally for scripting of
        user expressions (added via 'time_series_data' argument).

        :param name: Name of the time series data instance (string). Can be
            used for scripting of user expressions.
        :param data: Series data (type: list, dict, numpy array, pandas Series)
        :param weighting_factor: Weighting factor to use for the series
            aggregation (int or float, default: 1.0)
        """

        # Check correctness of input types
        assert isinstance(name, str), 'Expected "name" as a string'
        assert isinstance(data, (list, dict, np.ndarray, pd.Series)), \
            'Expected "data" with type list, dict, numpy array or pandas Series'
        assert isinstance(weighting_factor, (int, float)), \
            'Expected "weighting_factor" as integer or float'
        assert 0 <= weighting_factor <= 1, \
            'Expected weighting factor value between 0 and 1'

        self.name = name
        self.data = data
        self.weighting_factor = weighting_factor


if __name__ == '__main__':
    series = Series(name='my_data', data=[1, 2, 3], weighting_factor=0.8)

    for key, val in series.__dict__.items():
        print(key, val)
