#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
** The Flow class **

* Last edited: 2020-06-01
* Created by: Stefan Bruche (TU Berlin)
"""
from aristopy.energySystemModel import EnergySystemModel
from aristopy.sourceSink import Source

class Flow:
    def __init__(self, commodity, link=None, var_name='commodity_name', **kwargs):
        pass

    # test mit property

if __name__ == '__main__':

    es = EnergySystemModel(number_of_time_steps=3)
    src = Source(es, 'src', 'HEAT')  # inlets=Flow('HEAT', link='')
    es.optimize(results_file=None)
    es.pyM.pprint()
