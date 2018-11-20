#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All the available model ensembles
"""
from sklearn.neural_network import MLPClassifier
from fgclassifier.baseline import Baseline, Dummy

# Multi Layer Perceptron
MLP = MLPClassifier(
    hidden_layer_sizes=10,
)
