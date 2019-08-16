#!/usr/bin/python

import torch
import numpy as np
import pandas as pd

import MInteractionModel

# Create object of model class
mod = MInteractionModel.MInteractionModel(order=3)

# Add dataset
mod.add_dataset("data/referenz_data.csv")

# Fit model to data
mod.Q = mod.torch_optimize(5000)

# Check how the data frequiencies fit to the prediction of the model
mod.modeltest()
