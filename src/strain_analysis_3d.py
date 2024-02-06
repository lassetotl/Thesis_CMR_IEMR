# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:12:08 2024

@author: lassetot

we collect generated global radial strain (GRS) curves and plot them together.
make sure that all used data were generated using the same setup parameters!!

curve analysis parameters collected and used to construct a pandas dataframe
for correlation analysis
"""

import os, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ComboDataSR_2D import ComboDataSR_3D
from scipy.integrate import cumtrapz
from util import running_average, drop_outliers_IQR
import pandas 
import seaborn as sns