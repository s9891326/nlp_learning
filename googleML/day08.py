import pandas as pd
import numpy as np

# print(pd.__version__)

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 123])  # 少一個會怎樣? -> NaN
cities = pd.DataFrame({'City name': city_names, 'Population': population})

# california_housing_dataframe = pd.read_csv(
#     "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
# print(california_housing_dataframe.describe())
# print(california_housing_dataframe.hist('housing_median_age'))

# Dataframe是由rows跟named columns組成，而Series則是指單一一欄
# sub-dataframe
# print(type(cities[["City name"]]))
# print(cities[["City name"]])

# Series
# print(type(cities["City name"]))
# print(cities["City name"])

# # 取populatioin series 的log運算
# np.log(population)
#
# # Series 每個除以1000
# population / 1000
#
# # 每個element做個別運算(val > 1000000)
# population.apply(lambda val: val > 1000000)

cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
# Populatioin density不存在在原本的Cities dataframe裡
cities['Population density'] = cities['Population'] / cities['Area square miles']
# print(cities)

cities["is wide and has san name"] = (cities["Area square miles"] > 50) &\
                                     cities["City name"].apply(lambda name: name.startswith('San'))

# print(cities)
#
# print(city_names.index)
# print(cities.index)
#
# print(cities.reindex(np.random.permutation(cities.index)))
# print(cities.reindex(3, 2, 1, 0))


"""
first steps with tensorflow
- linearRegressor
- Predict
- Evaluate with Root Mean Squared Error
-  Improve hyperparameters
"""
import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import logging
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


logger = logging.getLogger()
