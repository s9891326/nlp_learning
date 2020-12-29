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
import numpy as np
import pandas as pd

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from absl import app
from absl import flags
from absl import logging
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# import data
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

# Prepare to run SGD
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
# Scale median_house_value values
california_housing_dataframe["median_house_value"] /= 1000.0

print(california_housing_dataframe.columns)

# Define the input feature: total_rooms.
my_feature = california_housing_dataframe[["total_rooms"]]  # dataframe

# Configure a numeric feature column for total_rooms.
feature_columns = [tf.feature_column.numeric_column("total_rooms")]  # [NumericColumn(key='total_rooms', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)]
# print(feature_columns)

targets = california_housing_dataframe["median_house_value"]

# Use gradient descent as the optimizer for training the model.
my_optimizer = tf.keras.optimizers.SGD(learning_rate=0.0000001, clipnorm=5.0)

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature

    Args:
        features:
        targets:
        batch_size:
        shuffle:
        num_epochs:
    Returns:
        Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays

    print(dict(features).items())
    # print(dict(features).items()[0])
    # print(dict(features).items()[0][1])
    # print(np.array(dict(features).items()[0][1]))

    features = {key: np.array(value) for key, value in dict(features).items()}
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def draw_picture():
    sample = california_housing_dataframe.sample(n=300)
    # Get the min and max total_rooms values.
    x_0 = sample["total_rooms"].min()
    x_1 = sample["total_rooms"].max()

    # Retrieve the final weight and bias generated during training.
    weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

    # Get the predicted median_house_values for the min and max total_rooms values.
    y_0 = weight * x_0 + bias
    y_1 = weight * x_1 + bias

    # Plot our regression line from (x_0, y_0) to (x_1, y_1).
    plt.plot([x_0, x_1], [y_0, y_1], c='r')

    # Label the graph axes.
    plt.ylabel("median_house_value")
    plt.xlabel("total_rooms")

    # Plot a scatter plot from our data sample.
    plt.scatter(sample["total_rooms"], sample["median_house_value"])

    # Display graph.
    plt.show()


if __name__ == '__main__':
    _ = linear_regressor.train(
        input_fn=lambda: my_input_fn(my_feature, targets),
        steps=100
    )

    prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)
    predictions = linear_regressor.predict(input_fn=prediction_input_fn)
    predictions = np.array([item['predictions'][0] for item in predictions])

    # Print Mean Squared Error and Root Mean Squared Error.
    mean_squared_error = metrics.mean_squared_error(predictions, targets)
    root_mean_squared_error = math.sqrt(mean_squared_error)
    print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
    print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)
    draw_picture()
