import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# machine learning
from tensorflow import keras
from keras import layers

# Linear regression is a popular algorithm in machine learning used for predicting 
# a numerical target variable based on one or more input features. It assumes a 
# linear relationship between the input features and the target variable and
# tries to find the best fit line that minimizes the distance between the predicted 
# and actual target values.

# use pandas to import data into a pandas data frame
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
COLUMN_NAMES = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=COLUMN_NAMES, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()

# unknown values
dataset.isna().sum()

# drop unknown values
dataset = dataset.dropna()

# change 'Origin' to country
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

# The "Origin" column is now categorical, not numeric. So the next step is to one-hot encode the values in the column with pd.get_dummies.
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

# Split the data into training and test sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Separate the target value—the "label"—from the features. 
# This label (MPG) is the value that you will train the model to predict.
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# normalize data for horsepower
horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

# Build the Keras Sequential model
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

print(horsepower_model.summary())

horsepower_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

horsepower_model.predict(horsepower)

test_results = {}
test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)

x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

def plot_horsepower(x, y):
  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()

plot_horsepower(x, y)
plt.show()