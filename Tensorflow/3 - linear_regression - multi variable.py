import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# machine learning
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

# remove data from dataset
raw_dataset.pop('Acceleration' )

dataset = raw_dataset.copy()

# unknown values
dataset.isna().sum()

# drop unknown values
dataset = dataset.dropna()

# change 'Origin' to country
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

# The "Origin" column is now categorical, not numeric. So the next step is to one-hot encode the values in the column with pd.get_dummies.
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
# Convert boolean values to integers
dataset[['Europe', 'Japan', 'USA']] = dataset[['Europe', 'Japan', 'USA']].astype(int)

# Split the data into training and test sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# inspect data
# sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
# plt.show()

# Separate the target value—the "label"—from the features. 
# This label (MPG) is the value that you will train the model to predict.
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# normalize all data
all_data = np.array(train_features)

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(all_data)

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.08), # options: https://keras.io/api/optimizers/
    loss='mean_absolute_error') # options: https://keras.io/api/losses/regression_losses/

history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

def predict_value(value, feature):
    pred_values = np.array(train_features[feature])
    mean_values = train_features.mean(axis=0)

    x_hp = np.linspace(np.min(pred_values), np.max(pred_values), 100)
    input_array = np.array([mean_values] * len(x_hp))
    input_array[:, train_features.columns.get_loc(feature)] = x_hp

    y_hp = linear_model.predict(input_array)

    return np.interp(value, x_hp, y_hp.flatten())

# Example usage: predict MPG for a vehicle with a horsepower of 100
test_value = 100
predicted_mpg = predict_value(test_value, 'Horsepower')
print(f"Predicted MPG for a vehicle with {test_value} horsepower: {predicted_mpg:.2f}")

value = 'Horsepower'

pred_values = np.array(train_features[value])
mean_values = train_features.mean(axis=0)

x_hp = np.linspace(np.min(pred_values), np.max(pred_values), 100)
input_array = np.array([mean_values] * len(x_hp))
input_array[:, train_features.columns.get_loc(value)] = x_hp

y_hp = linear_model.predict(input_array)

def plot_horsepower(x, y):
    plt.scatter(pred_values, train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel(value)
    plt.ylabel('MPG')
    plt.legend()

plot_horsepower(x_hp, y_hp)
plt.show()

# show error metric ('mean_absolute_error') over time ('loss')
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([1, 5])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)

plot_loss(history)
plt.show()
