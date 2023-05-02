# https://www.tensorflow.org/tutorials/keras/text_classification
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from keras import layers
from keras import losses

# url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# dataset = tf.keras.utils.get_file("aclImdb_v1", url,
#                                     untar=True, cache_dir='.',
#                                     cache_subdir='')
# dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
# Use a relative path

dataset_dir = 'aclImdb'

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

# remove unused folders to make it easier to load the data
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

# # access a file
# sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
# with open(sample_file) as f:
#   print(f.read())

# The IMDB dataset has already been divided into train and test, but it lacks a validation set. 
#   Let's create a validation set using an 80:20 split of the training data by using the 
#   validation_split argument below.
batch_size = 32
seed = 42
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed)


print("\n","Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1], "\n")

# create a validation and test dataset. You will use the remaining 5,000 reviews from 
#   the training set for validation
# Note: When using the validation_split and subset arguments, make sure to either specify a random seed, 
#   or to pass shuffle=False, so that the validation and training splits have no overlap.
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test', 
    batch_size=batch_size)

# Prepare the dataset for training

# standardize, tokenize, and vectorize the data using the helpful tf.keras.layers.TextVectorization layer.
#   Standardization refers to preprocessing the text, typically to remove punctuation or HTML elements to simplify the dataset. 
#   Tokenization refers to splitting strings into tokens (for example, splitting a sentence into individual words, by splitting on whitespace). 
#   Vectorization refers to converting tokens into numbers so they can be fed into a neural network. 
#       All of these tasks can be accomplished with this layer.

# the reviews contain various HTML tags like <br />. 
#   These tags will not be removed by the default standardizer in the TextVectorization layer 
#   (which converts text to lowercase and strips punctuation by default, but doesn't strip HTML). 
#   You will write a custom standardization function to remove the HTML

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

# create a TextVectorization layer
#   this layer is to standardize, tokenize, and vectorize the data
#   set the output_mode to int to create unique integer indices for each token
#               
max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)


# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# create a function to see the result of using this layer to preprocess some data
def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label
# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

# each token has been replaced by an integer. 
#   You can lookup the token (string) that each integer corresponds to by calling 
#   .get_vocabulary() on the layer.

print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))


# apply the TextVectorization layer to the train, validation, and test dataset
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Configure the dataset for performance
#   two important methods you should use when loading data to make sure that I/O does not become blocking.
#   .cache() keeps data in memory after it's loaded off disk. This will ensure the dataset does 
#       not become a bottleneck while training your model. If your dataset is too large to fit into memory, 
#       you can also use this method to create a performant on-disk cache, which is more efficient to read than many small files.
#   .prefetch() overlaps data preprocessing and model execution while training.

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create the model

embedding_dim = 16

# This callback will stop the training when there is no improvement in
# the loss for three consecutive epochs
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.01,
    patience=10,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0
)

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

print(model.summary())


# Loss function and optimizer
# A model needs a loss function and an optimizer for training. 
# Since this is a binary classification problem and the model outputs 
# a probability (a single-unit layer with a sigmoid activation), 
# you'll use losses.BinaryCrossentropy loss function.


optimizer = tf.keras.optimizers.RMSprop(    
    learning_rate=0.001,
    rho=0.95,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,) 
# best is RMSprop
# worst is Adadelta

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer=optimizer,
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# Train the model
epochs = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[callback],
    epochs=epochs)

# Evaluate the model
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)
# 782/782 [==============================] - 2s 2ms/step - loss: 0.3096 - binary_accuracy: 0.8742
# Loss:  0.30963364243507385
# Accuracy:  0.8741599917411804
# This fairly naive approach achieves an accuracy of about 86%.


# Create a plot of accuracy and loss over time
history_dict = history.history

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

# This is an example of overfitting: the model performs better on the training data than 
# it does on data it has never seen before. After this point, the model over-optimizes and 
# learns representations specific to the training data that do not generalize to test data.


# In the code above, you applied the TextVectorization layer to the dataset before feeding text 
# to the model. If you want to make your model capable of processing raw strings (for example, 
# to simplify deploying it), you can include the TextVectorization layer inside your model. To do so, 
# you can create a new model using the weights you just trained.

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

examples = [
  "I want to marry this movie",
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible...",
  "This movie is literally the worst I have ever seen"
]

print(export_model.predict(examples))