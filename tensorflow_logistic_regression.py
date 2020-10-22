import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# MNIST dataset parameters
number_of_classes = 10 # 0 to 9 digits
number_of_features = 784 # 28 * 28

# Training parameters
tf.enable_eager_execution()
learning_rate = 0.01
training_steps = 1000
batch_size = 256
display_step = 50
shuffle = 5000
optimiser = tf.keras.optimizers.SGD(learning_rate) # Stochastic gradient descent optimiser
number_of_images_to_predict = 5

print(f"Tensorflow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

# Prepare MNIST dataset
(train_input, train_target), (test_input, test_target) = tf.keras.datasets.mnist.load_data()

# Transform train_input data structure
train_input = np.array(train_input, np.float32) # Convert to float32
train_input = train_input.reshape([-1, number_of_features]) # Flatten images to 1 dimension vector of 784 features (28 * 28)
train_input = train_input / 255. # Normalise images value from [0, 255] to [0, 1]

# Transform test_input data structure
test_input = np.array(test_input, np.float32) # Convert to float32
test_input = test_input.reshape([-1, number_of_features]) # Flatten images to 1 dimension vector of 784 features (28 * 28)
test_input = test_input / 255. # Normalise images value from [0, 255] to [0, 1]

# Shuffle and batch data
train_data = tf.data.Dataset.from_tensor_slices((train_input, train_target))
train_data = train_data.repeat().shuffle(shuffle).batch(batch_size).prefetch(1)

# Weight of shape [784, 10] i.e. number of features (28 * 28) and number of classes
weight = tf.Variable(tf.ones([number_of_features, number_of_classes]), name = "weight")

# Bias of shape [10] i.e. number of classes
bias = tf.Variable(tf.zeros([number_of_classes]), name = "bias")

# Logistic regression
# target = (weight * input) + bias
def logistic_regression(input):
    return tf.nn.softmax(tf.matmul(input, weight) + bias) # Apply softmax to normalise the logits to a probability distribution

# Cross entropy loss function
def cross_entropy(predicted_target, actual_target):
    actual_target = tf.one_hot(actual_target, depth = number_of_classes) # Encode label using one hot encoding
    predicted_target = tf.clip_by_value(predicted_target, 1e-9, 1.) # Clip prediction values to avoid log(0) error
    return tf.reduce_mean(-tf.reduce_sum(actual_target * tf.math.log(predicted_target), 1)) # Compute cross entropy

# Accuracy metric
def accuracy_metric(predicted_target, actual_target):
    correct_prediction = tf.equal(tf.argmax(predicted_target, 1), tf.cast(actual_target, tf.int64)) # Predicted class is the index of the highest score in prediction vector i.e. argmax
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Optimisation process
def run_optimisation(input, target):
    with tf.GradientTape() as g: #Wrap computation inside a GradientTape for automatic differentiation
        predicted = logistic_regression(input)
        loss = cross_entropy(predicted, target)

    gradients = g.gradient(loss, [weight, bias]) # Compute gradients
    optimiser.apply_gradients(zip(gradients, [weight, bias]))

# Run tranining for the given number of steps
print(f"\n--- Start Training ---")
for each_step, (input_batch, target_batch) in enumerate(train_data.take(training_steps), 1):
    run_optimisation(input_batch, target_batch) # Run the optimisation to update weight and bias values

    if each_step % display_step == 0:
        predicted = logistic_regression(input_batch)
        loss = cross_entropy(predicted, target_batch)
        accuracy = accuracy_metric(predicted, target_batch)
        print(f"step: {each_step}, loss: {loss}, accuracy: {accuracy}")

print(f"--- End Training ---\n")

# Test model on validation set
print(f"\n--- Start Validation ---")
predicted = logistic_regression(test_input)
print(f"accuracy: {accuracy_metric(predicted, test_target)}")
print(f"--- End Validation ---\n")

# Predict images from validation set
number_of_images = number_of_images_to_predict
test_images = test_input[:number_of_images]
predictions = logistic_regression(test_images)

# Display image and model prediction
for each_index in range(number_of_images):
    print(f"Model prediction: {np.argmax(predictions.numpy()[each_index])}")
    plt.imshow(np.reshape(test_images[each_index], [28, 28]), cmap = "gray")
    plt.show()