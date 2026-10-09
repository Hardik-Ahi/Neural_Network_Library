import numpy as np

# All activations are true array operations (no np.vectorize): faster, and the output dtype is always float
# (np.vectorize infers dtype from the first element, which silently truncated floats to ints).

def relu(x):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x, 0.0)

def leaky_relu(leak = 0.1, alpha = 1):
    def func(x):
        x = np.asarray(x, dtype=float)
        return np.where(x > 0, alpha * x, x * leak)  # leak is accessed through a 'closure'
    func.__name__ = "leaky_relu"
    return func

def der_leaky_relu(leak = 0.1, alpha = 1):
    def func(x):
        x = np.asarray(x, dtype=float)
        return np.where(x > 0, float(alpha), float(leak))
    return func

def sigmoid_func(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))

sigmoid = sigmoid_func

def round_off(x):
    return np.where(np.asarray(x) >= 0.5, 1, 0)

def mirror(x):
    return np.asarray(x, dtype=float)

def der_mirror(x):
    return np.ones_like(np.asarray(x, dtype=float))

def der_sigmoid(x):
    s = sigmoid_func(x)
    return s * (1 - s)

def der_relu(x):
    return np.where(np.asarray(x) > 0, 1.0, 0.0)

# same interface as obtained thru np.vectorize
def softmax(layer):
    total = np.sum(np.exp(layer))
    return np.exp(layer) / total

class MSE:
    def calculate_loss(self, labels, predictions):
        labels = labels.reshape(predictions.shape)
        loss = (1 / labels.size) * np.sum((labels - predictions)**2)
        return loss
    
    def der_loss(self, label, output):
        return 2 * (output - label)
    
    def error_output_layer(self, layer, label):
        first_term = self.der_loss(label, layer.a_[0][0])
        second_term = layer.der_activate()
        error = np.clip(first_term * second_term, -5, 5)
        return error

class BinaryLoss(MSE):
    def calculate_loss(self, labels, predictions, epsilon = 1e-7):
        total = predictions.size
        labels = labels.reshape(predictions.shape)

        predictions = np.clip(predictions, epsilon, 1 - epsilon)  # avoiding values of 0 ( => inf loss) and 1 ( => 0 loss)
        summation = np.sum(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))
        return (-1 / total) * summation

    def der_loss(self, label, output, epsilon = 1e-7):  # output belongs to range [0, 1]
        output = np.clip(output, epsilon, 1 - epsilon)
        return ((1 - label)/(1 - output)) - (label / output)

class CategoricalLoss(MSE):
    # labels must be one-hot encoded with drop_first = False; labels & predictions = complete vector of all classes for each instance.
    def calculate_loss(self, labels, predictions, epsilon = 1e-9):
        total = predictions.shape[0]
        labels = labels.reshape(predictions.shape)

        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        summation = np.sum(labels * np.log(predictions))
        
        return (-1 / total) * summation
    
    # always use 'softmax' with this
    def error_output_layer(self, layer, label):
        label = label.reshape((label.size, 1))
        return layer.a_ - label
