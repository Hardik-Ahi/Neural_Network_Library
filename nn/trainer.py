import numpy as np
from numpy.random import default_rng
from nn.dataset_utils import get_minibatch
from nn.functions import round_off
from nn.optimizers import SGD
import json, os, time

class Trainer:

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = SGD() if optimizer is None else optimizer
        self.optimizer.set_model(self.model)
        self.logger = Logger()
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer.set_model(self.model)
    
    def error_output_layer(self, label):
        value = self.optimizer.error_output_layer(label)
        self.model.layers[-1].b_gradients += value / self.batch_size
        self.model.layers[-1].del_ = value

    def error_layer(self, this_index, weight_index):
        value = self.optimizer.error_layer(this_index, weight_index)
        self.model.layers[this_index].b_gradients += value / self.batch_size
        self.model.layers[this_index].del_ = value

    def backward_pass(self, label):  # single instance
        self.error_output_layer(label)
        for i in range(len(self.model.layers)-2, 0, -1):  # except the input layer
            self.error_layer(i, i)
        
        for i in range(len(self.model.weights)):
            self.current_gradient(i)
    
    def current_gradient(self, weight_index):
        self.model.weights[weight_index].gradients += self.optimizer.current_gradient(weight_index) / self.batch_size
    
    def update_biases(self, layer_index):
        return self.optimizer.update_biases(layer_index, self.learning_rate)  # final, applied value (includes scaling by learning rate)
    
    def update_weights(self, weight_index):
        return self.optimizer.update_weights(weight_index, self.learning_rate)
    
    def forward_pass(self, input_data):
        # input_data shape = (1, n_cols)
        self.model.layers[0].a_ = input_data.reshape((input_data.shape[-1], 1))
        for i in range(1, len(self.model.layers)):
            self.model.layers[i].z_ = np.matmul(self.model.weights[i-1].matrix, self.model.layers[i-1].a_) + self.model.layers[i].b_
            self.model.layers[i].activate()
        
    def save_history(self):
        return self.logger.write_log()
    
    def train_batch(self, real_features, real_targets):
        # one optimizer update from a single (mini)batch
        real_targets = real_targets.reshape((real_targets.size,))
        self.batch_size = real_features.shape[0]  # real batch size for correctly normalizing gradients for accumulation
        for i in range(real_features.shape[0]):
            self.forward_pass(real_features[i])
            self.backward_pass(real_targets[i])

        bias = [self.update_biases(j) for j in range(1, len(self.model.layers))]
        weights = [self.update_weights(j) for j in range(len(self.model.weights))]
        self.logger.log_update(weights, bias, self.model.weights, self.model.layers)
        self.optimizer.on_pass()

    def train_epoch(self, features, targets):
        start_index = 0
        while start_index < features.shape[0]:
            real_features, real_targets = get_minibatch(features, targets, self.minibatch_size, start_index)
            self.train_batch(real_features, real_targets)
            start_index += self.batch_size

    def train(self, features, targets, batch_size = None, learning_rate = 0.01, epochs = 1, log_epochs = None, verbose = True, on_epoch_end = None):
        # on_epoch_end(epochs_done, epochs_total): optional progress callback, called after each epoch
        self.minibatch_size = batch_size  # None means batch GD. for stochastic, specify '1'.
        self.learning_rate = learning_rate
        if self.minibatch_size is None:
            self.minibatch_size = features.shape[0]
        if log_epochs is None:
            log_epochs = epochs

        self.logger.set_limit(log_epochs)
        self.logger.log_init(self.model.weights, self.model.layers)
        for epoch in range(epochs):
            if verbose:
                print(f"epoch-{epoch}:")
            self.logger.log_epoch(epoch)
            self.train_epoch(features, targets)
            self.predict(features, targets, for_plot = True, quiet = not verbose)
            if on_epoch_end is not None:
                on_epoch_end(epoch + 1, epochs)
    
    def forward_batch(self, features):
        # whole-dataset forward pass without touching per-layer state; returns first output neuron, shape (n_samples,)
        layers = self.model.layers
        if any(getattr(layer, 'activation', None) is None for layer in layers[1:]):
            predictions = list()
            for i in range(features.shape[0]):
                self.forward_pass(features[i])
                predictions.append(self.model.layers[-1].a_[0][0])
            return np.asarray(predictions)

        a = features.T
        for i in range(1, len(layers)):
            a = layers[i].activation(np.matmul(self.model.weights[i-1].matrix, a) + layers[i].b_)
        return np.asarray(a[0])

    def loss_only(self, features, targets):
        targets = targets.reshape((targets.size,))
        return self.model.loss.calculate_loss(targets, self.forward_batch(features))

    def confusion_matrix(self, labels, predictions):
        labels = labels.reshape(-1) == 1
        predictions = predictions.reshape(-1) == 1
        return {
            'tp': int(np.sum(labels & predictions)),
            'tn': int(np.sum(~labels & ~predictions)),
            'fp': int(np.sum(~labels & predictions)),
            'fn': int(np.sum(labels & ~predictions)),
        }
    
    def accuracy(self, labels, predictions):
        confusion_matrix = self.confusion_matrix(labels, predictions)
        return (confusion_matrix['tp'] + confusion_matrix['tn']) / sum(confusion_matrix.values())
    
    def predict(self, features, targets, output_incorrect = False, for_plot = False, quiet = False):
        targets = targets.reshape((targets.size,))
        predictions = self.forward_batch(features)
        loss = self.model.loss.calculate_loss(targets, predictions)
        predictions = round_off(predictions)
        matrix = self.confusion_matrix(targets, predictions)
        score = (matrix['tp'] + matrix['tn']) / sum(matrix.values())

        if not quiet:
            print("Loss:", loss)
            print("Accuracy:", score)
            print("Confusion matrix:", matrix)

        if for_plot:
            self.logger.log_score(loss, score, matrix)
            self.logger.log_predictions(predictions)
        if output_incorrect:
            exists = False
            for i in range(targets.size):
                if targets[i] != predictions[i]:
                    if exists == False:
                        exists = True
                        print("incorrect predictions:")
                    print(f'input = {features[i]}, label = {targets[i]}, prediction = {predictions[i]}')
        return predictions, loss, score

class RegressionTrainer(Trainer):
    def predict(self, features, targets, for_plot = False, quiet = False):
        targets = targets.reshape((targets.size,))
        predictions = self.forward_batch(features)
        loss = self.model.loss.calculate_loss(targets, predictions)

        # R^2 score
        residuals = np.sum((targets - predictions)**2)
        means = np.sum((targets - np.mean(targets))**2)
        score = 1 - (residuals / means)

        if not quiet:
            print("Loss:", loss)
            print("R2 score:", score)

        if for_plot:
            self.logger.log_score(loss, score)
            self.logger.log_predictions(predictions)
        return predictions, loss, score

class Logger:
    """Records training history in a compact, position-indexed format (see file_format.txt)."""
    VERSION = 2
    N_ELEMENTS = 4  # only the first few elements of each tracked array are ever plotted
    MAX_UPDATES = 700  # plots only show the first 700 updates; logging more just bloats the history
    PREDICTION_LIMIT = 200  # to limit size of the log

    def __init__(self):
        self.limit = None
        self.stop = False
        self.reset()

    def reset(self):
        self.epoch = -1
        self.object = {
            'version': self.VERSION,
            'tracked_weights': [],
            'tracked_biases': [],
            'init': {'weights': {}, 'bias': {}},
            'updates': {'weights_gradient': {}, 'bias_gradient': {}, 'weights': {}, 'bias': {}},
            'epochs': {
                'score': [], 'loss': [],
                'confusion_matrix': {'tp': [], 'tn': [], 'fp': [], 'fn': []},
                'predictions': [],
            },
        }

    def set_limit(self, limit):
        self.limit = limit-1
        self.stop = False

    def _head(self, array):
        return np.ravel(array)[:self.N_ELEMENTS].tolist()

    def log_init(self, model_weights, model_layers):
        self.reset()
        obj = self.object
        obj['tracked_weights'] = [0, len(model_weights)-1]
        obj['tracked_biases'] = [1, len(model_weights)]
        for i in dict.fromkeys(obj['tracked_weights']):
            obj['init']['weights'][str(i)] = self._head(model_weights[i].matrix)
            obj['updates']['weights_gradient'][str(i)] = []
            obj['updates']['weights'][str(i)] = []
        for i in dict.fromkeys(obj['tracked_biases']):
            obj['init']['bias'][str(i)] = self._head(model_layers[i].b_)
            obj['updates']['bias_gradient'][str(i)] = []
            obj['updates']['bias'][str(i)] = []

    def log_epoch(self, epoch):
        if self.limit is not None and epoch > self.limit:
            self.stop = True
            return
        self.epoch = epoch

    def log_update(self, weights_gradients, bias_gradients, model_weights, model_layers):
        # *_gradients: lists indexed by weight position / (layer position - 1)
        if self.stop:
            return
        updates = self.object['updates']
        first = next(iter(updates['weights']), None)
        if first is not None and len(updates['weights'][first]) >= self.MAX_UPDATES:
            return
        for i in dict.fromkeys(self.object['tracked_weights']):
            updates['weights_gradient'][str(i)].append(self._head(weights_gradients[i]))
            updates['weights'][str(i)].append(self._head(model_weights[i].matrix))
        for i in dict.fromkeys(self.object['tracked_biases']):
            updates['bias_gradient'][str(i)].append(self._head(bias_gradients[i-1]))
            updates['bias'][str(i)].append(self._head(model_layers[i].b_))

    def log_predictions(self, predictions):
        if self.stop:
            return
        self.object['epochs']['predictions'].append(np.asarray(predictions)[:self.PREDICTION_LIMIT].tolist())

    def log_score(self, loss, score, confusion_matrix = None):
        if self.stop:
            return
        epochs = self.object['epochs']
        epochs['score'].append(float(score))
        epochs['loss'].append(float(loss))
        if confusion_matrix is not None:
            for key, value in confusion_matrix.items():
                epochs['confusion_matrix'][key].append(int(value))

    def write_log(self):
        self.object['n_epochs'] = len(self.object['epochs']['loss'])
        string = json.dumps(self.object)
        self.reset()
        return string

    @staticmethod
    def load_data(string):
        data = json.loads(string)
        if data.get('version') != Logger.VERSION:
            raise ValueError("Unsupported training history format; re-run training to regenerate it.")
        return data