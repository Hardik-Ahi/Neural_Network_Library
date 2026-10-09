import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def skip_samples(array, limit):
        # can discard a few points from the end of the array
        has = array.shape[0]
        initial_skip = has // limit
        if initial_skip == 0:
            return array
        array = array[::initial_skip]

        first = array.shape[0] - 2 * (array.shape[0] - limit)
        if first == array.shape[0]:
            return array
        array = np.vstack((array[:first], array[first::2]))
        return array

    @staticmethod
    def _matrix_columns(series):
        # series: (T, K) first K elements of a weight matrix; laid out as 2 x (K/2) (or a single element)
        if series.shape[1] > 1:
            series = series.reshape((series.shape[0], 2, -1))
            return [(f'{r*2 + c}', series[:, r, c]) for r in range(series.shape[1]) for c in range(series.shape[2])]
        return [('0', series[:, 0])]

    @staticmethod
    def _bias_columns(series):
        return [(f'{r}', series[:, r]) for r in range(series.shape[1])]

    def _plot_tracked(self, data, title, weights_series, bias_series, n_points):
        # weights_series / bias_series: dict[str index] -> (T, K) array
        tracked_weights = data['tracked_weights']
        tracked_biases = data['tracked_biases']
        n_cols = len(tracked_weights)

        fig, axs = plt.subplots(2, n_cols, figsize = (7 * n_cols, 8), gridspec_kw = {'wspace': 0.2, 'hspace': 0.3}, squeeze = False)
        fig.suptitle(title, size = 'xx-large')

        for ax, index in enumerate(tracked_weights):
            axis = axs[0, ax]
            for label, column in self._matrix_columns(weights_series[str(index)][:n_points]):
                axis.plot(column, label = label, linewidth = 0.7)
            axis.legend(title = 'elements')
            axis.set_title(f"weights-{index}")
            axis.set_xlabel("Updates")

        for ax, index in enumerate(tracked_biases):
            axis = axs[1, ax]
            for label, column in self._bias_columns(bias_series[str(index)][:n_points]):
                axis.plot(column, label = label, linewidth = 0.7)
            axis.legend(title = 'elements')
            axis.set_title(f"bias-{index}")
            axis.set_xlabel("Updates")

        return fig

    def plot_gradients(self, data, n_points = None):
        updates = data['updates']
        weights = {k: np.asarray(v) for k, v in updates['weights_gradient'].items()}
        bias = {k: np.asarray(v) for k, v in updates['bias_gradient'].items()}
        return self._plot_tracked(data, 'Gradients', weights, bias, n_points)

    def plot_weights(self, data, n_points = None):
        init = data['init']
        updates = data['updates']
        # initial values are the first point of each series
        weights = {k: np.vstack((np.asarray(init['weights'][k])[None, :], np.asarray(v))) for k, v in updates['weights'].items()}
        bias = {k: np.vstack((np.asarray(init['bias'][k])[None, :], np.asarray(v))) for k, v in updates['bias'].items()}
        return self._plot_tracked(data, "Weights & Bias Values", weights, bias, n_points)

    def plot_score(self, data, n_points = None, confusion_matrix = True):
        epochs = data['epochs']
        loss_list = np.asarray(epochs['loss'])[:n_points]
        score_list = np.asarray(epochs['score'])[:n_points]

        if confusion_matrix:
            fig, axs = plt.subplots(1, 2, figsize = (15, 5), gridspec_kw = {'wspace': 0.2, 'hspace': 0.3})
        else:
            fig, axs = plt.subplots(figsize = (8, 6))
            axs = np.array([axs])
        fig.suptitle("Metrics", size = 'xx-large')

        axs[0].plot(loss_list, label = "Loss")
        axs[0].plot(score_list, label = "Score")
        axs[0].set_xlabel("Epochs")
        axs[0].set_title("Loss & Score")
        axs[0].legend()

        if confusion_matrix:
            for key, values in epochs['confusion_matrix'].items():
                axs[1].plot(np.asarray(values)[:n_points], label = key)
            axs[1].set_xlabel("Epochs")
            axs[1].set_title("Confusion Matrix")
            axs[1].legend()

        return fig

    @staticmethod
    def plot_points(x, y):
        fig, ax = plt.subplots(figsize = (7, 5))
        fig.suptitle("Points", size = 'xx-large')

        ax.scatter(x, y)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        return fig

    @staticmethod
    def _sample_epochs(data):
        return np.linspace(0, data['n_epochs']-1, 4, dtype = int)

    def plot_predictions(self, data, features, predictions = None):
        if predictions is not None:
            fig, axis = plt.subplots(figsize = (6, 4))
            fig.suptitle("Predictions", size = "xx-large")

            zeros = (predictions == 0)  # use this boolean array to index features
            ones = (predictions == 1)
            axis.scatter(features[zeros, 0], features[zeros, 1], c = "red", label = "0")
            axis.scatter(features[ones, 0], features[ones, 1], c = "blue", label = "1")
            axis.legend()

            return fig

        fig, axs = plt.subplots(2, 2, figsize = (12, 8), gridspec_kw = {'wspace': 0.2, 'hspace': 0.3})
        fig.suptitle("Predictions", size = "xx-large")

        for i, epoch in enumerate(self._sample_epochs(data)):
            axis = axs[i//2][i%2]
            predictions = np.asarray(data['epochs']['predictions'][epoch], dtype = int)
            zeros = (predictions == 0)
            ones = (predictions == 1)
            subset = features[:predictions.shape[0]]
            axis.scatter(subset[zeros, 0], subset[zeros, 1], c = "red", label = "0")
            axis.scatter(subset[ones, 0], subset[ones, 1], c = "blue", label = "1")
            axis.set_title(f'Epoch-{epoch}')
            axis.legend()

        return fig

    def plot_classification(self, data, x_axis, y_axis, targets):
        fig, axs = plt.subplots(2, 2, figsize = (12, 8), gridspec_kw = {'wspace': 0.2, 'hspace': 0.3})
        fig.suptitle("Classification: First 200 samples", size = "x-large")

        for i, epoch in enumerate(self._sample_epochs(data)):
            axis = axs[i//2][i%2]

            predictions = np.asarray(data['epochs']['predictions'][epoch], dtype = int)
            shorter = min(predictions.shape[0], x_axis.shape[0])
            predictions = predictions[:shorter]
            x = x_axis[:shorter]
            y = y_axis[:shorter]
            labels = targets[:shorter].reshape(predictions.shape)

            zeros = (predictions == 0)  # use this boolean array to index features
            ones = (predictions == 1)
            incorrect = ((labels == 1) & zeros) | ((labels == 0) & ones)

            axis.scatter(x[zeros], y[zeros], c = "red", label = "0")
            axis.scatter(x[ones], y[ones], c = "blue", label = "1")
            axis.scatter(x[incorrect], y[incorrect], c = "black", label = "incorrect", marker = "x")
            axis.set_xlabel("PCA Component-1")
            axis.set_ylabel("PCA Component-2")
            axis.set_title(f'Epoch-{epoch}')
            axis.legend()

        return fig

    def plot_regression(self, data, features, targets, predictions = None, x_label = "PCA Component-1"):
        features = features.reshape((features.shape[0], 1))
        targets = targets.reshape((targets.shape[0], 1))
        prediction_size = 15

        if predictions is not None:
            fig, ax = plt.subplots(figsize = (6, 4))
            fig.suptitle("Predictions", size = "xx-large")

            ax.scatter(features[:, 0], targets[:, 0], label = "targets", c = "cornflowerblue")
            ax.scatter(features[:, 0], predictions, label = "predictions", marker = "x", c = "coral", s = prediction_size)
            ax.set_xlabel(x_label)
            ax.legend()

            return fig

        fig, axs = plt.subplots(2, 2, figsize = (12, 8), gridspec_kw = {'wspace': 0.2, 'hspace': 0.3})
        fig.suptitle("Predictions", size = "xx-large")

        for i, epoch in enumerate(self._sample_epochs(data)):
            axis = axs[i//2][i%2]
            predictions = np.asarray(data['epochs']['predictions'][epoch])
            n = predictions.shape[0]
            axis.scatter(features[:n, 0], targets[:n, 0], label = "targets", c = "cornflowerblue")
            axis.scatter(features[:n, 0], predictions, label = "predictions", marker = "x", c = "coral", s = prediction_size)
            axis.set_title(f'Epoch-{epoch}')
            axis.set_xlabel(x_label)
            axis.legend()

        return fig

    def plot_contours(self, data, trainer, features, targets, magnitude = 1, seed = 91):
        model = trainer.model
        original_vector = model.weights_elements()
        norm = np.linalg.norm(original_vector)

        seeds = default_rng(seed).choice(1000, (4, 2), replace = False)
        fig, axs = plt.subplots(2, 2, figsize = (12, 8), gridspec_kw = {'wspace': 0.2, 'hspace': 0.3})
        fig.suptitle("Loss Landscape", size = "xx-large")

        x = np.linspace(-magnitude, magnitude, 20)
        y = np.linspace(-magnitude, magnitude, 20)
        z = np.zeros((y.size, x.size))  # as per matplotlib's contour plot signature

        features = features[:200, :]
        targets = targets[:200, :]

        try:
            for ax in range(4):
                vector1 = default_rng(seeds[ax][0]).uniform(-1, 1, size = original_vector.shape)
                vector1 = (vector1 / np.linalg.norm(vector1)) * norm
                vector2 = default_rng(seeds[ax][1]).uniform(-1, 1, size = original_vector.shape)
                vector2 = (vector2 / np.linalg.norm(vector2)) * norm

                for i in range(y.size):  # y first
                    for j in range(x.size):
                        model.set_weights(original_vector + y[i] * vector1 + x[j] * vector2)
                        z[i, j] = trainer.loss_only(features, targets)

                axis = axs[ax//2][ax%2]
                contour_set = axis.contour(x, y, z, levels = 15, cmap = "magma")
                axis.clabel(contour_set)
                axis.set_xlabel("alpha")
                axis.set_ylabel("beta")
                axis.scatter(0, 0, c = "blue", label = "model")
                axis.legend()
        finally:
            model.set_weights(original_vector)
        return fig
