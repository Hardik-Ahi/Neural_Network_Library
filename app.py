import streamlit as st
import numpy as np
import pandas as pd
import io
import hashlib
import matplotlib.pyplot as plt

from nn.trainer import Trainer, RegressionTrainer, Logger
from nn.optimizers import SGD, Momentum, RMSProp
from nn.dataset_utils import and_gate_dataset, standardize_data, equalize_classes, split_classes, split_data, pca
from nn.model_classes import Model, Layer
from nn.functions import BinaryLoss, MSE, leaky_relu, der_leaky_relu, sigmoid, der_sigmoid, relu, der_relu, mirror, der_mirror
from nn.plotter import Plotter

# STREAMLIT CONFIG
st.set_page_config(page_title="Neural Network Demonstration", layout="wide")

st.title("Neural Network from Scratch", text_alignment="center", icon=":material/network_node:")

# SESSION STATE
if 'uploaded_dataset' not in st.session_state:
  st.session_state.uploaded_dataset = None
if 'target_type' not in st.session_state:
  st.session_state.target_type = None
if 'target_name' not in st.session_state:
  st.session_state.target_name = None
if 'model' not in st.session_state:
  st.session_state.model = None
if 'model_compiled' not in st.session_state:
  st.session_state.model_compiled = False
if 'train_set' not in st.session_state:
  st.session_state.train_set = None
if 'test_set' not in st.session_state:
  st.session_state.test_set = None
if 'X_y_train' not in st.session_state:
  st.session_state.X_y_train = []
if 'X_y_test' not in st.session_state:
  st.session_state.X_y_test = []
if 'activations' not in st.session_state:
  st.session_state.activations = []
if 'data' not in st.session_state:
  st.session_state.data = None
if 'history' not in st.session_state:
  st.session_state.history = None

def reset_session():
  st.session_state.clear()

st.button("Reset Current Session", type="primary", on_click=reset_session)

activation_functions = {
    "None": [None, None],
    "ReLU": [relu, der_relu],
    "Leaky ReLU": [leaky_relu(), der_leaky_relu()],
    "Sigmoid": [sigmoid, der_sigmoid],
    "Linear": [mirror, der_mirror]
}

optimizers = {
  "SGD": SGD,
  "Momentum": Momentum,
  "RMSProp": RMSProp
}

# DATASET

def clean_data():
  if st.session_state.form_drop_columns:
    st.session_state.uploaded_dataset.drop(columns=st.session_state.form_drop_columns, inplace=True)
  st.toast("Changes Applied.", icon=":material/check_circle:")

def preprocess_dataset():
  raw_data = st.session_state.uploaded_dataset
  raw_data = raw_data.dropna(ignore_index=True)  # drop NA rows
  
  # copy into permanent session state, not just for the form. Needed?
  target_col = st.session_state.form_target_column
  target_type = st.session_state.form_target_type
  st.session_state.target_type = target_type
  st.session_state.target_name = target_col

  if st.session_state.form_one_hot_columns:
    raw_data = pd.get_dummies(raw_data, columns=st.session_state.form_one_hot_columns, drop_first=True, dtype=int)

  if target_type == "classification":
    value_counts_old = raw_data[target_col].value_counts()
    equalize_classes(raw_data, target_col)  # downsampling
    value_counts_new = raw_data[target_col].value_counts()

    train_set, test_set = split_classes(raw_data, target_col)
    st.session_state.model = Model(BinaryLoss(), 1)
    st.session_state.trainer = Trainer(st.session_state.model)

    st.toast(f"Class counts before: {value_counts_old.to_dict()}\nAfter: {value_counts_new.to_dict()}", icon=":material/check_circle:", duration="infinite")
  else:
    train_set, test_set = split_data(raw_data)
    st.session_state.model = Model(MSE(), 1)
    st.session_state.trainer = RegressionTrainer(st.session_state.model)
  
  if st.session_state.form_standardize_columns:
    X_means, X_stds = standardize_data(train_set, st.session_state.form_standardize_columns)
    standardize_data(test_set, st.session_state.form_standardize_columns, from_means=X_means, from_stds=X_stds)

  if target_type == "regression" and target_col not in (st.session_state.form_standardize_columns or []):
    # a large-valued target makes the loss enormous and stalls learning, so always scale it (R2 is unaffected)
    y_means, y_stds = standardize_data(train_set, [target_col])
    standardize_data(test_set, [target_col], from_means=y_means, from_stds=y_stds)
    st.session_state.target_stats = (float(y_means[target_col]), float(y_stds[target_col]))
  st.session_state.uploaded_dataset = raw_data
  st.session_state.train_set = train_set
  st.session_state.test_set = test_set
  st.session_state.model_compiled = False

  X_train, y_train = train_set.drop(columns=[target_col]), train_set[target_col]
  st.session_state.X_y_train = [X_train.to_numpy(), y_train.to_numpy().reshape(-1, 1)]
  X_test, y_test = test_set.drop(columns=[target_col]), test_set[target_col]
  st.session_state.X_y_test = [X_test.to_numpy(), y_test.to_numpy().reshape(-1, 1)]

  st.toast("Preprocessing Applied.", icon=":material/check_circle:")

def adjust_dataset():
  if st.session_state.form_number_rows < len(st.session_state.uploaded_dataset):  # updated length after initial downsampling
    if st.session_state.target_type == "classification":
      equalize_classes(st.session_state.uploaded_dataset, st.session_state.target_name, st.session_state.form_number_rows)
    else:
      st.session_state.uploaded_dataset = st.session_state.uploaded_dataset.sample(n=st.session_state.form_number_rows, random_state=42).reset_index(drop=True)

st.header("Dataset")
dataset_option = st.selectbox("Select Dataset", ["AND Gate (Built-in)", "Upload a Dataset"])

if dataset_option == "Upload a Dataset":
  st.session_state.dataset_source = "upload"
  uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
  if uploaded_file is not None:
    if st.session_state.uploaded_dataset is None:
      dataset = pd.read_csv(uploaded_file)
      st.session_state.uploaded_dataset = dataset
    else:
      dataset = st.session_state.uploaded_dataset
    st.dataframe(dataset)
    st.subheader("Preprocess Dataset")

    with st.form("data_cleaning_form"):
      st.multiselect("Select columns to drop", dataset.columns, key="form_drop_columns")
      st.form_submit_button("Apply", on_click=clean_data)

    with st.form("preprocess_form"):  # separate re-runs based UI from logic
      st.markdown("For binary classification, the classes will be :color[equalized]{background='blue'} automatically.")
      st.selectbox("Select the target column", dataset.columns, key="form_target_column")
      st.selectbox("Select the target type", ["regression", "classification"], key="form_target_type")
      st.multiselect("Select columns to one-hot encode", dataset.columns, key="form_one_hot_columns")
      st.multiselect("Select columns to standardize", dataset.columns, key="form_standardize_columns")
      st.form_submit_button("Apply Preprocessing", on_click=preprocess_dataset)
    
    with st.form("additional_steps_form"):
      st.markdown(f"Current total size of dataset: :color[{len(dataset):,} rows]{{background='green'}}.")
      st.number_input("Adjust size of dataset (rows)", min_value=10, max_value=len(dataset), value=len(dataset), step=100, key="form_number_rows")
      st.form_submit_button("Adjust Dataset", on_click=adjust_dataset)

elif dataset_option == "AND Gate (Built-in)":
  # initialise once per selection; re-running on every interaction would discard layers added via the UI
  if st.session_state.get("dataset_source") != "and_gate":
    st.session_state.dataset_source = "and_gate"
    X_train, y_train = and_gate_dataset(100, 1)
    st.session_state.train_set = pd.DataFrame(np.hstack((X_train, y_train)), columns=["Input 1", "Input 2", "Output"])

    X_test, y_test = and_gate_dataset(50, 2)
    st.session_state.test_set = pd.DataFrame(np.hstack((X_test, y_test)), columns=["Input 1", "Input 2", "Output"])

    st.session_state.target_type = "classification"
    st.session_state.target_name = "Output"

    st.session_state.X_y_train = [X_train, y_train.reshape(-1, 1)]
    st.session_state.X_y_test = [X_test, y_test.reshape(-1, 1)]
    st.session_state.model = Model(BinaryLoss(), 1)
    st.session_state.trainer = Trainer(st.session_state.model)
    st.session_state.model_compiled = False
    st.session_state.activations = []
    st.session_state.history = None

# DISPLAY TRAIN, TEST SETS
if st.session_state.train_set is None:
  st.stop()

for title, dataset_part in [("Training Set", st.session_state.train_set), ("Testing Set", st.session_state.test_set)]:
  st.subheader(title)
  col_f, col_t = st.columns(2)
  with col_f:
    st.caption("Features")
    st.dataframe(dataset_part.drop(columns=[st.session_state.target_name]))
  with col_t:
    st.caption("Target")
    st.dataframe(dataset_part[[st.session_state.target_name]])

# MODEL
st.header("Model")
st.subheader("Add Layers")

def add_layer_callback():
    n_neurons = st.session_state.form_neurons
    activation = st.session_state.form_activation
    st.session_state.model.add_layer(Layer(n_neurons, activation_functions[activation][0], activation_functions[activation][1]))
    st.session_state.activations.append(activation)

with st.form("add_layers_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.number_input("Number of Neurons", min_value=1, max_value=20, value=1, step=1, key="form_neurons")
    with col2:
        st.selectbox("Activation Function", list(activation_functions.keys()), key="form_activation")
    with col3:
        st.form_submit_button("Add Layer", on_click=add_layer_callback)

# Display model summary
st.subheader("Model Summary")

def delete_layer_callback(index):
    st.session_state.model.layers.pop(index)
    st.session_state.activations.pop(index)
    # the weights no longer match the architecture, so the model must be recompiled and any history is stale
    st.session_state.model.weights = []
    st.session_state.model_compiled = False
    st.session_state.history = None
    st.session_state.data = None

if not st.session_state.model.layers:
  st.caption("No layers added yet.")
else:
  header = st.columns([1, 3, 3, 1])
  for col, title in zip(header, ["Layer", "Number of Neurons", "Activation Function", ""]):
    col.markdown(f"**{title}**")
  for i, l in enumerate(st.session_state.model.layers):
    row = st.columns([1, 3, 3, 1], vertical_alignment="center")
    row[0].write(i+1)
    row[1].write(l.n_neurons)
    row[2].write(st.session_state.activations[i])
    row[3].button("", icon=":material/delete:", key=f"delete_layer_{i}", help=f"Delete layer {i+1}",
                  on_click=delete_layer_callback, args=(i,))

if st.button("Compile Model"):
  if len(st.session_state.model.layers) < 2:
    st.error("Please add at least two layers (an input layer and an output layer) before compiling.")
  else:
    st.session_state.model.compile()
    st.session_state.model_compiled = True
    st.success("Model compiled!")

if st.checkbox("Show Weights and Biases") and st.session_state.model_compiled:
  st.subheader("Weights and Biases")
  for i, layer in enumerate(st.session_state.model.layers):
        st.write(f"Layer {i+1} Biases:", layer.b_.T)
  for i, weight in enumerate(st.session_state.model.weights):
      st.write(f"Layer {i+1} to {i+2} Weights:", weight.matrix)

# PREP PLOTTING

def figure_to_png(figure):
  try:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    return buffer.getvalue()
  finally:
    plt.close(figure)

def trainer_cache_key(trainer):
  model = trainer.model
  activations = tuple(
    (
      layer.n_neurons,
      getattr(layer.activation, "__module__", None),
      getattr(layer.activation, "__qualname__", None)
    )
    for layer in model.layers
  )
  loss = model.loss
  loss_type = loss if isinstance(loss, type) else type(loss)
  loss_key = (loss_type.__module__, loss_type.__qualname__)
  weights = tuple(np.asarray(model.weights_elements()).ravel().tolist())
  trainer_type = type(trainer)
  return (trainer_type.__module__, trainer_type.__qualname__), activations, loss_key, weights

@st.cache_data
def load_history(string):  # returns JSON (Python) object from string - serializable
  return Logger.load_data(string)

# `_data` (underscore) is excluded from Streamlit's hashing, which walks large nested lists element by
# element in Python and is very slow; `history_key` (a short digest of the history) identifies it instead.
@st.cache_data
def plot_gradients(_data, history_key):
  gradients = Plotter().plot_gradients(_data, n_points=700)
  return figure_to_png(gradients)

@st.cache_data
def plot_weights(_data, history_key):
  weights = Plotter().plot_weights(_data, n_points=700)
  return figure_to_png(weights)

@st.cache_data
def plot_score(_data, history_key, is_classification):
  score = Plotter().plot_score(_data, n_points=700, confusion_matrix=is_classification)
  return figure_to_png(score)

@st.cache_data
def plot_and_gate_predictions(_data, history_key, X_train):
  predictions = Plotter().plot_predictions(_data, X_train)
  return figure_to_png(predictions)

@st.cache_data
def plot_regression(_data, history_key, X_train, y_train):
  axis = pca(X_train, n_components=1)
  regression = Plotter().plot_regression(_data, axis, y_train)
  return figure_to_png(regression)

@st.cache_data
def plot_classification(_data, history_key, X_train, y_train):
  axis = pca(X_train, n_components=2)
  classification = Plotter().plot_classification(_data, axis[:, 0], axis[:, 1], y_train)
  return figure_to_png(classification)

@st.cache_data
def plot_loss_landscape(_data, history_key, _trainer, X_train, y_train, model_cache_key):
  contours = Plotter().plot_contours(_data, _trainer, X_train, y_train)
  return figure_to_png(contours)

# TRAIN

st.header("Training")
# input fields for training
with st.form("training_form"):
  batch_size = st.number_input("Batch Size (1 - 32)", min_value=1, max_value=32, value=1, step=1)
  learning_rate = st.number_input("Learning Rate (0.001 - 1.0)", min_value=0.001, max_value=1.0, value=0.02, step=0.001, format="%.3f")
  epochs = st.number_input("Epochs (1 - 500)", min_value=1, max_value=500, value=25, step=1)
  optimizer = st.selectbox("Optimizer", optimizers.keys())
  submitted = st.form_submit_button("Train Model")

if submitted:
  if not st.session_state.model_compiled:
    st.error("Please compile the model first.")
  else:
    progress_bar = st.progress(0.0, text=f"Epoch 0/{epochs}")

    def update_progress(done, total):
      progress_bar.progress(done / total, text=f"Epoch {done}/{total}")

    st.session_state.trainer.set_optimizer(optimizers[optimizer]())
    X_train, y_train = st.session_state.X_y_train
    st.session_state.trainer.train(X_train, y_train, batch_size, learning_rate, epochs = epochs, verbose = False, on_epoch_end = update_progress)
    st.session_state.history = st.session_state.trainer.save_history()
    progress_bar.empty()
    st.success("Training completed!")

# PLOT
st.header("Training History")

if st.button("Plot History"):
  if st.session_state.history is None:
    st.error("No training history found. Please train the model first.")
    st.stop()

  X_train, y_train = st.session_state.X_y_train

  with st.spinner("Reading training history..."):
    data = load_history(st.session_state.history)
    st.session_state.data = data
    history_key = hashlib.sha1(st.session_state.history.encode()).hexdigest()

  with st.spinner("Plotting gradients..."):
    gradient_path = plot_gradients(data, history_key)
    st.subheader("Gradients")
    st.image(gradient_path, width='stretch')

  with st.spinner("Plotting weights..."):
    weight_path = plot_weights(data, history_key)
    st.subheader("Weights")
    st.image(weight_path, width='stretch')

  with st.spinner("Plotting accuracy..."):
    score_path = plot_score(data, history_key, st.session_state.target_type == "classification")
    st.subheader("Accuracy")
    st.image(score_path, width='stretch')

  with st.spinner("Plotting outputs..."):
    if st.session_state.target_type == 'regression':
      predictions = plot_regression(data, history_key, X_train, y_train)
    elif st.session_state.target_type == 'classification':
      #predictions = plot_and_gate_predictions(data, history_key, X_train)
      predictions = plot_classification(data, history_key, X_train, y_train)
    st.subheader("Predictions")
    st.image(predictions, width='stretch')

  with st.spinner("Plotting loss landscape (this takes a while)..."):
    model_key = trainer_cache_key(st.session_state.trainer)
    loss_landscape_path = plot_loss_landscape(data, history_key, st.session_state.trainer, X_train, y_train, model_key)
    st.subheader("Loss Landscape")
    st.image(loss_landscape_path, width='stretch')

  st.success("Plots generated!")