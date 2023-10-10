# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape
from keras.callbacks import EarlyStopping
from datetime import datetime
import mlflow
import mlflow.pyfunc

# Helper Functions

# Load and preprocess the data
def load_and_preprocess_data(filename, column_name, years, points_per_year):
    try:
        data = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file {filename} is empty.")
        return
    except pd.errors.ParserError:
        print(f"Error: There was a problem parsing {filename}.")
        return

    # Create the new index
    index_list = []
    for year, points in zip(years, points_per_year):
        for i in range(1, points+1):
            index_list.append(f"{year}-{i:02d}")

    # Add the new index to the dataframe and drop the 'x' column
    data['date'] = index_list
    data.drop(columns=['x'], inplace=True)

    # Convert the data column to float and set 'date' as index
    data[column_name] = data[column_name].str.replace(',', '.').astype(float)
    data.set_index('date', inplace=True)

    # Interpolate the data
    full_date_range = pd.date_range(start="2019-01-01", end="2023-08-01", freq="MS")
    full_index = [f"{date.year}-{date.month:02d}" for date in full_date_range]
    data = data.reindex(full_index).interpolate()

    # Convert index values to datetime format
    data.index = pd.to_datetime(full_date_range)

    return data

# Plot the data
def plot_loaded_data(data, title, y_label):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Define the TSMixer-inspired model
def build_tsmixer_model(input_shape, num_features):
    # Input layer
    input_layer = Input(shape=input_shape)

    # Time-mixing: A simple linear model for temporal patterns
    x = Flatten()(input_layer)
    x = Dense(units=input_shape[0] * num_features, activation='linear')(x)
    x = Reshape(target_shape=(input_shape[0], num_features))(x)

    # Feature-mixing: A basic MLP to capture cross-variate patterns
    x = Flatten()(x)
    x = Dense(units=50, activation='relu')(x)
    output_layer = Dense(units=num_features, activation='linear')(x)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Train the TSMixer-inspired model
from keras.callbacks import EarlyStopping

def train_tsmixer_model(model, X_train, Y_train, X_test, Y_test, epochs=20, batch_size=8, patience=5):
    """
    Train a TimeSeriesMixer model.

    Parameters:
    - model: The model to be trained.
    - X_train (numpy array): Training features.
    - Y_train (numpy array): Training labels.
    - X_test (numpy array): Testing features.
    - Y_test (numpy array): Testing labels.
    - epochs (int, optional): Number of epochs for training. Default is 20.
    - batch_size (int, optional): Batch size for training. Default is 8.
    - patience (int, optional): Number of epochs with no improvement after which training will be stopped. Default is 5.

    Returns:
    - history: Training history object containing training and validation loss.
    """

    # Initialize the EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=1,
        restore_best_weights=True
    )

    try:
        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, Y_test),
            verbose=1,
            callbacks=[early_stopping]
        )

    except ValueError as e:
        print(f"Error during model training: {e}")
        return None

    # Plot training & validation loss values
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss Progress')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    plt.savefig("plots/training_loss_plot.png")
    plt.show()

    return history

# Generate input-output pairs for supervised learning
def generate_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        Y.append(data[i + look_back, :])
    return np.array(X), np.array(Y)

# Splitting the data based on time
def time_based_split(data, train_size=0.8):
    train_length = int(len(data) * train_size)
    train, test = data[:train_length], data[train_length:]
    return train, test

# Scale the data and return the scalers for later inverse transformation
def scale_data(data):
    scalers = {}
    scaled_data = data.copy()

    for column in data.columns:
        try:
            scaler = MinMaxScaler()
            scaled_data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1)).flatten()
            scalers[column] = scaler
        except ValueError as e:
            print(f"Error scaling the column {column}: {e}")
            return scaled_data, scalers

    return scaled_data, scalers

# Inverse Scale the data
def inverse_transform_array(data, column_scalers):
    inverse_data = data.copy()

    for idx, column in enumerate(column_scalers):
        inverse_data[:, idx] = column_scalers[column].inverse_transform(data[:, idx].reshape(-1, 1)).flatten()

    return inverse_data

# Function to create train and test sets
def train_test_time_based_split(data, look_back, train_size=0.8):
    """
    Generate train and test datasets using time-based split.

    Parameters:
    - data (numpy array): Time series data.
    - look_back (int): Number of previous time steps to use as input variables to predict the next time period.
    - train_size (float): Proportion of the dataset to include in the train split.

    Returns:
    - X_train, X_test, Y_train, Y_test
    """
    # Assuming the function 'generate_dataset' is already defined in your code
    X, Y = generate_dataset(data, look_back)

    # Assuming the function 'time_based_split' is already defined in your code
    X_train, X_test = time_based_split(X, train_size=train_size)
    Y_train, Y_test = time_based_split(Y, train_size=train_size)

    return X_train, X_test, Y_train, Y_test, X, Y

# Function to do Backtesting
def backtest_model(model, X, Y, features, data_scalers, retrain=True, epochs=20, batch_size=8, patience=5, plot_results=False):
    try:
        predictions = []
        true_values = []

        # Initialize the EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor='loss',
            patience=patience,
            verbose=0,
            restore_best_weights=True
        )

        for t in range(len(X)):
            # Split data up to t
            X_train_bt = X[:t+1]
            Y_train_bt = Y[:t+1]

            # Retrain the model if needed
            if retrain:
                model.fit(
                    X_train_bt, Y_train_bt,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[early_stopping]
                )

            # Forecast the next value
            predicted_value = model.predict(X[t:t+1])
            true_value = Y[t]

            predictions.append(predicted_value[0])
            true_values.append(true_value)

        predictions = np.array(predictions)
        true_values = np.array(true_values)

        # If plot_results is set to True, plot the backtesting results
        if plot_results:
            # Use the inverse_transform_array function to get original data scale
            predictions_original = inverse_transform_array(predictions, data_scalers)
            true_values_original = inverse_transform_array(true_values, data_scalers)

            # Create subplots
            fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(15, 10))

            # Plotting for each feature
            for idx, feature in enumerate(features):
                axes[idx].plot(true_values_original[:, idx], label=f'Real {feature}', color='blue' if idx == 0 else 'green')
                axes[idx].plot(predictions_original[:, idx], label=f'Backtest Predicted {feature}', alpha=0.7, color='lightblue' if idx == 0 else 'lightgreen')
                axes[idx].set_title(f'Backtest Results - {feature}')
                axes[idx].legend()

            # Display the plots
            plt.tight_layout()
            plt.savefig("plots/backtesting_predictions.png")
            plt.show()

        return predictions, true_values

    except NameError:
        print("Error during backtesting.")

# Function to make predictions with the test set
def predict_test_set(model, X_test, Y_test, data_scalers, features):
    """
    Predict values on a test set using the provided model and visualize the predictions.

    Parameters:
    - model: The model to use for predictions.
    - X_test (numpy array): Testing features.
    - data_scalers (list): List of scalers used for inverse transformation.
    - features (list): List of feature names for plotting.

    Returns:
    - Y_test_original: Original test labels after inverse transformation.
    - predictions_original: Predicted values after inverse transformation.
    """

    try:
        predictions = model.predict(X_test)
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None, None

    # Inverse transform the data
    Y_test_original = inverse_transform_array(Y_test, data_scalers)
    predictions_original = inverse_transform_array(predictions, data_scalers)

    # Create subplots
    fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(15, 10))

    for i, feature in enumerate(features):
        axes[i].plot(Y_test_original[:, i], label=f'Real {feature}', color='blue' if i==0 else 'green')
        axes[i].plot(predictions_original[:, i], label=f'Predicted {feature}', alpha=0.7, color='lightblue' if i==0 else 'lightgreen')
        axes[i].set_title(f'Real vs Predicted - {feature}')
        axes[i].legend()

    # Display the plots
    plt.tight_layout()
    plt.show()

    return Y_test_original, predictions_original, predictions

# Generate a Forecast
def forecast_data(tsmixer_model, data, look_back, forecast_steps, data_scalers, features, columns=['Forecast_1', 'Forecast_2']):
    """
    Forecast data using the given model.

    Parameters:
    - tsmixer_model: Model to use for forecasting.
    - data: Data to forecast from.
    - look_back: Number of previous data points to consider for forecasting.
    - forecast_steps: Number of steps to forecast.
    - data_scalers: Scalers used for inverse transformation.
    - columns: Column names for the forecasted data.

    Returns:
    - forecasted_df: DataFrame containing the forecasted data.
    """

    forecasted_data = []
    input_data = np.array(data[-look_back:])

    # try:
    for step in range(forecast_steps):
        # predicted_step = tsmixer_model.predict(input_data[-look_back:].reshape(1, look_back, 2))
        predicted_step = tsmixer_model.predict(input_data[-look_back:].reshape(1, look_back, len(features)))
        input_data = np.vstack([input_data, predicted_step])
        forecasted_data.append(predicted_step[0])
    # except Exception as e:
    #     print(f"Error during forecasting: {e}")
    #     return None

    forecasted_data = np.array(forecasted_data)
    forecasted_data_original = inverse_transform_array(forecasted_data, data_scalers)
    forecasted_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq="MS")
    forecasted_df = pd.DataFrame(forecasted_data_original, columns=columns, index=forecasted_dates)

    return forecasted_df

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error (MAPE).

    Parameters:
    - y_true (numpy array): True values.
    - y_pred (numpy array): Predicted values.

    Returns:
    - mape (float): The MAPE score.
    """
    # Avoid division by zero and convert to percentage
    y_true = np.where(y_true == 0, 1, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

# Generate metrics for the test set
def evaluate_test_set(Y_test, predictions, data_scalers, features):
    """
    Evaluate the performance of the predictions against the true test set values.

    Parameters:
    - Y_test (numpy array): True test labels.
    - predictions (numpy array): Predicted values.
    - data_scalers (list): List of scalers used for inverse transformation.
    - features (list): List of feature names.

    Returns:
    - metrics (dict): Dictionary with MAE, RMSE, and MAPE for each feature.
    """

    # Inverse transform the Y_test and predictions arrays
    Y_test_original = inverse_transform_array(Y_test, data_scalers)
    predictions_original = inverse_transform_array(predictions, data_scalers)

    metrics = {}

    for i, feature in enumerate(features):
        mae = mean_absolute_error(Y_test_original[:, i], predictions_original[:, i])
        rmse = np.sqrt(mean_squared_error(Y_test_original[:, i], predictions_original[:, i]))
        mape = mean_absolute_percentage_error(Y_test_original[:, i], predictions_original[:, i])

        metrics[f"{feature}_MAE"] = mae
        metrics[f"{feature}_RMSE"] = rmse
        metrics[f"{feature}_MAPE"] = mape

        print(f"{feature} - MAE: {mae}, RMSE: {rmse}, MAPE: {mape}%")

    return metrics

# Make a plot of the test set predictions with the time series history
def plot_test_set_predictions_with_history(Y_train, Y_test, predictions, data_scalers, features, save_path="plots/test_set_predictions.png"):
    """
    Plots test set predictions alongside the training data (history).

    Parameters:
    - Y_train (numpy array): Training set labels.
    - Y_test (numpy array): Test set labels.
    - predictions (numpy array): Predicted values for the test set.
    - data_scalers (list): List of scalers used for inverse transformation.
    - features (list): List of feature names.
    - save_path (str): Path to save the plot.

    """
    # Initialize empty arrays for the full sequence
    full_true_sequence = np.vstack([Y_train, Y_test])
    full_predicted_sequence = np.vstack([np.full((Y_train.shape[0], len(features)), np.nan), predictions])

    # Inverse transform the entire sequences
    full_true_sequence_original = inverse_transform_array(full_true_sequence, data_scalers)
    full_predicted_sequence_original = inverse_transform_array(full_predicted_sequence, data_scalers)

    # Plot
    plt.figure(figsize=(10,6))

    for i, feature in enumerate(features):
        plt.subplot(len(features), 1, i+1)
        plt.plot(full_true_sequence_original[:, i], label=f"Real {feature}", color="blue" if i==0 else "green")
        plt.plot(full_predicted_sequence_original[:, i], label=f"Predicted {feature}", alpha=0.7, color="lightblue" if i==0 else "lightgreen")
        plt.title(f'{feature}: Real vs Predicted')
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# Plot a future forecast with history
def plot_future_forecast_with_history(history_and_forecast_df, features, save_filename="plots/future_forecast.png"):
    """
    Plots the history and future forecast of the provided DataFrame.

    Args:
    - history_and_forecast_df (pd.DataFrame): A DataFrame containing the history and forecast data.
    - features (list): A list of feature names present in the DataFrame.
    - save_filename (str, optional): The name of the file to save the plot. Defaults to "future_forecast.png".

    Returns:
    None
    """

    # Create two subplots
    fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(10, 8))

    # Adjust the axes list in case of only one feature
    if len(features) == 1:
        axes = [axes]

    for idx, feature in enumerate(features):
        # Construct the forecasted feature name
        forecasted_feature = "forecasted_" + feature

        # Plot feature and its forecasted value
        history_and_forecast_df[[feature, forecasted_feature]].plot(ax=axes[idx])
        axes[idx].set_title(f"{feature} and {forecasted_feature}")
        axes[idx].set_ylabel("Value")
        axes[idx].grid(True)
        axes[idx].legend(loc='upper left')

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(save_filename)
    plt.show()

# Function to create experiments and save models and artifacts
def log_experiment_to_mlflow(model, model_name, experiment_version, model_artifact_name,
                             epochs, batch_size, metrics, artifacts, history_and_forecast_df,
                             tf_saved_model_dir=None):
    """
    Log an experiment to MLflow.

    Parameters:
    - model: Model to be logged.
    - model_name: Name of the model.
    - experiment_version: Version of the experiment.
    - model_artifact_name: Name for the model artifact.
    - epochs: Number of epochs.
    - batch_size: Batch size.
    - metrics: Dictionary of metrics to be logged.
    - artifacts: List of artifact paths to be logged.
    - history_and_forecast_df: DataFrame to be saved as CSV and logged as an artifact.
    - tf_saved_model_dir (optional): Path to the TensorFlow saved model directory.
    """

    experiment_id = mlflow.create_experiment(f"{model_name}_v{experiment_version}_exp")

    with mlflow.start_run(experiment_id=experiment_id):
        # Log the model
        mlflow.sklearn.log_model(model, model_artifact_name)

        # Log the TensorFlow model (if provided)
        if tf_saved_model_dir:
            mlflow.tensorflow.log_model(tf_saved_model_dir=tf_saved_model_dir, artifact_path="model")

        # Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log artifacts
        for artifact in artifacts:
            mlflow.log_artifact(artifact)

        # Log CSV as an artifact
        csv_path = "outputs/history_and_forecast_df.csv"
        history_and_forecast_df.to_csv(csv_path, index=True)
        mlflow.log_artifact(csv_path, artifact_path="data")

# Loads an experimental MLFlow model
def load_mlflow_experimental_model(run_id, artifact_name):
    """
    Load a model from an MLflow experiment.

    Parameters:
    - run_id: ID of the MLflow run.
    - artifact_name: Name of the artifact containing the model.

    Returns:
    - Loaded model.
    """
    model_uri = f"runs:/{run_id}/{artifact_name}"
    model = mlflow.sklearn.load_model(model_uri)
    return model

# Loads a staged MLFlow model
def load_mlflow_staged_model(model_name, version, stage):
    """
    Load a model from MLflow that is in a specific stage (e.g., Staging or Production).

    Parameters:
    - model_name: Name of the model.
    - version: Version of the model.
    - stage: Stage of the model (e.g., "Staging" or "Production").

    Returns:
    - Loaded model.
    """
    model_uri = f"models:/{model_name}_v{version}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model
