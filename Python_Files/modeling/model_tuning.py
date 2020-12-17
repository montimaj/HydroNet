# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import tensorflow.keras.backend as kb
import tensorflow as tf
import numpy as np
import random as python_random
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Input, Activation, Dense, PReLU, Dropout, BatchNormalization
from tensorflow.keras.layers import LSTM, ConvLSTM2D, Conv1D
from tensorflow.keras.layers import Bidirectional, TimeDistributed, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel
from tensorflow import keras
from Python_Files.datalibs.sysops import make_proper_dir_name, makedirs, deletedirs


def r2(y_true, y_pred):
    """
    Calculate coefficient of determination using keras backend
    :param y_true: True values
    :param y_pred: Predicted values
    :return: R2 score
    """
    ss_res = kb.sum(kb.square(y_true - y_pred))
    ss_tot = kb.sum(kb.square(y_true - kb.mean(y_true)))
    return 1 - ss_res/(ss_tot + kb.epsilon())


def scikit_hypertuning(X_train_data, y_train_data, model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
                       grid_iter=10, random_state=0):
    """
    Creates a generalized algorithm pipeline and applies RandomizedSearchCV
    :param X_train_data: Training data
    :param y_train_data: Training labels
    :param model: Model object
    :param param_grid: Dictionary of hyperparameters
    :param cv: Number of cross-validation folds
    :param scoring_fit: Scoring metric
    :param grid_iter: Number of grid iterations
    :param random_state: PRNG seed
    :return: Best fit model
    """

    rs = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=cv, n_jobs=-2, scoring=scoring_fit,
                            verbose=2, random_state=random_state, n_iter=grid_iter)
    fitted_model = rs.fit(X_train_data, y_train_data)
    return fitted_model


class HydroHyperModel(HyperModel):

    def __init__(self, num_features, model_number=1, negative_values=False):
        """
        HyperModel child class constructor
        :param num_features: Number of features in training data
        :param model_number: Set model number, each model has different layers and parameters
        :param negative_values: Set True if data has negative values
        """

        super().__init__()
        self.num_features = num_features
        self.model_number = model_number
        self.negative_values = negative_values

    def build(self, hp):
        """
        This function is overriden
        :param hp: HyperModel object
        :return: None
        """

        model = None
        if self.model_number == 1:
            model = self.model1(hp)
        elif self.model_number == 2:
            model = self.model2(hp)
        # lr = hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])
        # beta_1 = hp.Choice('beta_1', [0.7, 0.8, 0.9])
        # beta_2 = hp.Choice('beta_2', [0.799, 0.899, 0.999])
        # momentum = hp.Choice('momentum', [0.0, 0.3, 0.5, 0.7, 0.9])
        # rho = hp.Choice('rho', [0.7, 0.8, 0.9])
        # epsilon = hp.Choice('epsilon', [1e-7, 1e-4, 1e-2, 0.1, 1])
        optimizer = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta'])
        optimizer_dict = {
            'adam': keras.optimizers.Adam(),
            'sgd': keras.optimizers.SGD(),
            'rmsprop': keras.optimizers.RMSprop(),
            'adagrad': keras.optimizers.Adagrad(),
            'adadelta': keras.optimizers.Adadelta(),
        }
        loss = hp.Choice('loss', ['mse', 'huber_loss'])
        model.compile(
            optimizer=optimizer_dict[optimizer],
            loss=loss,
            metrics=['mse', 'mae', r2])
        return model

    def model1(self, hp):
        """
        Custom model 1
        :param hp: HyperModel object
        :return: Keras model object
        """

        model = Sequential()
        input_activation_list = ['linear', 'relu', 'sigmoid', 'hard_sigmoid', 'softmax']
        if self.negative_values:
            input_activation_list += ['tanh', 'swish', 'softsign']
        input_activation = hp.Choice('input_activation', input_activation_list)
        if input_activation == 'swish':
            input_activation = tf.nn.swish
        init_units = hp.Choice('init_units', [128, 256, 512, 1024])
        model.add(Dense(units=init_units, activation=input_activation, input_shape=[self.num_features]))
        for i in range(hp.Int('num_layers', 1, 10)):
            units = hp.Int(
                'units_' + str(i),
                min_value=8,
                max_value=init_units,
                step=8
            )
            hidden_layer_activation = 'relu'
            if self.negative_values:
                alpha_leaky_relu = hp.Choice('alpha_leaky_relu', [0.1, 0.2, 0.3, 0.4])
                hidden_layer_activation = hp.Choice('hl_activation_' + str(i), ['relu', 'leaky_relu', 'swish'])
                if hidden_layer_activation == 'leaky_relu':
                    hidden_layer_activation = keras.layers.LeakyReLU(alpha=alpha_leaky_relu)
                elif hidden_layer_activation == 'swish':
                    hidden_layer_activation = tf.nn.swish
            model.add(BatchNormalization())
            model.add(Dense(units=units, activation=hidden_layer_activation))
            drop_rate = hp.Choice('drop_rate_' + str(i), [0.01, 0.05, 0.1, 0.15, 0.2])
            model.add(Dropout(rate=drop_rate))
        output_activation_list = ['tanh', 'swish', 'softsign']
        if not self.negative_values:
            output_activation_list = input_activation_list
        output_activation = hp.Choice('output_activation', output_activation_list)
        model.add(BatchNormalization())
        model.add(Dense(1, activation=output_activation))
        return model

    def model2(self, hp, hidden_units=(256, 128, 128, 128, 128, 256), output_features=1):
        """
        Custom model 2
        :param hp: HyperModel object
        :param hidden_units: Tupple of hidden units with each tuple value representing number of neurons in
        that hidden layer
        :param output_features: Number of output features
        :return: Keras model object
        """

        input_layer = Input(shape=(self.num_features,), name='input_layer',)
        hidden_layer = Dense(units=hidden_units[0],)(input_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Activation('relu')(hidden_layer)
        drop_rate = hp.Choice('drop_rate_' + str(0), [0.0, 0.01, 0.05, 0.08, 0.1])
        hidden_layer = Dropout(drop_rate)(hidden_layer)
        for unit in hidden_units[1:]:
            hidden_layer = Dense(units=unit,)(hidden_layer)
            hidden_layer = BatchNormalization()(hidden_layer)
            hidden_layer = Activation('relu')(hidden_layer)
            hidden_layer = Dropout(drop_rate)(hidden_layer)
        output_layer = Dense(units=output_features)(hidden_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Activation('relu')(output_layer)
        model = Model(input_layer, output_layer)
        return model


class HydroTuner(RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        """
        This is a overridden method
        :param trial: Trial object
        :param args: Additional arguments (defaults to RandomSearch arguments)
        :param kwargs: More arguments ((defaults to RandomSearch arguments)
        :return: None
        """

        batch_size = kwargs['batch_size']
        epochs = kwargs['epochs']
        if batch_size is None:
            kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 50, 1000, step=10)
        if epochs is None:
            kwargs['epochs'] = trial.hyperparameters.Int('epochs', 100, 1000, step=100)
        print('Batch Size, Epochs:', batch_size, epochs)
        super(HydroTuner, self).run_trial(trial, *args, **kwargs)


class KerasANN:
    """
    This class has been reproduced from https://github.com/cryptomare/RadarBackscatterModel
    Original authors: Abhisek Maiti, Shashwat Shukla
    Modifier: Sayantan Majumdar
    """

    def __init__(self, output_dir, input_features=6, hidden_units=(2048, 1024, 512, 256, 128, 64, 32, 16),
                 output_features=1, dropout=0.01, random_state=0, load_weights=None):
        """
        Constructor for KerasANN
        :param output_dir: Output directory for storing plots and history
        :param input_features: Number of input features
        :param hidden_units: Tupple of hidden units with each tuple value representing number of neurons in
        that hidden layer
        :param output_features: Number of output features
        :param dropout: Dropout probability
        :param random_state: PRNG seed
        :param load_weights: Set saved model file path to load weights from that pre-trained model
        """

        np.random.seed(random_state)
        python_random.seed(random_state)
        tf.random.set_seed(random_state)
        input_layer = Input(shape=(input_features,), name='input_layer',)
        hidden_layer = Dense(units=hidden_units[0], kernel_initializer='he_normal')(input_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = PReLU()(hidden_layer)
        hidden_layer = Dropout(dropout)(hidden_layer)
        for unit in hidden_units[1:]:
            hidden_layer = Dense(units=unit, kernel_initializer='he_normal')(hidden_layer)
            hidden_layer = BatchNormalization()(hidden_layer)
            hidden_layer = PReLU()(hidden_layer)
            hidden_layer = Dropout(dropout)(hidden_layer)
        output_layer = Dense(units=output_features, kernel_initializer='he_normal')(hidden_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Activation('relu')(output_layer)
        self._model = Model(input_layer, output_layer)
        if load_weights:
            saved_model_weights = store_load_keras_model(load_weights).get_weights()
            self._model.set_weights(saved_model_weights)
        self._input_features = input_features
        self._output_features = output_features
        self._output_dir = make_proper_dir_name(output_dir + 'KerasANN_History')
        deletedirs([self._output_dir])
        makedirs([self._output_dir])
        self._is_ready = False
        self._is_trained = False

    def ready(self, optimizer='adam', loss='mse', metrics=('mse', 'mae')):
        """
        Compiles model object
        :param optimizer: Keras optimizer function
        :param loss: Keras loss function
        :param metrics: Keras metric
        :return: None
        """

        optimizer_dict = {
            'adam': keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-7),
            'sgd': keras.optimizers.SGD(momentum=0.3),
            'rmsprop': keras.optimizers.RMSprop(centered=True, momentum=0.3),
            'adagrad': keras.optimizers.Adagrad(),
            'adadelta': keras.optimizers.Adadelta(),
            'nadam': keras.optimizers.Nadam()
        }

        self._model.compile(
            optimizer=optimizer_dict[optimizer],
            loss=loss,
            metrics=list(metrics)
        )
        print(self._model.summary())
        plot_model(self._model, to_file=self._output_dir + 'model_plot.png', show_shapes=True, show_layer_names=True,
                   dpi=330)
        self._is_ready = True

    def learn(self, x_train, x_test, y_train, y_test, batch_size=1000, epochs=5, fold_count=10, repeats=10):
        """
        Call this to build model
        :param x_train: Training data as numpy array
        :param x_test: Test data as numpy array
        :param y_train: Training labels as numpy array
        :param y_test: Test labels as numpy array
        :param batch_size: Batch size
        :param epochs: Epochs
        :param fold_count: Number of cross-validation folds
        :param repeats: KFold repeats
        :return: Trained model object
        """

        assert self._is_ready
        kfold = RepeatedKFold(n_splits=fold_count, n_repeats=repeats, random_state=42)
        csv_logger = CSVLogger(self._output_dir + 'Model_History.csv', append=True)
        for train, validation in kfold.split(x_train, y_train):
            self._model.fit(
                x=x_train[train],
                y=y_train[train],
                validation_data=(x_train[validation], y_train[validation]),
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                callbacks=[EarlyStopping('val_loss', patience=50), csv_logger]
            )
        test_scores = self._model.evaluate(x=x_test, y=y_test, verbose=1)
        print('Test Scores\n', test_scores)
        self._is_trained = True
        return self._model


class TreeML:

    def __init__(self, output_dir, ml_model='RF', random_state=0):
        """
        Constructor for KerasANN
        :param output_dir: Output directory for storing plots and history
        :param ml_model: Set ML model, models include 'RF', 'ETR', 'XGB'
        :param random_state: PRNG seed
        """

        self._output_dir = make_proper_dir_name(output_dir + 'TreeML_History/' + ml_model)
        deletedirs([self._output_dir])
        makedirs([self._output_dir])
        print('Performing', ml_model)
        if ml_model == 'RF':
            self._model = RandomForestRegressor(n_estimators=500, n_jobs=-2, random_state=random_state)
        elif ml_model == 'ETR':
            self._model = ExtraTreesRegressor(n_estimators=500, n_jobs=-2, random_state=random_state, bootstrap=True)
        else:
            self._model = XGBRegressor(n_estimators=20000, n_jobs=-2, eta=1e-3, random_state=random_state,
                                       objective='reg:squarederror', tree_method='hist', grow_policy='lossguide',
                                       rate_drop=0.01)

    def learn(self, x_train, x_test, y_train, y_test):
        """
        Call this to build model
        :param x_train: Training data as numpy array
        :param x_test: Test data as numpy array
        :param y_train: Training labels as numpy array
        :param y_test: Test labels as numpy array
        :return: Trained model object
        """

        self._model.fit(x_train, y_train)
        train_score = np.round(self._model.score(x_train, y_train), 2)
        test_score = np.round(self._model.score(x_test, y_test), 2)
        print(train_score, test_score)
        return self._model


class HydroLSTM:
    def __init__(self, x_train, x_test, y_train, y_test, output_dir, timesteps=1):
        """
        Constructor for class
        :param x_train: Training data as numpy array
        :param x_test: Test data as numpy array
        :param y_train: Training labels as numpy array
        :param y_test: Test labels as numpy array
        :param output_dir: Output directory for storing plots and history
        :param timesteps: Number of timesteps
        """

        self.x_train = x_train.reshape(x_train.shape[0], timesteps, x_train.shape[1])
        self.x_test = x_test.reshape(x_test.shape[0], timesteps, x_test.shape[1])
        self.x_train_cnn_lstm = x_train.reshape(x_train.shape[0], 1, timesteps, x_train.shape[1])
        self.x_test_cnn_lstm = x_test.reshape(x_test.shape[0], 1, timesteps, x_test.shape[1])
        self.x_train_conv_lstm = x_train.reshape(x_train.shape[0], 1, 1, timesteps, x_train.shape[1])
        self.x_test_conv_lstm = x_test.reshape(x_test.shape[0], 1, 1, timesteps, x_test.shape[1])
        self.y_train = y_train
        self.y_test = y_test
        self.output_dir = make_proper_dir_name(output_dir + 'LSTM_History')
        deletedirs([self.output_dir])
        makedirs([self.output_dir])
        self.timesteps = timesteps
        self.n_seq = 1
        self.model = None

    def vanilla_lstm(self, units=1024, dropout=0.01):
        """
        Implements vanilla LSTM
        :param units: Number of units in LSTM
        :param dropout: Dropout probability
        :return: Model object
        """

        model = Sequential()
        model.add(LSTM(units, activation='relu', input_shape=(self.x_train.shape[1], self.x_train.shape[2]),
                       dropout=dropout))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='relu'))
        self.model = model
        return model

    def vanilla_lstm_ann(self, lstm_units=50, hidden_units=(256, 128, 128, 128, 128, 256), dropout=0.1):
        """
        Implements vanilla LSTM
        :param lstm_units: Number of units in LSTM
        :param hidden_units: Number of units in KerasANN
        :param dropout: Dropout probability
        :return: Model object
        """

        model = Sequential()
        model.add(LSTM(lstm_units, activation='relu', input_shape=(self.x_train.shape[1], self.x_train.shape[2]),
                       dropout=dropout))
        model.add(BatchNormalization())
        model.add(Dense(units=1, activation='relu'))
        for unit in hidden_units:
            model.add(BatchNormalization())
            model.add(Dense(units=unit, activation='relu'))
            model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Dense(units=1, activation='relu'))
        self.model = model
        return model

    def stacked_lstm(self, units=(1024, 1024), dropout=0.01, bidirectional=False):
        """
        Implements stacked LSTM
        :param units: Tuple of units in the LSTM, tuple length indicates the total number of stacks used
        :param dropout: Dropout probability
        :param bidirectional: Set True to use bidrectional LSTM
        :return: Model object
        """

        model = Sequential()
        if not bidirectional:
            model.add(LSTM(units[0], activation='relu', return_sequences=True, dropout=dropout,
                           input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
        else:
            model.add(Bidirectional(LSTM(units[0], activation='relu', dropout=dropout, return_sequences=True),
                                    input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
        for unit in units[1:-1]:
            model.add(BatchNormalization())
            lstm = LSTM(unit, activation='relu', return_sequences=True, dropout=dropout)
            if not bidirectional:
                model.add(lstm)
            else:
                model.add(Bidirectional(lstm))
        model.add(BatchNormalization())
        lstm = LSTM(units[-1], activation='relu', dropout=dropout)
        if not bidirectional:
            model.add(lstm)
        else:
            model.add(Bidirectional(lstm))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='relu'))
        self.model = model
        return model

    def cnn_lstm(self, filters=64, kernel_size=1, units=(50, 50, 50, 50, 50), dropout=0.01, bidirectional=False):
        """
        Implements CNN LSTM
        :param filters: Number of Conv1D filters
        :param kernel_size: 1D kernel size for Conv1D
        :param units: Tuple of units in the LSTM, tuple length indicates the total number of stacks used
        :param dropout: Dropout probability
        :param bidirectional: Set True to use bidrectional LSTM
        :return: Model object
        """

        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
                                  input_shape=(None, self.timesteps, self.x_train_cnn_lstm.shape[-1])))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(BatchNormalization())
        model.add(self.stacked_lstm(units, dropout, bidirectional))
        self.x_train = self.x_train_cnn_lstm
        self.x_test = self.x_test_cnn_lstm
        self.model = model
        return model

    def conv_lstm(self, filters=64, kernel_size=(1, 2)):
        """
        Implements ConvLSTM
        :param filters: Number of Conv2D filters
        :param kernel_size: 2D kernel size for Conv2D
        :return: Model object
        """

        model = Sequential()
        model.add(
            ConvLSTM2D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                input_shape=(self.n_seq, 1, self.timesteps, self.x_train_conv_lstm.shape[-1])
            )
        )
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(1, activation='relu'))
        self.x_train = self.x_train_conv_lstm
        self.x_test = self.x_test_conv_lstm
        self.model = model
        return model

    def ready(self, optimizer='adam', loss='huber_loss', metrics=('mse', 'mae', r2)):
        """
        Compiles model object
        :param optimizer: Keras optimizer function
        :param loss: Keras loss function
        :param metrics: Keras metric
        :return: None
        """

        optimizer_dict = {
            'adam': keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-7),
            'sgd': keras.optimizers.SGD(momentum=0.3),
            'rmsprop': keras.optimizers.RMSprop(centered=True, momentum=0.3),
            'adagrad': keras.optimizers.Adagrad(),
            'adadelta': keras.optimizers.Adadelta(),
            'nadam': keras.optimizers.Nadam()
        }

        self.model.compile(
            optimizer=optimizer_dict[optimizer],
            loss=loss,
            metrics=list(metrics)
        )
        print(self.model.summary())
        plot_model(self.model, to_file=self.output_dir + 'model_plot.png', show_shapes=True, show_layer_names=True,
                   dpi=330)

    def learn(self, batch_size=1000, epochs=5, fold_count=10, repeats=10):
        """
        Call this to build model
        :param batch_size: Batch size
        :param epochs: Epochs
        :param fold_count: Number of cross-validation folds
        :param repeats: KFold repeats
        :return: Trained model object
        """

        kfold = RepeatedKFold(n_splits=fold_count, n_repeats=repeats, random_state=42)
        csv_logger = CSVLogger(self.output_dir + 'LSTM_History.csv', append=True)
        for train, validation in kfold.split(self.x_train, self.y_train):
            self.model.fit(
                x=self.x_train[train],
                y=self.y_train[train],
                validation_data=(self.x_train[validation], self.y_train[validation]),
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                shuffle=True,
                callbacks=[EarlyStopping('val_loss', patience=50), csv_logger]
            )
        test_scores = self.model.evaluate(x=self.x_test, y=self.y_test, verbose=1)
        print('Test Scores\n', test_scores)
        return self.model


def store_load_keras_model(output_file, model=None):
    """
    Save model
    :param output_file: Output file path
    :param model: Model object, set None when loading
    :return: None
    """

    if model:
        model.save(filepath=output_file, include_optimizer=True)
    else:
        return keras.models.load_model(output_file)
