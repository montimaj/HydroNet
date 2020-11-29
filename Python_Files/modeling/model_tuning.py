# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import tensorflow.keras.backend as kb
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel
from tensorflow import keras


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

    def __init__(self, num_features, negative_values=False):
        """
        HyperModel child class constructor
        :param num_features: Number of features in training data
        :param negative_values: Set True if data has negative values
        """

        super().__init__()
        self.num_features = num_features
        self.negative_values = negative_values

    def build(self, hp):
        """
        This function is overriden
        :param hp: HyperModel object
        :return: None
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
        lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])
        beta_1 = hp.Choice('beta_1', [0.7, 0.8, 0.9])
        beta_2 = hp.Choice('beta_2', [0.799, 0.899, 0.999])
        epsilon = hp.Choice('epsilon', [1e-7, 1e-8, 1e-9])
        momentum = hp.Choice('momentum', [0.0, 0.3, 0.5, 0.7, 0.9])
        rho = hp.Choice('rho', [0.7, 0.8, 0.9])
        optimizer = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta'])
        optimizer_dict = {
            'adam': keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon),
            'sgd': keras.optimizers.SGD(learning_rate=lr, momentum=momentum),
            'rmsprop': keras.optimizers.RMSprop(learning_rate=lr, rho=rho, momentum=momentum, epsilon=epsilon),
            'adagrad': keras.optimizers.Adagrad(learning_rate=lr, epsilon=epsilon),
            'adadelta': keras.optimizers.Adadelta(learning_rate=lr, rho=rho, epsilon=epsilon),
        }
        loss = hp.Choice('loss', ['mse', 'mae', 'huber_loss'])
        model.compile(
            optimizer=optimizer_dict[optimizer],
            loss=loss,
            metrics=['mse', 'mae', r2])
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
        super(HydroTuner, self).run_trial(trial, *args, **kwargs)


class KerasANN:
    """
    This class has been reproduced from https://github.com/cryptomare/RadarBackscatterModel
    Original authors: Abhisek Maiti, Shashwat Shukla
    Modifier: Sayantan Majumdar
    """
    def __init__(self, input_features=6, hidden_units=(128, 64, 32), output_features=1, droupout=0.01):
        """
        Constructor for KerasANN
        :param input_features: Number of input features
        :param hidden_units: Tupple of hidden units with each tuple value representing number of neurons in
        that hidden layer
        :param output_features: Number of output features
        :param droupout: Droupout probability
        """

        input_layer = Input(
                shape=(input_features,),
                name='input_layer',
            )
        hidden_layer = Dense(
                units=hidden_units[0],
            )(input_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Activation('relu')(hidden_layer)
        hidden_layer = Dropout(droupout)(hidden_layer)
        for unit in hidden_units[1:]:
            hidden_layer = Dense(
                    units=unit,
                )(hidden_layer)
            hidden_layer = BatchNormalization()(hidden_layer)
            hidden_layer = Activation('relu')(hidden_layer)
            hidden_layer = Dropout(droupout)(hidden_layer)

        output_layer = Dense(
                    units=output_features,
                )(hidden_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Activation('relu')(output_layer)
        self._model = Model(input_layer, output_layer)
        self._input_features = input_features
        self._output_features = output_features
        self._is_ready = False
        self._is_trained = False

    def ready(self, optimizer='adam', loss='mse', metrics=('mse', 'mae', r2)):
        """
        Compiles model object
        :param optimizer: Keras optimizer function
        :param loss: Keras loss function
        :param metrics: Keras metric
        :return: None
        """

        self._model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=list(metrics)
        )
        self._is_ready = True

    def learn(self, x_train, x_test, y_train, y_test, batch_size=1000, epochs=5, fold_count=3):
        """
        Call this to build model
        :param x_train: Training data as numpy array
        :param x_test: Test data as numpy array
        :param y_train: Training labels as numpy array
        :param y_test: Test labels as numpy array
        :param batch_size: Batch size
        :param epochs: Epochs
        :param fold_count: Number of cross-validation folds
        :return: Trained model object
        """

        assert self._is_ready
        kfold = KFold(n_splits=fold_count, shuffle=True, random_state=23)
        for train, validation in kfold.split(x_train, y_train):
            self._model.fit(
                x=x_train[train],
                y=y_train[train],
                batch_size=batch_size,
                epochs=epochs,
                verbose=1
            )
            scores = self._model.evaluate(
                x_train[validation],
                y_train[validation],
                verbose=1,
                batch_size=batch_size
            )
            print(scores)
        test_scores = self._model.evaluate(x=x_test, y=y_test, verbose=1)
        print('Test Scores\n', test_scores)
        self._is_trained = True
        return self._model


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
