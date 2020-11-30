# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import sklearn.utils as sk
import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow import keras
from glob import glob
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from Python_Files.datalibs import rasterops as rops
from Python_Files.modeling import model_analysis as ma
from Python_Files.modeling.model_tuning import HydroHyperModel, HydroTuner, KerasANN
from Python_Files.modeling.model_tuning import scikit_hypertuning, store_load_keras_model


def create_dataframe(input_file_dir, out_df, column_names=None, pattern='*.tif', exclude_years=(), exclude_vars=(),
                     make_year_col=True, ordering=False, remove_na=True):
    """
    Create dataframe from file list
    :param input_file_dir: Input directory where the file names begin with <Variable>_<Year>, e.g, ET_2015.tif
    :param out_df: Output Dataframe file
    :param column_names: Dataframe column names, these must be df headers
    :param pattern: File pattern to look for in the folder
    :param exclude_years: Exclude these years from the dataframe
    :param exclude_vars: Exclude these variables from the dataframe
    :param make_year_col: Make a dataframe column entry for year
    :param ordering: Set True to order dataframe column names
    :param remove_na: Set False to disable NA removal
    :return: Pandas dataframe
    """

    raster_file_dict = defaultdict(lambda: [])
    for f in glob(input_file_dir + pattern):
        sep = f.rfind('_')
        variable, year = f[f.rfind(os.sep) + 1: sep], f[sep + 1: f.rfind('.')]
        if variable not in exclude_vars and int(year) not in exclude_years:
            raster_file_dict[int(year)].append(f)

    raster_dict = {}
    flag = False
    years = sorted(list(raster_file_dict.keys()))
    df = None
    raster_arr = None
    for year in years:
        file_list = raster_file_dict[year]
        for raster_file in file_list:
            raster_arr = rops.read_raster_as_arr(raster_file, get_file=False)
            raster_arr = raster_arr.reshape(raster_arr.shape[0] * raster_arr.shape[1])
            variable = raster_file[raster_file.rfind(os.sep) + 1: raster_file.rfind('_')]
            raster_dict[variable] = raster_arr
        if make_year_col:
            raster_dict['YEAR'] = [year] * raster_arr.shape[0]
        if not flag:
            df = pd.DataFrame(data=raster_dict)
            flag = True
        else:
            df = df.append(pd.DataFrame(data=raster_dict))

    if remove_na:
        df = df.dropna(axis=0)
    df = reindex_df(df, column_names=column_names, ordering=ordering)
    df.to_csv(out_df, index=False)
    return df


def reindex_df(df, column_names, ordering=False):
    """
    Reindex dataframe columns
    :param df: Input dataframe
    :param column_names: Dataframe column names, these must be df headers
    :param ordering: Set True to apply ordering
    :return: Reindexed dataframe
    """
    if not column_names:
        column_names = df.columns
        ordering = True
    if ordering:
        column_names = sorted(column_names)
    return df.reindex(column_names, axis=1)


def get_rf_model(rf_file):
    """
    Get existing RF model object
    :param rf_file: File path to RF model
    :return: RandomForestRegressor
    """

    return pickle.load(open(rf_file, mode='rb'))


def split_data_train_test(input_df, pred_attr='GW', shuffle=True, random_state=0, test_size=0.2, outdir=None,
                          drop_attrs=(), test_year=None):
    """
    Split data preserving temporal variations
    :param input_df: Input dataframe
    :param pred_attr: Prediction attribute name
    :param shuffle: Default True for shuffling
    :param random_state: Random state used during train test split
    :param test_size: Test data size percentage (0<=test_size<=1)
    :param outdir: Set path to store intermediate files
    :param drop_attrs: Drop these specified attributes
    :param test_year: Build test data from only this year
    :return: X_train, X_test, y_train, y_test
    """

    years = set(input_df['YEAR'])
    x_train_df = pd.DataFrame()
    x_test_df = pd.DataFrame()
    y_train_df = pd.DataFrame()
    y_test_df = pd.DataFrame()
    flag = False
    if test_year in years:
        flag = True
    drop_columns = [pred_attr] + list(drop_attrs)
    for year in years:
        selected_data = input_df.loc[input_df['YEAR'] == year]
        y = selected_data[pred_attr]
        selected_data = selected_data.drop(columns=drop_columns)
        x_train, x_test, y_train, y_test = train_test_split(selected_data, y, shuffle=shuffle,
                                                            random_state=random_state, test_size=test_size)
        x_train_df = x_train_df.append(x_train)
        if (flag and test_year == year) or not flag:
            x_test_df = x_test_df.append(x_test)
            y_test_df = pd.concat([y_test_df, y_test])
        y_train_df = pd.concat([y_train_df, y_train])

    if outdir:
        x_train_df.to_csv(outdir + 'X_Train.csv', index=False)
        x_test_df.to_csv(outdir + 'X_Test.csv', index=False)
        y_train_df.to_csv(outdir + 'Y_Train.csv', index=False)
        y_test_df.to_csv(outdir + 'Y_Test.csv', index=False)

    return x_train_df, x_test_df, y_train_df[0].ravel(), y_test_df[0].ravel()


def split_yearly_data(input_df, pred_attr='GW', outdir=None, drop_attrs=(), test_years=(2016,), shuffle=True,
                      random_state=0):
    """
    Split data based on the years
    :param input_df: Input dataframe
    :param pred_attr: Prediction attribute name
    :param outdir: Set path to store intermediate files
    :param drop_attrs: Drop these specified attributes
    :param test_years: Build test data from only these years
    :param shuffle: Set False to stop data shuffling
    :param random_state: Seed for PRNG
    :return: X_train, X_test, y_train, y_test
    """

    years = set(input_df['YEAR'])
    x_train_df = pd.DataFrame()
    x_test_df = pd.DataFrame()
    y_train_df = pd.DataFrame()
    y_test_df = pd.DataFrame()
    drop_columns = [pred_attr] + list(drop_attrs)
    for year in years:
        selected_data = input_df.loc[input_df['YEAR'] == year]
        y_t = selected_data[pred_attr]
        x_t = selected_data.drop(columns=drop_columns)
        if year not in test_years:
            x_train_df = x_train_df.append(x_t)
            y_train_df = pd.concat([y_train_df, y_t])
        else:
            x_test_df = x_test_df.append(x_t)
            y_test_df = pd.concat([y_test_df, y_t])

    if shuffle:
        x_train_df = sk.shuffle(x_train_df, random_state=random_state)
        y_train_df = sk.shuffle(y_train_df, random_state=random_state)
        x_test_df = sk.shuffle(x_test_df, random_state=random_state)
        y_test_df = sk.shuffle(y_test_df, random_state=random_state)

    if outdir:
        x_train_df.to_csv(outdir + 'X_Train.csv', index=False)
        x_test_df.to_csv(outdir + 'X_Test.csv', index=False)
        y_train_df.to_csv(outdir + 'Y_Train.csv', index=False)
        y_test_df.to_csv(outdir + 'Y_Test.csv', index=False)

    return x_train_df, x_test_df, y_train_df.to_numpy().ravel(), y_test_df.to_numpy().ravel()


def scale_df(input_df, output_dir, load_data=False):
    """
    Scale dataframe columns
    :param input_df: Input Pandas dataframe
    :param output_dir: Output directory to store scaled df and object
    :param load_data: Set True to load existing df and scaler object
    :return: Scaled scaled df and scaler object as tuples
    """

    scaled_csv = output_dir + 'Scaled_DF.csv'
    scaler_obj_file = output_dir + 'Scaler_Obj'
    if load_data:
        scaled_df, scaler = pd.read_csv(scaled_csv), pickle.load(open(scaler_obj_file, mode='rb'))
    else:
        scaler = MinMaxScaler()
        scaled_df = input_df.copy()
        scaled_df[scaled_df.columns] = scaler.fit_transform(input_df[input_df.columns])
        scaled_df.to_csv(scaled_csv, index=False)
        pickle.dump(scaler, open(scaler_obj_file, mode='wb'))
    return scaled_df, scaler


def split_data(input_df, output_dir, pred_attr='GW', shuffle=False, drop_attrs=(), test_year=(2012,), test_size=0.2,
               split_yearly=True, random_state=0, load_data=False):
    """

    :param input_df: Input Pandas dataframe
    :param output_dir: Output directory
    :param pred_attr: Target attribute
    :param shuffle: Set True to enabling data shuffling
    :param drop_attrs: Drop these specified attributes
    :param test_year: Build test data from only this year(s)
    :param test_size: Required only if split_yearly=False
    :param split_yearly: Split train and test data based on years
    :param random_state: PRNG seed for reproducibility
    :param load_data: Set True to load existing scaled train and test data
    :return: None
    """

    if load_data:
        x_train = pd.read_csv(output_dir + 'X_Train.csv')
        y_train = pd.read_csv(output_dir + 'Y_Train.csv')
        y_train = y_train.to_numpy().ravel()
        x_test = pd.read_csv(output_dir + 'X_Test.csv')
        y_test = pd.read_csv(output_dir + 'Y_Test.csv')
        y_test = y_test.to_numpy().ravel()
    else:
        if not split_yearly:
            x_train, x_test, y_train, y_test = split_data_train_test(input_df, pred_attr=pred_attr, test_size=test_size,
                                                                     random_state=random_state, shuffle=shuffle,
                                                                     outdir=output_dir, drop_attrs=drop_attrs)
        else:
            x_train, x_test, y_train, y_test = split_yearly_data(input_df, pred_attr=pred_attr, outdir=output_dir,
                                                                 drop_attrs=drop_attrs, test_years=test_year,
                                                                 shuffle=shuffle, random_state=random_state)
    return x_train, x_test, y_train, y_test


def perform_kerasregression(X_train_data, X_test_data, y_train_data, y_test_data, output_dir, max_trials=20,
                            max_exec_trial=5, validation_split=0.1, batch_size=None, epochs=None, random_state=0,
                            use_keras_tuner=True, load_model=False, model_number=2):
    """
    Perform regression using Tensorflow and Keras
    :param X_train_data: Training data as Pandas dataframe
    :param X_test_data: Test data as Pandas dataframe
    :param y_train_data: Training labels as numpy array
    :param y_test_data:Test labels as numpy array
    :param output_dir: Output directory to dump the best-fit model
    :param max_trials: Maximum Keras Tuner trials
    :param max_exec_trial: Maximum executions per trial for Keras Tuner
    :param validation_split: Amount of validation data to set aside during training
    :param batch_size: Set a positive value. By default, batch size is auto-tuned
    :param epochs: Set a positive value. By default, epochs is auto-tuned
    :param random_state: PRNG seed
    :param use_keras_tuner: Set False to use KerasANN without auto hypertuning
    :param load_model: Set True to load existing model. Load model won't work with custom metric in TF 2.1.0
    :param model_number: Set model number for Keras-Tuner
    :return: Fitted model and prediction statistics
    """

    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    kerastuner_output_file = output_dir + 'KerasTunerANN.tf'
    kerasann_output_file = output_dir + 'KerasANN.tf'
    if not load_model:
        num_features = X_train_data.shape[1]
        if use_keras_tuner:
            objective_func = 'mse'
            project_name = 'HydroNet_Keras'
            negative_values = (X_train_data.values < 0).any()
            hypermodel = HydroHyperModel(num_features, model_number, negative_values)
            tuner = HydroTuner(
                hypermodel,
                objective=objective_func,
                max_trials=max_trials,
                executions_per_trial=max_exec_trial,
                directory=output_dir,
                project_name=project_name,
                seed=random_state
            )
            print(tuner.search_space_summary())
            tuner.search(X_train_data, y_train_data, validation_split=validation_split, batch_size=batch_size,
                         epochs=epochs, callbacks=[keras.callbacks.EarlyStopping(objective_func, patience=5)])
            trained_model = tuner.get_best_models()[0]
            store_load_keras_model(model=trained_model, output_file=kerastuner_output_file)
        else:
            keras_ann = KerasANN(input_features=num_features, output_features=1)
            keras_ann.ready()
            trained_model = keras_ann.learn(X_train_data.to_numpy(), X_test_data.to_numpy(), y_train_data, y_test_data,
                                            batch_size=batch_size, epochs=epochs, fold_count=max_exec_trial)
            store_load_keras_model(model=trained_model, output_file=kerasann_output_file)
    else:
        if use_keras_tuner:
            trained_model = store_load_keras_model(output_file=kerastuner_output_file)
        else:
            trained_model = store_load_keras_model(output_file=kerasann_output_file)

    print(trained_model.summary())
    print('Keras Regressor')
    if use_keras_tuner:
        print(trained_model.evaluate(X_test_data, y_test_data))
    pred = trained_model.predict(X_test_data)
    ma.generate_scatter_plot(y_test_data, pred)
    test_stats = ma.get_error_stats(y_test_data, pred)
    pred = trained_model.predict(X_train_data)
    ma.generate_scatter_plot(y_train_data, pred)
    train_stats = ma.get_error_stats(y_train_data, pred)
    print('Train, Test Stats:', train_stats, test_stats)
    return trained_model


def perform_linearregression(X_train_data, X_test_data, y_train_data, y_test_data):
    """
    Perform linear regression
    :param X_train_data: Training data
    :param X_test_data: Test data
    :param y_train_data: Training labels
    :param y_test_data:Test labels
    :return: Fitted model and prediction statistics
    """

    lreg = LinearRegression(n_jobs=-2).fit(X_train_data, y_train_data)
    pred = lreg.predict(X_test_data)
    pred_stats = ma.get_error_stats(y_test_data, pred)
    print('Linear regression:', pred_stats)
    return lreg


def perform_mlpregression(X_train_data, X_test_data, y_train_data, y_test_data, output_dir, cv=10, grid_iter=10,
                          scoring_fit='neg_mean_squared_error', random_state=0, load_model=False):
    """
    Perform regression using scikit-learn MLPRegressor
    :param X_train_data: Training data
    :param X_test_data: Test data
    :param y_train_data: Training labels
    :param y_test_data:Test labels
    :param output_dir: Output directory to dump the best-fit model
    :param cv: Number of cross-validation folds
    :param grid_iter: Number of grid iterations
    :param scoring_fit: Scoring metric
    :param random_state: PRNG seed
    :param load_model: Set True to load existing best-fit model
    :return: Fitted model and prediction statistics
    """

    if not load_model:
        mlp_regressor = MLPRegressor()
        np.random.seed(random_state)
        param_grid = {
            'hidden_layer_sizes': [(128,), (128, 64), (128, 64, 32), (128, 64, 32, 8), (128, 64, 32, 8, 4)],
            'activation': ['logistic', 'relu'],
            'solver': ['adam', 'sgd'],
            'alpha': [0, 1e-3, 1e-4],
            'batch_size': [200, 500, 1000, 5000],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': [1e-3, 1e-4],
            'power_t': np.random.uniform(0.1, 1, 10).tolist(),
            'max_iter': [500, 1000, 1500],
            'random_state': [0],
            'tol': [1e-4],
            'momentum': np.random.uniform(0.1, 1, 10).tolist(),
            'beta_1': np.random.uniform(0.1, 1, 10).tolist(),
            'beta_2': np.random.uniform(0.1, 1, 10).tolist()
        }
        mlp_regressor = scikit_hypertuning(X_train_data, y_train_data, mlp_regressor, param_grid, cv, scoring_fit,
                                           grid_iter, random_state)
        pickle.dump(mlp_regressor, open(output_dir + 'MLP_model', mode='wb'))
    else:
        mlp_regressor = pickle.load(open(output_dir + 'MLP_model', mode='rb'))
    print(np.sqrt(-mlp_regressor.best_score_))
    print(mlp_regressor.best_params_)
    pred = mlp_regressor.predict(X_test_data)
    pred_stats = ma.get_error_stats(y_test_data, pred)
    print('MLPRegressor:', pred_stats)
    return mlp_regressor


def create_pred_raster(rf_model, out_raster, actual_raster_dir, column_names=None, exclude_vars=(), pred_year=2015,
                       pred_attr='GW', drop_attrs=(), only_pred=False, calculate_errors=True, ordering=False):
    """
    Create prediction raster
    :param rf_model: Pre-built Random Forest Model
    :param out_raster: Output raster
    :param actual_raster_dir: Ground truth raster files required for prediction
    :param column_names: Dataframe column names, these must be df headers
    :param exclude_vars: Exclude these variables from the model prediction and analysis
    :param pred_year: Prediction year
    :param pred_attr: Prediction attribute name in the dataframe
    :param drop_attrs: Drop these specified attributes (Must be exactly the same as used in rf_regressor module)
    :param only_pred: Set True to disable raster creation and for showing only the error metrics,
    automatically set to False if calculate_errors is False
    :param calculate_errors: Calculate error metrics if actual observations are present
    :param ordering: Set True to order dataframe column names
    :return: MAE, RMSE, and R^2 statistics (rounded to 2 decimal places)
    """

    raster_files = glob(actual_raster_dir + '*_' + str(pred_year) + '*.tif')
    raster_arr_dict = {}
    nan_pos_dict = {}
    actual_file = None
    raster_shape = None
    for raster_file in raster_files:
        sep = raster_file.rfind('_')
        variable, year = raster_file[raster_file.rfind(os.sep) + 1: sep], raster_file[sep + 1: raster_file.rfind('.')]
        if variable not in exclude_vars:
            raster_arr, actual_file = rops.read_raster_as_arr(raster_file)
            raster_shape = raster_arr.shape
            raster_arr = raster_arr.reshape(raster_shape[0] * raster_shape[1])
            nan_pos_dict[variable] = np.isnan(raster_arr)
            if not only_pred:
                raster_arr[nan_pos_dict[variable]] = 0
            raster_arr_dict[variable] = raster_arr
            raster_arr_dict['YEAR'] = [year] * raster_arr.shape[0]

    input_df = pd.DataFrame(data=raster_arr_dict)
    input_df = input_df.dropna(axis=0)
    input_df = reindex_df(input_df, column_names=column_names, ordering=ordering)
    drop_columns = [pred_attr] + list(drop_attrs)
    if not calculate_errors:
        drop_cols = drop_columns
        if not column_names:
            drop_cols.remove(pred_attr)
        input_df = input_df.drop(columns=drop_cols)
        pred_arr = rf_model.predict(input_df)
        if not only_pred:
            for nan_pos in nan_pos_dict.values():
                pred_arr[nan_pos] = actual_file.nodata
        mae, rmse, r2_score, nrmse, nmae = (np.nan,) * 5
    else:
        if only_pred:
            actual_arr = input_df[pred_attr]
        else:
            actual_arr = raster_arr_dict[pred_attr]
        input_df = input_df.drop(columns=drop_columns)
        pred_arr = rf_model.predict(input_df)
        if not only_pred:
            for nan_pos in nan_pos_dict.values():
                actual_arr[nan_pos] = actual_file.nodata
                pred_arr[nan_pos] = actual_file.nodata
            actual_values = actual_arr[actual_arr != actual_file.nodata]
            pred_values = pred_arr[pred_arr != actual_file.nodata]
        else:
            actual_values = actual_arr
            pred_values = pred_arr
        r2_score, mae, rmse, nmae, nrmse = ma.get_error_stats(actual_values, pred_values)
    if not only_pred:
        pred_arr = pred_arr.reshape(raster_shape)
        rops.write_raster(pred_arr, actual_file, transform=actual_file.transform, outfile_path=out_raster)
    return mae, rmse, r2_score, nrmse, nmae


def predict_rasters(rf_model, actual_raster_dir, out_dir, pred_years, column_names=None, drop_attrs=(), pred_attr='GW',
                    only_pred=False, exclude_vars=(), exclude_years=(2019,), ordering=False):
    """
    Create prediction rasters from input data
    :param rf_model: Pre-trained Random Forest Model
    :param actual_raster_dir: Directory containing input rasters
    :param out_dir: Output directory for predicted rasters
    :param pred_years: Tuple containing prediction years
    :param column_names: Dataframe column names, these must be df headers
    :param drop_attrs: Drop these specified attributes (Must be exactly the same as used in rf_regressor module)
    :param pred_attr: Prediction Attribute
    :param only_pred: Set true to disable raster creation and for showing only the error metrics
    :param exclude_vars: Exclude these variables from the model prediction
    :param exclude_years: Exclude these years from error analysis, only the respective predicted rasters are generated
    :param ordering: Set True to order dataframe column names
    :return: None
    """

    for pred_year in pred_years:
        out_pred_raster = out_dir + 'pred_' + str(pred_year) + '.tif'
        calculate_errors = True
        if pred_year in exclude_years:
            calculate_errors = False
        mae, rmse, r_squared, normalized_rmse, normalized_mae = create_pred_raster(rf_model, out_raster=out_pred_raster,
                                                                                   actual_raster_dir=actual_raster_dir,
                                                                                   exclude_vars=exclude_vars,
                                                                                   pred_year=pred_year,
                                                                                   drop_attrs=drop_attrs,
                                                                                   pred_attr=pred_attr,
                                                                                   only_pred=only_pred,
                                                                                   calculate_errors=calculate_errors,
                                                                                   column_names=column_names,
                                                                                   ordering=ordering)
        print('YEAR', pred_year, ': MAE =', mae, 'RMSE =', rmse, 'R^2 =', r_squared,
              'Normalized RMSE =', normalized_rmse, 'Normalized MAE =', normalized_mae)
