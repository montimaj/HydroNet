# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import numpy as np
import pandas as pd
from Python_Files.modeling import gw_driver, ml_driver, pca_reduce
from Python_Files.modeling import model_analysis as ma
from Python_Files.datalibs.sysops import make_proper_dir_name, makedirs


class HydroNet:
    def __init__(self, input_df, output_dir, random_state=0):
        """
        Constructor for initializing class variables
        :param input_df: Input dataframe obtained using gw_driver
        :param output_dir: Output directory
        :param random_state: PRNG seed
        """

        self.input_df = input_df
        self.output_dir = make_proper_dir_name(output_dir)
        self.model_output_dir = make_proper_dir_name(self.output_dir + 'Model_Dumps')
        makedirs([self.output_dir, self.model_output_dir])
        self.random_state = random_state
        self.scaled_df = None
        self.scaler = None
        self.x_train, self.x_test, self.y_train, self.y_test = [None] * 4
        self.x_train_pca, self.x_test_pca, self.fit_pca = [None] * 3
        self.train_test_out_dir = None
        self.pca_output_dir = None
        self.built_model = None
        self.drop_attrs = None
        self.pred_attr = None
        self.x_scaler = None
        self.y_scaler = None
        self.scaling = True
        self.timesteps = 1

    def scale_and_split_df(self, pred_attr='GW', shuffle=False, drop_attrs=(), test_year=(2012,), test_size=0.2,
                           split_yearly=True, scaling=True, load_data=False, pred_attr_threshold=None):
        """
        Scale input dataframe and then generate training and testing data
        :param pred_attr: Target attribute
        :param shuffle: Set True to enabling data shuffling
        :param drop_attrs: Drop these specified attributes
        :param test_year: Build test data from only this year(s)
        :param test_size: Required only if split_yearly=False
        :param split_yearly: Split train and test data based on years
        :param scaling: Set False to use original data
        :param load_data: Set True to load existing scaled train and test data
        :param pred_attr_threshold: Threshold value to discard outliers, default 1500 mm for GW.
        Set None to disable discarding outliers
        :return: None
        """

        print('Scaling and splitting df...')
        self.train_test_out_dir = make_proper_dir_name(self.output_dir + 'Train_Test_Data')
        self.drop_attrs = drop_attrs
        self.pred_attr = pred_attr
        if not load_data:
            makedirs([self.train_test_out_dir])
        self.x_train, self.y_train, self.x_test, self.y_test = ml_driver.split_data(
            self.input_df, self.train_test_out_dir, pred_attr=pred_attr,
            shuffle=shuffle, drop_attrs=drop_attrs, test_year=test_year,
            test_size=test_size, split_yearly=split_yearly,
            random_state=self.random_state, load_data=load_data,
            pred_attr_threshold=pred_attr_threshold
        )
        self.scaling = scaling
        if scaling:
            self.x_scaler, self.y_scaler, self.x_train, self.y_train, self.x_test, self.y_test = ml_driver.scale_df(
                self.x_train, self.y_train, self.x_test,
                self.y_test, self.train_test_out_dir, load_data=load_data
            )

    def perform_pca(self, kpca_type='poly', gamma=10., degree=3, n_components=5, n_samples=1000,
                    already_transformed=False):
        """
        Perform PCA-based dimensionality reduction, initially, eigenvalue plot is displayed to identify the
        appropriate number of components
        :param kpca_type: Set PCA kernel, default is polynomial. Others include 'linear', 'rbf', 'sigmoid', and 'cosine'
        :param gamma: Gamma parameter for polynomial and RBF PCA
        :param degree: Degree parameter for polynomial PCA
        :param n_components: Number of PCA components
        :param n_samples: Number of samples to randomly choose from the input data for fitting. Full data cannot be used
        because of computational cost
        :param already_transformed: Set True to load already transformed data
        :return: None
        """

        print('Performing PCA...')
        self.pca_output_dir = make_proper_dir_name(self.output_dir + 'PCA_Data')
        if not already_transformed:
            makedirs([self.pca_output_dir])
            fit_pca = pca_reduce.fit_pca(self.x_train, self.pca_output_dir, kpca_type, gamma, degree, n_components=None,
                                         n_samples=n_samples, random_state=self.random_state)
            pca_reduce.plot_kpca(fit_pca)
            self.fit_pca = pca_reduce.fit_pca(self.x_train, self.pca_output_dir, kpca_type, gamma, degree,
                                              n_components=n_components, n_samples=n_samples,
                                              random_state=self.random_state)
            _, self.x_train_pca = pca_reduce.pca_transform(fit_pca_obj=self.fit_pca, input_data=self.x_train,
                                                           output_dir=self.pca_output_dir, output_suffix='X_Train_PCA',
                                                           load_data=False)
            _, self.x_test_pca = pca_reduce.pca_transform(fit_pca_obj=self.fit_pca, input_data=self.x_test,
                                                           output_dir=self.pca_output_dir, output_suffix='X_Test_PCA',
                                                           load_data=False)
        else:
            self.fit_pca, self.x_train_pca = pca_reduce.pca_transform(output_dir=self.pca_output_dir,
                                                                      output_suffix='X_Train_PCA', load_data=True)
            _, self.x_test_pca = pca_reduce.pca_transform(output_dir=self.pca_output_dir, output_suffix='X_Test_PCA',
                                                          load_data=True)
            print('Transformed PCA data loaded...')

    def perform_regression(self, use_pca_data=False, model_type='kreg', load_model=False, **kwargs):
        """
        Perform regression using different models
        :param use_pca_data: Set True to use PCA transformed data
        :param model_type: Regression model, default is Keras Regression ('kreg').
        Others include 'mlp,'lreg', 'vlstm', 'lstm', 'cnnlstm'
        :param load_model: Set True to load existing model
        :return: None
        """

        x_train_data = self.x_train
        x_test_data = self.x_test
        if use_pca_data:
            x_train_data = self.x_train_pca
            x_test_data = self.x_test_pca
        if model_type == 'mlp':
            cv = kwargs['cv']
            grid_iter = kwargs['grid_iter']
            self.built_model = ml_driver.perform_mlpregression(x_train_data, self.y_train, self.model_output_dir, cv=cv,
                                                               grid_iter=grid_iter, random_state=self.random_state,
                                                               load_model=load_model)
        elif model_type == 'lreg':
            self.built_model = ml_driver.perform_linearregression(x_train_data, x_test_data, self.y_train, self.y_test)

        elif model_type == 'kreg':
            validation_split = kwargs.get('validation_split', 0.1)
            max_trials = kwargs.get('max_trials', 10)
            max_exec_trial = kwargs.get('max_exec_trials', 3)
            batch_size = kwargs.get('batch_size', None)
            epochs = kwargs.get('epochs', None)
            use_keras_tuner = kwargs.get('use_keras_tuner', True)
            model_number = kwargs.get('model_number', 2)
            load_weights = kwargs.get('load_weights', None)
            self.built_model = ml_driver.perform_kerasregression(x_train_data, x_test_data, self.y_train, self.y_test,
                                                                 self.model_output_dir, random_state=self.random_state,
                                                                 validation_split=validation_split,
                                                                 max_trials=max_trials, max_exec_trial=max_exec_trial,
                                                                 batch_size=batch_size, epochs=epochs,
                                                                 use_keras_tuner=use_keras_tuner, load_model=load_model,
                                                                 model_number=model_number, load_weights=load_weights)
        elif model_type == 'lstm':
            fold_count = kwargs.get('max_trials', 10)
            n_repeats = kwargs.get('max_exec_trials', 3)
            batch_size = kwargs.get('batch_size', None)
            epochs = kwargs.get('epochs', None)
            self.timesteps = kwargs.get('timesteps', 1)
            bidirectional = kwargs.get('bidirectional', None)
            self.built_model = ml_driver.perform_lstm_regression(x_train_data, x_test_data, self.y_train, self.y_test,
                                                                 self.model_output_dir, random_state=self.random_state,
                                                                 fold_count=fold_count, n_repeats=n_repeats,
                                                                 batch_size=batch_size, epochs=epochs,
                                                                 timesteps=self.timesteps,
                                                                 bidirectional=bidirectional)

        else:
            self.built_model = ml_driver.perform_ml_regression(x_train_data, x_test_data, self.y_train, self.y_test,
                                                               output_dir=self.model_output_dir, ml_model=model_type,
                                                               random_state=self.random_state)

    def get_error_stats(self, use_pca_data=False, model_type=None):
        """
        Get error statistics
        :param use_pca_data: Set True to use PCA transformed data
        :param model_type: Set to 'lstm', 'cnn_lstm', or 'conv_lstm' to reshape training and test data accordingly
        for model prediction
        :return: None
        """

        x_train_data = self.x_train
        x_test_data = self.x_test
        if use_pca_data:
            x_train_data = self.x_train_pca
            x_test_data = self.x_test_pca
        if not model_type:
            pred_train = self.built_model.predict(x_train_data)
            pred_test = self.built_model.predict(x_test_data)
        else:
            x_train_arr = x_train_data.to_numpy()
            x_test_arr = x_test_data.to_numpy()
            if model_type == 'lstm':
                x_train = x_train_arr.reshape(x_train_arr.shape[0], self.timesteps, x_train_arr.shape[1])
                x_test = x_test_arr.reshape(x_test_arr.shape[0], self.timesteps, x_test_arr.shape[1])
            elif model_type == 'cnn_lstm':
                x_train = x_train_arr.reshape(x_train_arr.shape[0], 1, self.timesteps, x_train_arr.shape[1])
                x_test = x_test_arr.reshape(x_test_arr.shape[0], 1, self.timesteps, x_test_arr.shape[1])
            else:
                x_train = x_train_arr.reshape(x_train_arr.shape[0], 1, 1, self.timesteps, x_train_arr.shape[1])
                x_test = x_test_arr.reshape(x_test_arr.shape[0], 1, 1, self.timesteps, x_test_arr.shape[1])
            pred_train = self.built_model.predict(x_train)
            pred_test = self.built_model.predict(x_test)
        if self.scaling:
            self.y_train = self.y_scaler.inverse_transform(self.y_train.reshape(-1, 1)).ravel()
            pred_train = self.y_scaler.inverse_transform(pred_train.reshape(-1, 1)).ravel()
            self.y_test = self.y_scaler.inverse_transform(self.y_test.reshape(-1, 1)).ravel()
            pred_test = self.y_scaler.inverse_transform(pred_test.reshape(-1, 1)).ravel()
        ma.generate_scatter_plot(self.y_test, pred_test)
        ma.generate_scatter_plot(self.y_train, pred_train)
        test_stats = ma.get_error_stats(self.y_test, pred_test)
        train_stats = ma.get_error_stats(self.y_train, pred_train)
        print('Train, Test Stats:', train_stats, test_stats)

    def get_prediction_results(self, gw_df, actual_gw, gw_ks, gw_az, grace_ks, grace_az, exclude_vars_ks,
                               exclude_vars_az, test_years, pred_years, forecast_years=(2019,)):
        """
        Generate prediction results for Overall data, Kansas data, and Arizona data
        :param gw_df: Input dataframe
        :param actual_gw: Actual GW values
        :param gw_ks: HydroML Kansas object
        :param gw_az: HydroML Arizona object
        :param grace_ks: GRACE CSV path for Kansas
        :param grace_az: GRACE CSV path for Arizona
        :param exclude_vars_ks: Exclude these variables from plot for Kansas
        :param exclude_vars_az: Exclude these variables from plot for Arizona
        :param test_years: Test years
        :param pred_years: Prediction years
        :param forecast_years: Forecast years
        :return: None
        """

        actual_gw_dir, pred_gw_dir = gw_ks.get_predictions(fitted_model=self.built_model, pred_years=pred_years,
                                                           drop_attrs=self.drop_attrs, exclude_vars=exclude_vars_ks,
                                                           pred_attr=self.pred_attr, only_pred=False,
                                                           x_scaler=self.x_scaler, y_scaler=self.y_scaler)
        ma.run_analysis(actual_gw_dir, pred_gw_dir, grace_ks, use_gmds=False, input_gmd_file=None,
                        out_dir=self.output_dir, forecast_years=forecast_years, ty_start=test_years[0],
                        ty_end=test_years[-1])
        actual_gw_dir, pred_gw_dir = gw_az.get_predictions(fitted_model=self.built_model, pred_years=pred_years,
                                                           drop_attrs=self.drop_attrs, pred_attr=self.pred_attr,
                                                           exclude_vars=exclude_vars_az, exclude_years=(),
                                                           only_pred=False, use_full_extent=False,
                                                           x_scaler=self.x_scaler, y_scaler=self.y_scaler)
        ma.run_analysis(actual_gw_dir, pred_gw_dir, grace_az, use_gmds=False, input_gmd_file=None,
                        out_dir=self.output_dir, forecast_years=forecast_years,
                        ty_start=test_years[0], ty_end=test_years[-1])

        gw_year = gw_df['YEAR'].to_numpy().ravel()
        drop_attrs = [drop_attr for drop_attr in self.drop_attrs]
        gw_df = gw_df.drop(columns=drop_attrs)
        pred_gw = self.built_model.predict(self.x_scaler.transform(gw_df))
        pred_gw = self.y_scaler.inverse_transform(pred_gw.reshape(-1, 1)).ravel()
        subset_gw_df = pd.DataFrame(data={'YEAR': gw_year, 'Pred_GW': pred_gw})
        print(subset_gw_df)
        year_list = sorted(set(subset_gw_df['YEAR']))
        mean_actual_gw = []
        mean_pred_gw = []
        for year in year_list:
            sub_df = subset_gw_df[subset_gw_df.YEAR == year]
            if year not in forecast_years:
                actual_df = actual_gw[actual_gw.YEAR == year]
                mean_actual_gw.append(np.mean(actual_df[self.pred_attr]))
            else:
                mean_actual_gw.append(np.nan)
            mean_pred_gw.append(np.mean(sub_df.Pred_GW))
        mean_gw_df = pd.DataFrame(data={'YEAR': year_list, 'Actual_GW': mean_actual_gw, 'Pred_GW': mean_pred_gw})
        mean_gw_df.to_csv(self.output_dir + 'Overall_Mean_GW_Pred.csv', index=False)
        ma.create_time_series_forecast_plot(mean_gw_df, ty_start=test_years[0], ty_end=test_years[-1], plot_grace=False)


def run_ml_gw():
    """
    Main caller function
    :return: None
    """

    gw_df, gw_ks, gw_az = gw_driver.create_ml_data(load_df=False)
    output_dir = r'..\Outputs\All_Data'
    test_years = range(2016, 2019)
    drop_attrs = ('YEAR',)
    input_df = gw_df.dropna(axis=0)
    pred_attr = 'GW'
    hydronet = HydroNet(input_df, output_dir, random_state=42)
    hydronet.scale_and_split_df(scaling=True, test_year=test_years, drop_attrs=drop_attrs, split_yearly=True,
                                load_data=True, pred_attr=pred_attr)
    hydronet.perform_pca(gamma=1/6, degree=2, n_components=5, already_transformed=True)
    hydronet.perform_regression(use_pca_data=False, model_type='kreg', use_keras_tuner=False, validation_split=0.1,
                                max_trials=10, max_exec_trials=1, batch_size=512, epochs=100, load_model=True,
                                model_number=2, timesteps=1, bidirectional=None, random_state=123,
                                load_weights=None)
    hydronet.get_error_stats()
    exclude_vars_ks = ('ET',)
    exclude_vars_az = ('ET', 'WS_PA', 'WS_PA_EA', 'WS_PT', 'WS_PT_ET')
    grace_ks = '../Inputs/Data/Kansas_GW/GRACE/TWS_GRACE.csv'
    grace_az = '../Inputs/Data/Arizona_GW/GRACE/TWS_GRACE.csv'
    pred_years = range(2002, 2020)
    forecast_years = (2019,)
    actual_gw = gw_df[[pred_attr, 'YEAR']].dropna(axis=0)
    gw_df = gw_df.drop(columns=[pred_attr])
    gw_df = gw_df.dropna(axis=0)
    hydronet.get_prediction_results(gw_df, actual_gw, gw_ks, gw_az, grace_ks, grace_az, exclude_vars_ks,
                                    exclude_vars_az, test_years, pred_years, forecast_years)


run_ml_gw()
