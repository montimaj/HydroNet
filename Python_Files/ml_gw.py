# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import numpy as np
from Python_Files.modeling import gw_driver, ml_driver, pca_reduce
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
        makedirs([self.output_dir])
        self.random_state = random_state
        self.scaled_df = None
        self.scaler = None
        self.x_train, self.x_test, self.y_train, self.y_test = [None] * 4
        self.x_train_pca, self.x_test_pca, self.fit_pca = [None] * 3
        self.train_test_out_dir = None
        self.pca_output_dir = None
        self.model_output_dir = None
        self.built_model = None

    def scale_and_split_df(self, pred_attr='GW', shuffle=False, drop_attrs=(), test_year=(2012,), test_size=0.2,
                           split_yearly=True, load_data=False):
        """
        Scale input dataframe and then generate training and testing data
        :param pred_attr: Target attribute
        :param shuffle: Set True to enabling data shuffling
        :param drop_attrs: Drop these specified attributes
        :param test_year: Build test data from only this year(s)
        :param test_size: Required only if split_yearly=False
        :param split_yearly: Split train and test data based on years
        :param load_data: Set True to load existing scaled train and test data
        :return: None
        """

        print('Scaling and splitting df...')
        self.train_test_out_dir = make_proper_dir_name(self.output_dir + 'Train_Test_Data')
        if not load_data:
            makedirs([self.train_test_out_dir])
        self.scaled_df, self.scaler = ml_driver.scale_df(self.input_df, self.train_test_out_dir, load_data=load_data)
        test_year_scaled = [ty.ravel()[-1] for ty in
                            [self.scaler.transform(np.array([[ty] * self.scaled_df.shape[1]])) for ty in test_year]]
        self.x_train, self.x_test, self.y_train, self.y_test = ml_driver.split_data(self.scaled_df,
                                                                                    self.train_test_out_dir,
                                                                                    pred_attr=pred_attr,
                                                                                    shuffle=shuffle,
                                                                                    drop_attrs=drop_attrs,
                                                                                    test_year=test_year_scaled,
                                                                                    test_size=test_size,
                                                                                    split_yearly=split_yearly,
                                                                                    random_state=self.random_state,
                                                                                    load_data=load_data)

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

    def perform_regression(self, use_pca_data=False, model_type='mlp', load_model=False, **kwargs):
        """
        Perform regression using different models
        :param use_pca_data: Set True to use PCA transformed data
        :param model_type: Regression model, default is MLP. Others include ,'lreg', 'kreg', 'vlstm', 'lstm', 'cnnlstm'
        :param load_model: Set True to load existing model
        :return: None
        """

        self.model_output_dir = make_proper_dir_name(self.output_dir + 'Model_Dumps')
        if not load_model:
            makedirs([self.model_output_dir])
        x_train_data = self.x_train
        x_test_data = self.x_test
        if use_pca_data:
            x_train_data = self.x_train_pca
            x_test_data = self.x_test_pca
        if model_type == 'mlp':
            cv = kwargs['cv']
            grid_iter = kwargs['grid_iter']
            self.built_model = ml_driver.perform_mlpregression(x_train_data, x_test_data, self.y_train, self.y_test,
                                                               self.model_output_dir, cv=cv, grid_iter=grid_iter,
                                                               random_state=self.random_state, load_model=load_model)
        elif model_type == 'lreg':
            self.built_model = ml_driver.perform_linearregression(x_train_data, x_test_data, self.y_train, self.y_test)

        elif model_type == 'kreg':
            validation_split = kwargs['validation_split']
            self.built_model = ml_driver.perform_kerasregression(x_train_data, x_test_data, self.y_train, self.y_test,
                                                                 self.model_output_dir, random_state=self.random_state,
                                                                 validation_split=validation_split)


def run_ml_gw():
    """
    Main caller function
    :return: None
    """

    gw_df = gw_driver.create_ml_data(load_df=True)
    output_dir = r'..\Outputs\All_Data'
    test_years = range(2011, 2019)
    drop_attrs = ('YEAR',)
    hydronet = HydroNet(gw_df, output_dir)
    hydronet.scale_and_split_df(test_year=test_years, drop_attrs=drop_attrs, split_yearly=False, load_data=True)
    hydronet.perform_pca(gamma=1/6, n_components=6, already_transformed=True)
    hydronet.perform_regression(model_type='kreg', validation_split=0.1)


run_ml_gw()
