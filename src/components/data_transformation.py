import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.exception import CustomException
import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]
            )

            logging.info('Scaling of numerical features completed')

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    # ('scaler', StandardScaler())
                ]
            )

            logging.info('Encoding of categorical features completed')

            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipeline, numerical_features),
                    ('categorical_pipeline', categorical_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiateTransformation(self, train_data_path, test_data_path):
        try:
            train_data_df = pd.read_csv(train_data_path)
            test_data_df = pd.read_csv(test_data_path)

            logging.info('Reading train and test data completed')

            logging.info('Getting preprocessor object')

            preprocessor_object = self.get_transformer_object()

            target_column_name = 'math_score'
            numerical_columns = ['reading_score', 'writing_score']

            input_feature_train_df = train_data_df.drop(
                columns=[target_column_name], axis=1)
            target_feature_train_df = train_data_df[target_column_name]

            input_feature_test_df = test_data_df.drop(
                columns=[target_column_name], axis=1)
            target_feature_test_df = test_data_df[target_column_name]

            logging.info(
                'Applying preprocessing object on training and testing dataframes')

            input_feature_train_arr = preprocessor_object.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = preprocessor_object.transform(
                input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,
                              np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]

            logging.info('Preprocessing object saved')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_object
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
