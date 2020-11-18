import os
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import numpy as np


def _merge_external_data(X):
    # Make sure that DateOfDeparture is of dtype datetime
    X = X.copy()  # modify a copy of X
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])

    filepath = os.path.join(
        os.path.dirname(__file__), 'external_data.csv'
    )
    pdf_ext = pd.read_csv(filepath, parse_dates=["Date"])

    df = pd.merge(
        X, pdf_ext.rename(columns={"Date": "DateOfDeparture", "AirPort": "Departure"}),
        how='left', on=['DateOfDeparture', 'Departure'], sort=False, suffixes=(None, "_departure")
    )
    df = pd.merge(
        df, pdf_ext.rename(columns={"Date": "DateOfDeparture", "AirPort": "Arrival"}),
        how='left', on=['DateOfDeparture', 'Arrival'], sort=False, suffixes=(None, "_arrival")
    )

    def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
        """
        slightly modified version: of http://stackoverflow.com/a/29546836/2901002

        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees or in radians)

        All (lat, lon) coordinates must have numeric dtypes and be of equal length.

        """
        if to_radians:
            lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

        a = np.sin((lat2 - lat1) / 2.0) ** 2 + \
            np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2

        return earth_radius * 2 * np.arcsin(np.sqrt(a))

    df['dist'] = haversine(df.lat, df.long, df.lat_arrival, df.long_arrival)
    df = df.drop(["lat", "long", "lat_arrival", "long_arrival", ], axis=1)

    return df


def _encode_dates(X):
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, 'year'] = X['DateOfDeparture'].dt.year
    X.loc[:, 'month'] = X['DateOfDeparture'].dt.month
    X.loc[:, 'day'] = X['DateOfDeparture'].dt.day
    X.loc[:, 'weekday'] = X['DateOfDeparture'].dt.weekday
    X.loc[:, 'week'] = X['DateOfDeparture'].dt.isocalendar().week
    # X.loc[:, 'n_days'] = X['DateOfDeparture'].apply(
    #     lambda date: (date - pd.to_datetime("1970-01-01")).days
    # )
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["DateOfDeparture"])


def pipeline_elements():
    data_merger = FunctionTransformer(_merge_external_data)

    date_encoder = make_pipeline(
        FunctionTransformer(_encode_dates),
        SimpleImputer(strategy="constant", fill_value=0),
        OneHotEncoder(sparse=False),
    )
    categorical_encoder = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(sparse=False, handle_unknown="ignore"),
    )
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='median'),
        RobustScaler()
    )

    date_cols = ["DateOfDeparture"]
    cat_cols = ['Departure', 'Arrival', 'Events', 'Events_arrival']
    num_col = ['USALORSGPNOSTSAM_arrival',
                 'MeanDew PointC_arrival',
                 'mean_frequentation_arrival',
                 'CloudCover_arrival',
                 'Min DewpointC',
                 'Dew PointC_arrival',
                 'Max TemperatureC_arrival',
                 'CloudCover',
                 'Mean Sea Level PressurehPa_arrival',
                 'familyday_score_arrival',
                 'Max Wind SpeedKm/h_arrival',
                 'population_arrival',
                 'Mean VisibilityKm',
                 'Max Gust SpeedKm/h_arrival',
                 'Mean Sea Level PressurehPa',
                 'we',
                 'Min DewpointC_arrival',
                 'Max VisibilityKm_arrival',
                 'Min VisibilitykM',
                 'mean_frequentation',
                 'Precipitationmm_arrival',
                 'familyday_sum_arrival',
                 'Min Sea Level PressurehPa',
                 'Min TemperatureC_arrival',
                 'doy',
                 'Dew PointC',
                 'school_off_arrival',
                 'Max Humidity',
                 'holiday',
                 'Mean TemperatureC_arrival',
                 'Min Sea Level PressurehPa_arrival',
                 'Max Gust SpeedKm/h',
                 'Mean Humidity',
                 'MeanDew PointC',
                 'Max Humidity_arrival',
                 'Min Humidity_arrival',
                 'Min Humidity',
                 'Mean TemperatureC',
                 'WindDirDegrees_arrival',
                 'std_wtd',
                 'Max TemperatureC',
                 'dist',
                 'Precipitationmm',
                 'WindDirDegrees',
                 'Mean Wind SpeedKm/h_arrival',
                 'USALORSGPNOSTSAM',
                 'Mean VisibilityKm_arrival',
                 'Max Sea Level PressurehPa_arrival',
                 'Min VisibilitykM_arrival',
                 'agg_frequentation_arrival',
                 'agg_frequentation',
                 'Max Sea Level PressurehPa',
                 'Max VisibilityKm',
                 'WeeksToDeparture',
                 'doy_arrival',
                 'population',
                 'familyday_score',
                 'we_arrival',
                 'school_off',
                 'Mean Wind SpeedKm/h',
                 'Max Wind SpeedKm/h',
                 'holiday_arrival',
                 'familyday_sum',
                 'Mean Humidity_arrival',
                 'Min TemperatureC']

    preprocessor = make_column_transformer(
        (date_encoder, date_cols),
        (categorical_encoder, cat_cols),
        (numeric_transformer, num_col),
        # remainder='passthrough',
        remainder='drop',
    )

    return (data_merger, preprocessor)


def get_estimator():
    regressor = xgb.XGBRegressor(objective="reg:squarederror",
                                 base_score=0.5, booster='dart', colsample_bylevel=1,
                                 colsample_bynode=1, colsample_bytree=0.6, eta=0.3, gamma=0.5,
                                 gpu_id=-1, importance_type='gain', interaction_constraints='',
                                 learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                                 min_child_weight=3, monotone_constraints='()',
                                 n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=42,
                                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1.0,
                                 tree_method='exact', validate_parameters=1, verbosity=None)
    # params = {
    #     'min_child_weight': [1, 2, 3, 5, 10],
    #     'gamma': [0, 0.5, 1, 1.5, 2, 5],
    #     'subsample': [0.6, 0.8, 1.0],
    #     'colsample_bytree': [0.6, 0.8, 1.0],
    #     'max_depth': [2, 4, 6, 8, 12],
    #     'booster': ["gbtree", "gblinear", "dart"],
    #     'eta': [0.2, 0.3, 0.4, 0.5]
    # }
    # random_search = (
    #     RandomizedSearchCV(
    #         regressor, param_distributions=params,
    #         n_iter=10,
    #         scoring='neg_mean_squared_error',
    #         n_jobs=-1, cv=3,
    #         verbose=3, random_state=42)
    # )
    pipeline = make_pipeline(*pipeline_elements(), regressor)
    return pipeline

