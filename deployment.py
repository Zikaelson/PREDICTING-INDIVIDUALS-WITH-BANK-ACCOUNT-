
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scipy.sparse import hstack
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, r2_score
from tqdm import tqdm_notebook, tqdm # type: ignore
from scipy import stats
import warnings
import seaborn as sns
from matplotlib import pyplot as plt


from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn import model_selection
import category_encoders as ce # type: ignore
import xgboost as xgb # type: ignore
import lightgbm as lgb # type: ignore
from sklearn import metrics
from scipy.optimize import minimize
from lightgbm import LGBMRegressor # type: ignore
from catboost import CatBoostRegressor # type: ignore
warnings.filterwarnings('ignore')
#Run this to check for updates : git fetch

