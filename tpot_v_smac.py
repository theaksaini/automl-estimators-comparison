# add a new directory to the path
import sys
sys.path.append('../')

import nest_asyncio
import sklearn.feature_selection
import sklearn.model_selection  # Only necessary in Notebooks
import time
import sklearn.metrics
from collections.abc import Iterable
import pandas as pd
import sklearn
import numpy as np
from typing import Any

import openml
from sklearn.preprocessing import LabelEncoder

from amltk.sklearn import split_data

import numpy as np
import openml
from sklearn.compose import make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC

from amltk.data.conversions import probabilities_to_classes
from amltk.ensembling.weighted_ensemble_caruana import weighted_ensemble_caruana
from amltk.optimization import History, Metric, Trial
from amltk.optimization.optimizers.smac import SMACOptimizer
from amltk.pipeline import Choice, Component, Sequential, Split
from amltk.scheduling import Scheduler
from amltk.sklearn.data import split_data
from amltk.store import PathBucket
from amltk.optimization import Metric
from amltk.scheduling import Scheduler
from amltk.optimization.optimizers.smac import SMACOptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from amltk.pipeline import Component, Node, Sequential, Split
from functools import partial
import pandas as pd

import automl_estimators
import tpot2

pipeline1 = (
    Sequential(name="Pipeline")
    >> Choice(  # <!> (1)!
        Component(SVC, space={"C": (0.1, 10.0)}, config={"probability": True}),
        Component(
            RandomForestClassifier,
            space={"n_estimators": (10, 100), "criterion": ["gini", "log_loss"]},
        ),
    )
)

pipeline2 = (
    Sequential(name="Pipeline")
    >> Choice(  # <!> (1)!
        Component(SVC, space=tpot2.config.classifiers.get_SVC_ConfigurationSpace(random_state=42)),
        Component(RandomForestClassifier, space=tpot2.config.classifiers.get_RandomForestClassifier_ConfigurationSpace(random_state=42))
    )
)



nest_asyncio.apply()
scorers = ['accuracy']
cv = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X = np.random.rand(100,10)
y = np.random.randint(0,2,100)

X = pd.DataFrame(X)
y = pd.DataFrame(y)

est = automl_estimators.SMACEstimator(pipeline=pipeline1, scorers=scorers, other_objective_functions=[], 
                                      cv=cv, max_time_seconds=5, n_jobs=1, seed=None, max_evals=2)

est.fit(X,y)

est.fitted_pipeline_