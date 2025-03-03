import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
heartDisease= pd.read_csv('prog5.csv')
heartDisease=heartDisease.replace('?',np.nan)

print('datset columns:',heartDisease.columns)

print('sample instances from the dataset:')
print(heartDisease.head)
print("\n attributes and datatypes")
print(heartDisease.dtypes)

model=BayesianModel([
     ('age','HeartDisease'),
     ('sex','HeartDisease'),
     ('exang','HeartDisease'),
     ('cp','HeartDisease'),
     ('HeartDisease','restecg'),
     ('HeartDisease','chol')
])
print("\n laerning  cpd using maximum lilelihood estimators")
model.fit(heartDisease,estimator=MaximumLikelihoodEstimator)

print("\n inferering with bayesian network:")
HeartDisease_infer= VariableElimination(model)

print('\n1. probability of haert disease given evidence= restecg')
q1=HeartDisease_infer.query(variables=['HeartDisease'], evidence={'restecg':1})

print("\n2 probability pf heart disease given evidence=cp")
q2=HeartDisease_infer.query(variables=['HeartDisease'], evidence={'cp':2})
print(q2)