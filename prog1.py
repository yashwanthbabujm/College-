import pandas as pd
import numpy as np
data = pd.read_csv('data.csv')
print(data)

d= np.array(data)[:,:-1]
print("n the attributes are:",d)

target= np.array(data)[:,-1]
print("n the target is:", target)

def train (c,t):
    for i, val in enumerate(t):
        if val=="yes":
            specific_hypothesis=c[i].copy()
            break
    for i,val in enumerate(c):
        if t[i]=="yes":
                for x in range(len(specific_hypothesis)):
                    if val[x]!= specific_hypothesis[x]:
                        specific_hypothesis[x]="?"
                    else:
                        pass
                    
    return specific_hypothesis
print("/n the final hypothesis is:",train(d,target))            
        