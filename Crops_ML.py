# All required libraries are imported here for you.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import f1_score
!pip install mlxtend

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

# Write your code here

# dataset checking
# print(crops.tail())

# Get an Overview over the Dataframe

# Checking dataset information
# crops.info()

# checking for duplicates
mask = crops.duplicated()
print(crops[mask])

# show only the Rows with no input...
empty_rows = crops.isnull().any(axis=1)
print(crops[empty_rows])

# Each column has to be numeric
crops[pd.isna(crops.iloc[:, :4])]                     # Our Solution
# crops.isna().sum()                                      # Real solution

# crops.describe()
# print(crops[crops['N'] == 0])

# Check Crop types
crop_types = crops['crop'].unique()
print(crop_types)
len(crop_types)

# split the Data into Training and Test set
X =crops.iloc[:, 0:4]
y =crops.loc[: ,'crop']
#print(X)
#print(y)
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state =42)

#X= crops.iloc[:, 0:1]
#for i in X:
 #   X=crops.iloc[:, 0:i]
#print((X_train))
#e= np.array([X_train.iloc[:, i]])
#print(type(e))

#Predict the crop using each feature
#4 models and a for loop

#print(Xnew_train)
num_columns=4
train_list=[]
for i in range(4):
    a = [X_train.iloc[:, i]]
    train_list.append(a)
array_of_Xtrain_arrays = [np.array(sublist) for sublist in train_list]
arrays_Xtrain = np.array(array_of_Xtrain_arrays)
print(arrays_Xtrain.shape)
#print(len(train_list))

num_columns=4
test_list=[]
for i in range(4):
    a = [X_test.iloc[:, i]]
    test_list.append(a)
array_of_Xtest_arrays = [np.array(sublist) for sublist in test_list]
arrays_Xtest = np.array(array_of_Xtest_arrays)
print(arrays_Xtest.shape)
#print(len(train_list))

num_columns=4
test_list_list=[]
for i in range(4):
    a = [X_test.iloc[:, i]]
    test_list_list.append(a)
array_of_Xtest_arrays = [np.array(sublist) for sublist in test_list]
#arrays_Xtest = np.concatenate(array_of_Xtest_arrays,)
result = np.vstack(array_of_Xtest_arrays)

print("Ergebnis: ", result, result.shape)
#print(arrays_Xtest)
#print(len(train_list))


model=LogisticRegression(max_iter=2000)
#y_train=y_train.reshape(-1,1)
#print(arrays_Xtrain[0].shape , y_train.shape)
for i in range(4):
    model.fit(X_train[i],y_train)
    y_pred=model.predict(X_test[i])
    score= f1_score(y_test, y_pred, average="macro")
    print(score)

print(X_train.iloc[:, [0,2]])
print(X_train.shape[1])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

model = LogisticRegression(max_iter=2000)


def mult_feat_f1(modeling, X_train, y_train, X_test, y_test, average="macro", min_features=1, max_features=1,
                 print_progress=True, show_corr=True):
    if min_features == 1 and max_features == 1:
        arrays_Xtrain = iter_df(X_train)
        arrays_Xtest = iter_df(X_test)
        score = []

        for i in range(4):
            model.fit(arrays_Xtrain[i].T, y_train)
            y_pred = model.predict(arrays_Xtest[i].T)
            score.append(f1_score(y_test, y_pred, average="macro"))
        best_index = X_train.columns[score.index(max(score))]
        print(f" your maximum score is: {max(score)}" + f"for the Feature {best_index}")

    else:
        modeling = model
        efs = EFS(model, min_features, max_features, print_progress=True)

        efs = efs.fit(X_train, y_train)
        # print(efs.best_idx_)
        best_feature_indices = list(efs.best_idx_)
        print(X_train.columns[best_feature_indices])

        # Train a model using the best features
        best_features_X_train = X_train.iloc[:, best_feature_indices]
        best_features_X_test = X_test.iloc[:, best_feature_indices]

        model.fit(best_features_X_train, y_train)
        y_pred = model.predict(best_features_X_test)

        f1_sc = f1_score(y_test, y_pred, average='macro')

        # print(f'Best Features: {efs.best_idx_}')
        print(f'Best F1 Score: {f1_sc}')

    if show_corr == True:
        correlation_matrix = X_train.corr()
        fig, ax = plt.subplots()
        cax = ax.matshow(correlation_matrix, vmin=-1, vmax=1)

        fig.colorbar(cax)
        plt.show()

# mult_feat_f1(model, X_train, y_train, X_test, y_test, min_features=2, max_features=2)
mult_feat_f1(model, X_train, y_train, X_test, y_test, min_features=2, max_features=2)

correlation_matrix = X_train.corr()
fig, ax =plt.subplots()
cax= ax.matshow(correlation_matrix, vmin=-1, vmax=1)

fig.colorbar(cax)
plt.show()

for i in range(4):
    accuracy = model.score(arrays_Xtest[i].T, y_test)
    print(accuracy)
    print(model.decision_function(arrays_Xtest[i].T))

#Predict the crop using each feature
#4 models and a for loop
model=LogisticRegression()
model.fit(X_train,y_train)
predictions= model.predict(X_test)
predictions