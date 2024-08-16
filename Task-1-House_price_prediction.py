import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_excel("/HousePricePrediction.xlsx")

# Print first 5 records of the dataset
print(dataset.head(5))

# Identify categorical and numerical columns
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)

# Plot correlation heatmap for numerical columns
plt.figure(figsize=(12, 6))
numeric_dataset = dataset.select_dtypes(include=['number'])
sns.heatmap(numeric_dataset.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.show()

# Plot unique values of categorical features
unique_values = [dataset[col].unique().size for col in object_cols]
plt.figure(figsize=(10, 6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols, y=unique_values)
plt.show()

# Plot distribution of categorical features
plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
for index, col in enumerate(object_cols, start=1):
	y = dataset[col].value_counts()
	ax = plt.subplot(11, 4, index)
	plt.xticks(rotation=90)
	sns.barplot(x=list(y.index), y=y)
	if index > 1:
		ax.remove()
plt.show()

# Data preprocessing
dataset.drop(['Id'], axis=1, inplace=True)
dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())
new_dataset = dataset.dropna()

# One-hot encode categorical variables
object_cols = [col for col in new_dataset.columns if new_dataset[col].dtype == 'object']
print("Categorical variables:", object_cols)
print('No. of categorical features:', len(object_cols))

OH_encoder = OneHotEncoder(sparse_output=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out(input_features=object_cols)

df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

# Split the data into features and target variable
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# Split the data into training and validation sets
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

# Train and evaluate models
def evaluate_model(model, X_train, Y_train, X_valid, Y_valid):
	model.fit(X_train, Y_train)
	Y_pred = model.predict(X_valid)
	mape = mean_absolute_percentage_error(Y_valid, Y_pred)
	print(f"{model.__class__.__name__} MAPE:", mape)
	return Y_pred

# Support Vector Regressor
model_SVR = svm.SVR()
evaluate_model(model_SVR, X_train, Y_train, X_valid, Y_valid)

# Random Forest Regressor
model_RFR = RandomForestRegressor(n_estimators=10)
evaluate_model(model_RFR, X_train, Y_train, X_valid, Y_valid)

# Linear Regression
model_LR = LinearRegression()
evaluate_model(model_LR, X_train, Y_train, X_valid, Y_valid)

# CatBoost Regressor
cb_model = CatBoostRegressor(verbose=0)
cb_model.fit(X_train, Y_train)
preds = cb_model.predict(X_valid)
cb_r2_score = r2_score(Y_valid, preds)
print("CatBoost Regressor R2 Score:", cb_r2_score)
