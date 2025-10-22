import os
import  pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser("~/Desktop/kaggle")
url="/Users/amirali/PycharmProjects/PythonProject/house_prediction/dataset_for_house_prediction/train.csv"
df=pd.read_csv(url)




mean=df[["LotFrontage"]].mean()
df[["LotFrontage"]]=df[["LotFrontage"]].replace(np.nan,mean)
df=df.drop(axis=1,labels=["PoolQC","Fence","MiscFeature","FireplaceQu","Alley","MasVnrType"])
df[["BsmtQual","BsmtExposure","BsmtFinType1","BsmtFinType2","GarageType","GarageYrBlt","GarageFinish","GarageQual","GarageCond"]]=df[["BsmtQual","BsmtExposure","BsmtFinType1","BsmtFinType2","GarageType","GarageYrBlt","GarageFinish","GarageQual","GarageCond"]].replace(np.nan,"unknown",)
df[["Electrical","BsmtQual","BsmtCond"]]=df[["Electrical","BsmtQual","BsmtCond"]].replace(np.nan,"unknown")


gp=df[["OverallQual","GrLivArea","GrLivArea","GarageArea","YearBuilt","YearBuilt","SalePrice"]]
gp_for_scater_polt=df[["OverallQual","GrLivArea","GrLivArea","GarageArea","YearBuilt","YearBuilt"]]
corr=gp.corr()
#OverallQual has the highest correlation with SalePrice columns

sns.scatterplot(data=df,x="OverallQual",y="SalePrice")
plt.xlabel("quality of the house(OverallQual)")
plt.ylabel("sale price")
#scatter plot with price and OverrallQual
#plt.show()
#plt.close()

sns.boxplot(data=df,x="Neighborhood",y="SalePrice")
plt.xlabel("Neighborhood")
plt.ylabel("price")
#plt.show()
#plt.legend()

df_group=df.groupby(["GrLivArea","OverallQual","GarageArea"],as_index=False)["SalePrice"].mean()
clean_df=df[["OverallQual","GrLivArea","GarageArea","SalePrice","Neighborhood"]]

#i have moved these columns to other data frame because they have the highest correlation with Sale Price

one_hotNeighborhood=pd.get_dummies(df[["Neighborhood"]],dtype=int,drop_first=True)
df_final=pd.concat([clean_df.drop(axis=1,labels="Neighborhood"),one_hotNeighborhood],axis=1)
print(df.head())

#linear reggression
x_train,x_test,y_train,y_test=train_test_split(df_final[["OverallQual","GrLivArea","GarageArea"]],df_final["SalePrice"],train_size=0.8,random_state=1)
lr=LinearRegression()
lr.fit(x_train,y_train)
ythat=lr.predict(x_test)
mse=mean_squared_error(y_test,ythat)
print(f"mean square error : {mse}")
r2_test=lr.score(x_test,y_test)
r2_train=lr.score(x_train,y_train)
print(f"r2 train  {r2_train} and r2 test {r2_test}")

#pipleline part
Input=[("Scale",StandardScaler()),("Polynomial",PolynomialFeatures(include_bias=False)),("model",LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(x_train,y_train)
pipPredict=pipe.predict(x_test)

#Ridge
Ridgemodel=Ridge(alpha=1)
Ridgemodel.fit(x_train,y_train)
RidgePredict=Ridgemodel.predict(x_test)
print(RidgePredict[0:11])

#GridsearchCV
parameter=[{"alpha":[1,2,3,4,5,6,7,8,9,10,11]}]
Rid=Ridge()
Grid=GridSearchCV(Rid,parameter,cv=4)
Grid.fit(x_train,y_train)
bestGrid=Grid.best_estimator_
print(f"best estimator is {bestGrid}")

#cross_validation
cross=cross_val_score(bestGrid,x_train,y_train,cv=4)
print(f"the mean of cross validation is {cross.mean()}")

bestModel=Ridge(alpha=1)
bestModel.fit(df_final.drop("SalePrice",axis=1),df_final["SalePrice"])

urlTest="/Users/amirali/PycharmProjects/PythonProject/house_prediction/dataset_for_house_prediction/test.csv"
test_cv=pd.read_csv(urlTest)

# Preprocess test data like train data
mean = test_cv[["LotFrontage"]].mean()
test_cv[["LotFrontage"]] = test_cv[["LotFrontage"]].replace(np.nan, mean)

test_cv = test_cv.drop(axis=1, labels=["PoolQC", "Fence", "MiscFeature", "FireplaceQu", "Alley", "MasVnrType"])
test_cv[["BsmtQual","BsmtExposure","BsmtFinType1","BsmtFinType2","GarageType","GarageYrBlt","GarageFinish","GarageQual","GarageCond"]] = (
    test_cv[["BsmtQual","BsmtExposure","BsmtFinType1","BsmtFinType2","GarageType","GarageYrBlt","GarageFinish","GarageQual","GarageCond"]]
    .replace(np.nan,"unknown")
)
test_cv[["Electrical","BsmtQual","BsmtCond"]] = test_cv[["Electrical","BsmtQual","BsmtCond"]].replace(np.nan,"unknown")

# Select the same columns
clean_test = test_cv[["OverallQual","GrLivArea","GarageArea","Neighborhood"]]
one_hotNeighborhood_test=pd.get_dummies(clean_test[["Neighborhood"]],dtype=int,drop_first=True)
Test_final=pd.concat([clean_test.drop("Neighborhood",axis=1),one_hotNeighborhood_test],axis=1)
Test_final=Test_final.reindex(columns=df_final.drop("SalePrice",axis=1).columns,fill_value=0)
print(Test_final.isna().sum()[Test_final.isna().sum() > 0])
Test_final = Test_final.fillna(0)
Predict=bestModel.predict(Test_final)
print(Predict[0:10])
# Create a submission DataFrame
submission = pd.DataFrame({
    "Id": test_cv["Id"],
    "SalePrice": Predict
})

# Save to CSV for Kaggle submission
submission.to_csv("/Users/amirali/PycharmProjects/PythonProject/house_prediction/submission.csv", index=False)
print("Submission file created successfully!")


