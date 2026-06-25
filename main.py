import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import shap
import joblib
df=pd.read_excel(r"C:\Users\anany\Downloads\archive (3)\Telco_customer_churn_IBM_dataset\Telco_customer_churn.xlsx")
#print(df.head())
#print(df.shape)
# print(df.isnull().sum())
#print(df.describe())
#sns.histplot(df["Monthly Charges"])
#plt.show()
# print("Before graph")
# sns.histplot(df["Monthly Charges"])
# plt.show()


#analysis 1:contract type vs churn
# sns.countplot(x='Contract',hue='Churn Label',data=df)
# plt.xticks(rotation=45)
# plt.show()
#analysis 2: Gender vs churn
# sns.countplot(x='Gender',hue='Churn Label',data=df)
# plt.show()
#analysis 3 : senior citizen vs churn
# sns.countplot(x="Senior Citizen",hue="Churn Label",data=df)
# plt.show()
#analysis 4 : internet services vs churn
# sns.countplot(x='Internet Service',hue="Churn Label",data=df)
# plt.xticks(rotation=45)
# plt.show()
#analyis 5: Monthly Charges vs churn
# sns.boxplot(x="Churn Label",y="Monthly Charges",data=df)
# plt.show()
#analyis 6: Tenure vs churn
# sns.boxplot(x="Churn Label",y="Tenure Months",data=df)
# plt.show()
# df["Churn Reason"]=df["Churn Reason"].fillna("No churn")
# print(df.duplicated().sum())
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
# Fill missing Total Charges only
df["Total Charges"] = df["Total Charges"].fillna(
    df["Total Charges"].median()
)
#df.drop("Churn Reason",axis=1,inplace=True)
df.drop("CustomerID",axis=1,inplace=True)
df.drop("State",axis=1,inplace=True)
df.drop("Country",axis=1,inplace=True)
# print(df.info())
# sns.boxplot(df["Monthly Charges"])
# plt.show()
# df["Gender"].unique()
# for col in df.select_dtypes(include='object').columns:
#     print(col, df[col].nunique())
df.drop("City", axis=1, inplace=True)
df.drop("Lat Long", axis=1, inplace=True)
df.drop("Churn Reason", axis=1, inplace=True)
y=df["Churn Value"]
df.drop([
    "Churn Value",
    "Churn Label",
    "Churn Score"
],axis=1,inplace=True)

print(df.isnull().sum()[df.isnull().sum()>0])
#feature encoding
df=pd.get_dummies(df,drop_first=True)
x=df
# print(df.select_dtypes(include='object').columns)
# print(df.head())
# print(df.dtypes)
# print(df.info())
# for col in df.columns:
#     if "churn" in col.lower():
#         print(col)
print(y.value_counts())


#train-test
#x=df.drop("Churn Label_Yes",axis=1)
#y=df["Churn Label_Yes"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42,stratify=y
)
print(x_train.shape)
print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(y.value_counts(normalize=True))
# print(y_train.value_counts(normalize=True))
# print(y_test.value_counts(normalize=True))
#for col in df.select_dtypes(include='object').columns:
    #print(col, df[col].nunique())
#print(df.select_dtypes(include='object').columns.tolist())


#feature Scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test) 


#train logistic regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(
    max_iter=1000,
    random_state=42
)
model.fit(x_train_scaled,y_train)


#prediction
y_pred=model.predict(x_test_scaled)


#calculate accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


#visualize confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(
    cm,annot=True,fmt='d'
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


#train random forests
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(x_train,y_train)
y_pred_rf=rf.predict(x_test)


#EValuate
from sklearn.metrics import accuracy_score,classification_report
print("Accuracy:",accuracy_score(y_test,y_pred_rf))
print(classification_report(y_test,y_pred_rf))


#XGBOOST
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(x_train, y_train)
y_pred_xgb = xgb_model.predict(x_test)


#Evaluate
print("\nXGBOOST RESULTS")
print(
    "Accuracy:",
    accuracy_score(y_test, y_pred_xgb)
)
print(
    classification_report(
        y_test,
        y_pred_xgb
    )
)


#comparing the model
log_acc = accuracy_score(
    y_test,
    y_pred
)

rf_acc = accuracy_score(
    y_test,
    y_pred_rf
)

xgb_acc = accuracy_score(
    y_test,
    y_pred_xgb
)

#RESULT
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [
        log_acc,
        rf_acc,
        xgb_acc
    ]
})

print(results)


#FEATURE IMPORTANCE
coefficients = pd.DataFrame({
    'Feature': x.columns,
    'Coefficient': model.coef_[0]
})

coefficients['Abs_Coefficient'] = coefficients['Coefficient'].abs()

coefficients = coefficients.sort_values(
    by='Abs_Coefficient',
    ascending=False
)

print(coefficients.head(15))
coefficients = pd.DataFrame({
    'Feature': x.columns,
    'Coefficient': model.coef_[0]
})

coefficients = coefficients.sort_values(
    by='Coefficient',
    ascending=False
)

print(coefficients.head(10))
print("\n")
print(coefficients.tail(10))


#SHAP = SHapley Additive exPlanations
explainer=shap.LinearExplainer(
    model,
    x_train_scaled
)
shap_values=explainer.shap_values(
    x_test_scaled
)
shap.summary_plot(
    shap_values,x_test,
    feature_names=x.columns
)


#explain first customer
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=x_test.iloc[0],
        feature_names=x.columns
    )
)


#high risk customer
import numpy as np

pred_probs = model.predict_proba(
    x_test_scaled
)[:,1]

high_risk_idx = np.argmax(pred_probs)

print(
    "Highest Churn Probability:",
    pred_probs[high_risk_idx]
)
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[high_risk_idx],
        base_values=explainer.expected_value,
        data=x_test.iloc[high_risk_idx],
        feature_names=x.columns
    )
)
print(len(x.columns))
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(x.columns.tolist(), "feature_names.pkl")

print("Everything saved successfully!")


