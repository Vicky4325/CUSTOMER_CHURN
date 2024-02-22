import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from google.colab import drive

drive.mount('/content/drive')

data = pd.read_csv('/content/drive/MyDrive/Code-ways/Churn_Modelling.csv')

data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Use LabelEncoder to transform categorical variables
label_encoder = LabelEncoder()
data['Geography'] = label_encoder.fit_transform(data['Geography'])
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Use StandardScaler to transform numerical variables
scaler = StandardScaler()
features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Split the data into training and testing sets
X = data.drop('Exited', axis=1)
y = data['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the StandardScaler on the training data and transform both the training and testing data
X_train_scaled = scaler.fit_transform(X_train[features])
X_test_scaled = scaler.transform(X_test[features])

# Combine the scaled features with the non-scaled categorical variables
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features)

X_train_combined = pd.concat([X_train.drop(features, axis=1), X_train_scaled_df], axis=1)
X_test_combined = pd.concat([X_test.drop(features, axis=1), X_test_scaled_df], axis=1)

# Train the XGBoost model
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train_combined, y_train)

# Make predictions and calculate the accuracy
y_pred = xgb_model.predict(X_test_combined)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

feature_importances = pd.DataFrame({'feature': X_train_combined.columns, 'importance': xgb_model.feature_importances_})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

print("Feature Importances:")
print(feature_importances)
