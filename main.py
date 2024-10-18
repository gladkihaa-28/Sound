import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

# Load the training data (from CSV or prepare it)
directory = 'train'
csv_file = "train_gt.csv"

if os.path.exists(csv_file):
    data = pd.read_csv(csv_file)
else:
    raise FileNotFoundError(f"File {csv_file} not found")

# Update file paths
data['filename'] = data['filename'].apply(lambda x: os.path.join(directory, x))

# Function to extract features (MFCCs)
def extract_features(file_path):
    try:
        x, sr = librosa.load(file_path, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=128), axis=1)
        return mfccs
    except Exception as e:
        print(f"Error extracting features from file {file_path}: {e}")
        return None

# Extract features from training data
train_features = []
train_labels = []

for i, row in tqdm(data.iterrows(), total=data.shape[0], desc="Extracting features from training data"):
    features = extract_features(row['filename'])
    if features is not None:
        train_features.append(features)
        train_labels.append(row['label'])

# Convert to numpy arrays
X_train = np.array(train_features)
y_train = np.array(train_labels)

# Balance the training data using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Standardize the data
scaler = StandardScaler()
X_train_balanced = scaler.fit_transform(X_train_balanced)

# Train-test split
X_train, X_validation, y_train, y_validation = train_test_split(X_train_balanced, y_train_balanced, test_size=0.2, random_state=42)

### Step 1: Hyperparameter Tuning

# Naive Bayes
nb_model = GaussianNB()

# XGBoost
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3)

# XGBoost Hyperparameter Search Space
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'colsample_bytree': [0.3, 0.7, 1],
    'subsample': [0.5, 0.7, 1]
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Randomized Search for XGBoost
xgb_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_params,
    n_iter=10,
    scoring='f1_macro',
    n_jobs=-1,
    cv=3,  # 3-fold cross-validation
    random_state=42,
    verbose=1
)

# Fit XGBoost Randomized Search
xgb_search.fit(X_train, y_train)

# Best XGBoost model
best_xgb_model = xgb_search.best_estimator_


# Naive Bayes Hyperparameter Tuning
nb_params = {'var_smoothing': np.logspace(0, -9, num=100)}

# Randomized Search for Naive Bayes
nb_search = RandomizedSearchCV(
    estimator=nb_model,
    param_distributions=nb_params,
    n_iter=10,
    scoring='f1_macro',
    n_jobs=-1,
    cv=cv,
    random_state=42,
    verbose=1
)

# Fit Naive Bayes Randomized Search
nb_search.fit(X_train, y_train)

# Best Naive Bayes model
best_nb_model = nb_search.best_estimator_

### Step 2: Ensemble the Models

# Voting Classifier (Soft Voting)
ensemble_model = VotingClassifier(
    estimators=[('naive_bayes', best_nb_model), ('xgboost', best_xgb_model)],
    voting='soft'  # Use probabilities for voting
)

# Train the ensemble
ensemble_model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = ensemble_model.predict(X_validation)
f1_macro = f1_score(y_validation, y_pred, average='macro')

# Print results
print(f"F1 Macro Score: {f1_macro:.4f}")
print("Classification Report:")
print(classification_report(y_validation, y_pred))
