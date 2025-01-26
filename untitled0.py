

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score

#Load dataset
data_url = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
data = pd.read_csv(data_url)

#Balance the dataset using SMOTE
X = data.drop('Class', axis=1)
y = data['Class']
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Create five samples using random sampling
sample_sizes = [int(len(X_balanced) * p) for p in [0.2, 0.4, 0.6, 0.8, 1.0]]
samples = [X_balanced.sample(n=size, random_state=42) for size in sample_sizes]
labels = [y_balanced.loc[sample.index] for sample in samples]

# Define sampling techniques
def sampling_technique(X, y, technique):
    if technique == 'under_sampling':
        return RandomUnderSampler(random_state=42).fit_resample(X, y)
    elif technique == 'smote':
        return SMOTE(random_state=42).fit_resample(X, y)
    elif technique == 'smoteenn':
        return SMOTEENN(random_state=42).fit_resample(X, y)
    elif technique == 'random_sampling':
        idx = np.random.choice(len(X), size=len(X) // 2, replace=False)
        return X.iloc[idx], y.iloc[idx]
    elif technique == 'original':
        return X, y

techniques = ['under_sampling', 'smote', 'smoteenn', 'random_sampling', 'original']

# Train ML models on samples
models = {
    'M1': RandomForestClassifier(random_state=42),
    'M2': LogisticRegression(max_iter=1000, random_state=42),
    'M3': SVC(random_state=42),
    'M4': KNeighborsClassifier(),
    'M5': DecisionTreeClassifier(random_state=42)
}

results = {}

for model_name, model in models.items():
    for technique in techniques:
        accuracies = []
        for X_sample, y_sample in zip(samples, labels):
            X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)
            X_resampled, y_resampled = sampling_technique(X_train, y_train, technique)
            model.fit(X_resampled, y_resampled)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        results[f"{model_name}_{technique}"] = np.mean(accuracies)

# Determine best sampling technique for each model
best_results = {}

for model_name in models.keys():
    best_technique = max([(technique, results[f"{model_name}_{technique}"]) for technique in techniques], key=lambda x: x[1])
    best_results[model_name] = best_technique

# Display results
print("Best sampling techniques for each model:")
for model, (technique, accuracy) in best_results.items():
    print(f"{model}: {technique} with accuracy {accuracy:.2f}")
