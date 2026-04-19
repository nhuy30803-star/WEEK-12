import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    # 1. Load Dataset
    print("Loading Iris dataset...")
    iris = datasets.load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                     columns=iris['feature_names'] + ['target'])
    
    # Map target numbers to names
    target_names = {i: name for i, name in enumerate(iris.target_names)}
    df['species'] = df['target'].map(target_names)
    
    print("\nDataset Head:")
    print(df.head())

    # 2. Exploratory Data Analysis (EDA)
    print("\nGenerating EDA plots...")
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.pairplot(df.drop('target', axis=1), hue='species', palette='husl')
    plt.savefig('iris_pairplot.png')
    print("Pairplot saved as 'iris_pairplot.png'.")

    # 3. Data Preprocessing
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Model Training and Evaluation
    models = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy
        print(f"\n{name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, predictions, target_names=iris.target_names))

    # 5. Conclusion
    best_model = max(results, key=results.get)
    print(f"\nBest Model: {best_model} with {results[best_model]:.4f} accuracy")

if __name__ == "__main__":
    main()
