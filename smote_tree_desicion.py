import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import tree

def load_the_data():
    # Load the datasets
    data = pd.read_csv('assets/MEELMMPI.csv')
    judges = pd.read_csv('assets/MEELJUD.csv')
    
    judges=judges.drop(judges.columns[-1],axis=1)
    data=data.drop(data.columns[-1],axis=1)
    
    return data, judges

def predictions_with_smote_decision_tree(X, df_jud):
    all_predictions = []
    
    for judge_index in range(df_jud.shape[1]):
        y = df_jud.iloc[:, judge_index]
        X_res, y_res, df_resampled = smote_augmentation(X, y)
        print(f"Number of samples after SMOTE for Judge {judge_index + 1}: {len(y_res)}")
        y_test, y_pred = decision_tree_on_judge(X_res, y_res)
        all_predictions.append(y_pred)
    
    return all_predictions

def predictions_with_smote_linear_regression(X, df_jud):
    all_predictions = []
    
    for judge_index in range(df_jud.shape[1]):
        y = df_jud.iloc[:, judge_index]
        X_res, y_res, df_resampled = smote_augmentation(X, y)
        print(f"Number of samples after SMOTE for Judge {judge_index + 1}: {len(y_res)}")
        y_test, y_pred = linear_regression_on_judge(X_res, y_res)
        all_predictions.append(y_pred)
    
    return all_predictions

def smote_augmentation(df_characteristics, judge):
    # Features (X) from MEELMMPI dataset
    X = df_characteristics

    # Target (y) column from the specified judge in MEELJUD dataset
    y = judge

    # Apply SMOTE
    smote = SMOTE(sampling_strategy={cls: 500 for cls in y.unique()}, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Convert the resampled dataset into a DataFrame
    df_resampled = pd.DataFrame(X_res, columns=X.columns)
    df_resampled['target'] = y_res
    
    return X_res, y_res, df_resampled

# desicion tree on judge
def decision_tree_on_judge(X, judge):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    
    decision_tree_classifier = DecisionTreeClassifier(max_depth=3)
    X_train, X_test, y_train, y_test = train_test_split(X, judge, test_size=0.2, random_state=42)

    decision_tree_classifier.fit(X_train, y_train)
    print(tree.plot_tree(decision_tree_classifier))
    y_pred = decision_tree_classifier.predict(X_test)
    
    return y_test, y_pred

# linear regression on judge
def linear_regression_on_judge(X, judge):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    
    linear_regression = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, judge, test_size=0.2, random_state=42)

    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    
    return y_test, y_pred

def mse_predictions(all_predictions, model_name='Decision Tree'):
    # Initialize MSE matrix
    num_judges = len(all_predictions)
    mse_matrix = np.zeros((num_judges, num_judges))

    # Calculate pairwise MSE values
    for i in range(num_judges):
        for j in range(num_judges):
            mse_matrix[i, j] = mean_squared_error(all_predictions[i], all_predictions[j])
            
    # Create a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(mse_matrix, annot=True, fmt=".2f", cmap='viridis', xticklabels=[f'Judge {i+1}' for i in range(num_judges)], yticklabels=[f'Judge {i+1}' for i in range(num_judges)])
    plt.title(f'Pairwise MSE heatmap for {model_name} predictions of each judge')
    plt.xlabel('Judge Index')
    plt.ylabel('Judge Index')
    plt.show()
    return mse_matrix
    
def main():
    # Load the data
    df_characteristics, df_jud = load_the_data()
        
    # loop over the judges and run decision trees and return predictions
    all_predictions_trees = predictions_with_smote_decision_tree(df_characteristics, df_jud)
    all_predictions_linear_regression = predictions_with_smote_linear_regression(df_characteristics, df_jud)

    # Print the decision trees predictions for all judges
    for i, preds in enumerate(all_predictions_trees):
        print(f"Decision tree predictions for Judge {i + 1}: {preds}")

    mse_predictions(all_predictions_trees, model_name='Decision Tree')
    
    # Print the linear regression predictions for all judges
    for i, preds in enumerate(all_predictions_linear_regression):
        print(f"Linear regression predictions for Judge {i + 1}: {preds}")
        
    mse_predictions(all_predictions_linear_regression, model_name='Linear Regression')
    
    
main()
