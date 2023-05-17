from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree
import numpy as np
import pandas as pd

def load_the_data():
    from sklearn.preprocessing import MinMaxScaler
    judeges = pd.read_csv('MEELJUD.csv')
    data = pd.read_csv('MEELMMPI.csv')

    judeges=judeges.drop(judeges.columns[-1],axis=1)
    data=data.drop(data.columns[-1],axis=1)
    
    #Create a list of lists where each sub-list represents a column of the 'judeges' dataframe and contains all the values of that column
    judges_list = [judeges.iloc[:, i].tolist() for i in range(len(judeges.columns))]
    return data, judges_list, judeges.columns

data, judges_list, judges_names = load_the_data()

# def split_data(data, judges_list):
#     from sklearn.model_selection import train_test_split

#     # Split the data and labels into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(data, judges_list[1], test_size=0.2, random_state=42)
#     return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = split_data(data, judges_list)


def show_tree(X, y):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import datasets
    from sklearn import tree
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import confusion_matrix

    clf = DecisionTreeClassifier()
    decision_trees = {}
    confusion_matrics = []

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    for i, doctor in enumerate(y):
        # split the data and labels into training and test sets
        y_train, y_test = train_test_split(doctor, test_size=0.2, random_state=42)
        
        # Create and train the decision tree classifier
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Calculate the confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Append the confusion matrix to the list
        confusion_matrics.append(cm)

        # Print the confusion matrix for the current target variable
        # print(f"Confusion Matrix for Model {i+1}:")
        # print(cm)
        # print()

        # Store the decision tree in a dictionary
        decision_trees[i] = clf
    
    tree.plot_tree(decision_trees[i], filled=True, fontsize=8)   
    plt.show()

    # predictions_doctor1 = decision_trees[1].predict(X_test)
    # predictions_doctor2 = decision_trees[2].predict(X_test)


    return decision_trees, confusion_matrics


decision_trees, confusion_matrics = show_tree(data, judges_list)


def plot_disagreement_matrix(confusion_matrices):
    import numpy as np
    from sklearn.metrics import confusion_matrix

    num_doctors = 29
    disagreement_matrix = np.zeros((num_doctors, num_doctors))

    for i in range(num_doctors):
        for j in range(num_doctors):
            disagreement_matrix[i, j] = np.sum(confusion_matrices[i] != confusion_matrices[j])

    print("Disagreement Matrix:")
    print(disagreement_matrix)

plot_disagreement_matrix(confusion_matrics)
