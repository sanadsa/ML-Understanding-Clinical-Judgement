from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display

def load_the_data():
    from sklearn.preprocessing import MinMaxScaler
    judeges = pd.read_csv('assets/MEELJUD.csv')
    data = pd.read_csv('assets/MEELMMPI.csv')

    judeges=judeges.drop(judeges.columns[-1],axis=1)
    data=data.drop(data.columns[-1],axis=1)
    
    #Create a list of lists where each sub-list represents a column of the 'judeges' dataframe and contains all the values of that column
    judges_list = [judeges.iloc[:, i].tolist() for i in range(len(judeges.columns))]
    return data, judges_list, judeges.columns

# def split_data(data, judges_list):
#     from sklearn.model_selection import train_test_split

#     # Split the data and labels into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(data, judges_list[1], test_size=0.2, random_state=42)
#     return X_train, X_test, y_train, y_test

def get_decision_trees(X, y):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    from sklearn.tree import export_graphviz
    from sklearn.metrics import confusion_matrix
    import graphviz

    decision_trees = []
    confusion_matrics = []

    for i, judge in enumerate(y):
        clf = DecisionTreeClassifier(max_depth=3)
        X_train, X_test, y_train, y_test = train_test_split(X, judge, test_size=0.2, random_state=42)

        clf.fit(X_train, y_train)
        print(tree.plot_tree(clf))
        y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        confusion_matrics.append(cm)

        decision_trees.append(clf)
        
        #plt.figure(figsize=(14 ,8))    
        #tree.plot_tree(decision_trees[i], filled=True, fontsize=8)   
        
        dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=[str(i) for i in range(0,11)], filled=True)

        graph = graphviz.Source(dot_data)
        # display(graph)

    # plt.tight_layout()
    # plt.show()
    # plt.savefig("decision_trees.pdf")
    
    return decision_trees, confusion_matrics

def two_tree_distance(tree1, tree2):
    set1 = set(tree1.tree_.feature)
    set2 = set(tree2.tree_.feature)

    print("set1: tree features")
    print(set1)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    jaccard_similarity = len(intersection) / len(union)
    tree_distance = 1 - jaccard_similarity
    return tree_distance

def tree_distance_heatmap(decision_trees):
    n = len(decision_trees)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = two_tree_distance(decision_trees[i], decision_trees[j])
    
    display(pd.DataFrame(matrix))

    # Create a heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='coolwarm')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Distance', rotation=-90, va="bottom")

    # Set tick labels and axis labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(np.arange(n), fontsize=8)
    ax.set_yticklabels(np.arange(n), fontsize=8)
    ax.set_xlabel('Tree Index', fontsize=10)
    ax.set_ylabel('Tree Index', fontsize=10)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                           ha="center", va="center", color="w", fontsize=8)

    plt.show()
    return matrix

# def tree_to_bag_of_conditions(tree):
#     return [x.split('\n')[0] for x in tree.strip().split("--- ")[1:] if "class" not in x]

# def similarity(tree1, tree2):
#     bag1 = tree_to_bag_of_conditions(tree1)
#     bag2 = tree_to_bag_of_conditions(tree2)
#     matchings = 0
#     for condition in bag1:  
#         if condition in bag2:
#             if two_conditions_match(bag1, bag2):
#                 matchings += 1
#                 break
#     return matchings

# def two_conditions_match(tree1, tree2, threshold):
#     if tree1["feature"] != tree2["feature"]:
#         return None
    
#     if tree1["sign"] != tree2["sign"]:
#         return None
    
#     if tree1["value"] < tree2["value"]:
#         normalized_value = (tree1["value"] - tree1["min_value"]) / (tree1["max_value"] - tree1["min_value"])
#     else:
#         normalized_value = (tree2["value"] - tree2["min_value"]) / (tree2["max_value"] - tree2["min_value"])
    
#     if normalized_value < threshold:
#         return 1
#     else:
#         return 0

# def plot_disagreement_matrix(confusion_matrices):
#     import numpy as np
#     from sklearn.metrics import confusion_matrix

#     num_doctors = 29
#     disagreement_matrix = np.zeros((num_doctors, num_doctors))

#     for i in range(num_doctors):
#         for j in range(num_doctors):
#             disagreement_matrix[i, j] = np.sum(confusion_matrices[i] != confusion_matrices[j])

#     print("Disagreement Matrix:")
#     print(disagreement_matrix)

def judges_heatmap(judges_list, judges_names):
    import matplotlib.pyplot as plt
    import numpy as np

    num_doctors = 29
    num_patients = 860
    judges_matrix = np.zeros((num_doctors, num_doctors))

    for i in range(num_doctors):
        for j in range(num_doctors):
            # heatmap will contain
            # the value from the first patient from the first doc
            # minus the value from the first patient from the second doc
            # plus the value from the second patient from the first doc
            # minus the value from the second patient from the second doc
            # and so on...
            distance = 0
            for patient in range(0, num_patients):
                distance += judges_list[i][patient] - judges_list[j][patient]
            judges_matrix[i, j] = distance

    # Create a heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(judges_matrix, cmap='coolwarm')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Distance', rotation=-90, va="bottom")

    # Set tick labels and axis labels
    ax.set_xticks(np.arange(num_doctors))
    ax.set_yticks(np.arange(num_doctors))
    ax.set_xticklabels(judges_names, fontsize=8)
    ax.set_yticklabels(judges_names, fontsize=8)
    ax.set_xlabel('Judge', fontsize=10)
    ax.set_ylabel('Judge', fontsize=10)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    for i in range(num_doctors):
        for j in range(num_doctors):
            text = ax.text(j, i, f'{judges_matrix[i, j]:.2f}',
                           ha="center", va="center", color="w", fontsize=8)

    plt.show()

    df = pd.DataFrame(judges_matrix)
    display(df)

def main():
    data, judges_list, judges_names = load_the_data()
    judges_heatmap(judges_list, judges_names)
    decision_trees, confusion_matrics = get_decision_trees(data, judges_list)
    tree_distance_heatmap(decision_trees)

main()