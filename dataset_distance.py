import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

meeljud_df = pd.read_csv('assets/MEELJUD.csv')
meelmmpi_df = pd.read_csv('assets/MEELMMPI.csv')
judges_evaluations = meeljud_df.iloc[:, :-1]

# euclidean distance
distance_matrix = pdist(judges_evaluations.T, metric='euclidean')
distance_matrix_square = squareform(distance_matrix)

distance_df = pd.DataFrame(distance_matrix_square, index=judges_evaluations.columns, columns=judges_evaluations.columns)

plt.figure(figsize=(10, 8))
sns.heatmap(distance_df, annot=True, cmap='viridis')
plt.title('Judges Distance Matrix')
plt.show()


# Numerical Values:
# Each cell (i,j) in the matrix represents the Euclidean distance between the evaluations of Judge 
# i and Judge j.
# For example, the distance between Judge J1 and Judge 
# J2 is 51. This means the evaluations given by Judge J1 and Judge 
# J2 are 51 units apart in the Euclidean space formed by the patients' scores.

# Interpretation:
# Smaller values indicate that two judges have similar evaluations of the patients. For instance, the distance between Judge 
# J2 and Judge J5 is 44, suggesting they have relatively similar evaluation patterns.
# Larger values indicate that two judges have more dissimilar evaluations. For instance, the distance between Judge J10 and Judge 
# J14 is 77, suggesting their evaluations differ significantly.

# Pattern Analysis:
# By examining clusters of low distances, we can identify groups of judges who have similar evaluation patterns. Conversely, clusters of high distances can identify judges whose evaluations are quite different from others.
# Judges with consistently low distances to most other judges might indicate more typical or average evaluation patterns, while those with higher distances might indicate unique or atypical evaluation styles.

# This matrix helps in understanding how similarly or differently the judges evaluate patients based on the MMPI scores. It can be useful for identifying outlier judges or understanding the consensus among the judges.