import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
 
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
 
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import friedmanchisquare
from scipy import stats
from sklearn.metrics import confusion_matrix
 
import keras
from keras.models import Sequential
from keras.layers import Dense

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/website-phishing-data-set/Website Phishing.csv')
# Replacing the phishy class by the value 2
# I decided to use MLP so it would be better if
# the classes were all non negative values

df['Result'] = df['Result'].astype(str).replace('-1','2').astype(np.int64)
# Visualizing the data
df.head()
# Visualizing the data
df.tail()
#  Calculating if there are null or na values in the dataset
print('Verifying null and na data')
print()
print(df.isna().any())

print()
print(df.isnull().any())

# Obtaining some additional information about the dataset
df.info()
def process_data_class_distribution(df):
    class_names = ['Legitimate','Suspicious', 'Phishy']

    class_samples = [len(df[df['Result'] == 1]), len(df[df['Result'] == 0]), len(df[df['Result'] == 2])]

    data = {
      'Class': class_names,
      'Samples': class_samples,
    }
  
    df_class = pd.DataFrame(data)

    df_class.sort_values(by='Samples', ascending=True, inplace=True)

    return df_class

def fig_class_distribution(df):
    fig = go.Figure()

    for idx,classe in enumerate(list(df.Class.unique())):

        if idx == 0: # the class with less samples will be highlighted
            color = 'rgb(90, 90, 200)'

        else:
            color = 'rgb(120, 120, 120)'

        fig.add_trace(go.Bar(y=[classe], x=df[df['Class'] == classe]['Samples'], marker_color=color, name=classe, orientation='h'))
    
    return fig
    def update_layout_bar_chart(fig):
        fig.update_layout(
        title='Class Distribution', 
        font=dict(family="Arial", size=16),
        plot_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, b=20, t=50),
        width=600,
        height=400,
        hoverlabel=dict(
            font_color='rgb(120, 120, 120)',
            bgcolor="white",
            font_size=16,
            font_family="Arial"
        ),
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(family='Arial', size=16, color='rgb(82, 82, 82)')
        ),
        yaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(family='Arial', size=16, color='rgb(82, 82, 82)')
        )
    )
    return fig

# Now we will generate and display the class distribution bar chart
df_class = process_data_class_distribution(df)
fig = fig_class_distribution(df_class)
fig = update_layout_bar_chart(fig)
fig.show()
