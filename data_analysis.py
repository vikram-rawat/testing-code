import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import plotnine as pln
import matplotlib.pyplot as plt
import matplotlib as mpl

load_dotenv()

mpl.rcParams['backend'] = os.environ['plot_rcParams_backend']
mpl.rcParams['toolbar'] = os.environ['plot_rcParams_toolbar']
mpl.rcParams['interactive'] = os.environ['plot_rcParams_interactive']
mpl.rcParams['figure.figsize'] = os.environ['plot_rcParams_figure_figsize']

sales_data = pd.read_csv("data/sales_data.csv")

sales_data.head()
sales_data.tail()
sales_data.describe()
sales_data.info()
sales_data.shape
sales_data.describe().Day

sales_data.iloc[:, :]

sales_data["Unit_Cost"].describe()
sales_data["Unit_Cost"].mean()
sales_data["Unit_Cost"].median()
sales_data["Unit_Cost"].plot(kind = "box", vert=False, figsize=(14,6))

sales_data["Age_Group"].value_counts()
sales_data["Unit_Cost"].value_counts().describe()

sales_data["Unit_Cost"].value_counts().plot(kind="pie")
sales_data["Unit_Cost"].value_counts().plot.pie()

df = pd.DataFrame({'mass': [0.330, 4.87 , 5.97],
                   'radius': [2439.7, 6051.8, 6378.1]},
                  index=['Mercury', 'Venus', 'Earth'])
plot = df.plot.pie(y='mass', figsize=(5, 5))

import pandas as pd

df = pd.DataFrame([2,5,67,2,3,5,23,124])
plot = df.hist()
plot.show()
plt.show()
