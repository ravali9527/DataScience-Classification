from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



DIR='/Users/ravali/Desktop/DataScience-Classification/Data'
df = pd.read_csv(DIR+'/shakesphere.csv', delimiter=',')
sns.FacetGrid(df, hue="Player").map(plt.scatter, "PlayerLine", "PlayerLinenumber").add_legend()

plt.show()

