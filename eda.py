import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from zipfile import ZipFile

# Load dataset
with ZipFile("dataset/archive.zip", 'r') as z:
    with z.open("Human Development Index - Full.csv") as f:
        df = pd.read_csv(f)

df.columns = df.columns.str.strip()

# HDI distribution
sns.histplot(df["Human Development Index (2021)"], kde=True)
plt.title("HDI Distribution")
plt.show()

# Correlation heatmap
cols = [
    "Human Development Index (2021)",
    "Human Development Index (2020)",
    "Human Development Index (2019)",
    "HDI Rank (2021)"
]

sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Scatter plot
sns.scatterplot(x=df["HDI Rank (2021)"],
                y=df["Human Development Index (2021)"])
plt.title("HDI Rank vs HDI Score")
plt.show()