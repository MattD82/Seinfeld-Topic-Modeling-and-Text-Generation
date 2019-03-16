# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_lines_spoken(df_scripts):
    fig = plt.figure(figsize=(10,5))
    ax = sns.countplot(x="Character", 
                    data=df_scripts, 
                    order=pd.value_counts(df_scripts['Character']).iloc[:13].index,
                    palette="deep")
    ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
    ax.set_title("Total Lines Spoken By Main Characters")
    plt.show()