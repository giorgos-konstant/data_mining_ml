
import pandas as pd;
import numpy as np;
from matplotlib import pyplot as plt
import time
import seaborn as sns

def basic_stats(df):

    # extract basic stats of the dataframe
    summary_df = df.describe(include='number').round(2)
    summary_df = summary_df.drop('count')
    print(summary_df)
    
    return

def behave_over_time(df,cols):

    # keep median value from each column at each day to get a clear picture of how the
    # running median value is , we do that to the df that hasnt gone through preprocessing
    # to get a view of the already existent data
   
    for col in cols :

        median_per_day = df.groupby('Date')[col].median()
        median_per_day.plot(x='Date',y=col)
        plt.title(col+' over Time')
        plt.xlabel("Date")
        plt.ylabel(col)
        plt.show(block=True)
    
    return

def corr_map(df):
    
    # Correlation Heatmap to see possible relationships between columns,
    # some get renamed for better fit on graph
    cols = ['Average temperature per year','Population aged 65 and over (%)',\
              'Median age','Population','GDP/Capita',\
              'Medical doctors per 1000 people',\
              'Hospital beds per 1000 people']
    
    rename = ['Avg Temp',"% >=65",'Median age','Population','GDP/Capita','MDper1000',\
              "HBper1000"]
    rename_dict = dict(zip(cols,rename))
    temp = df.rename(columns=rename_dict)
    corr = temp.corr(numeric_only=True)
    plt.figure(figsize=(10,8))
    sns.heatmap(corr,annot= True,cmap='YlOrRd')
    plt.title("Correlation Heatmap")
    plt.show(block=True)

    return

def country_graphs(df,cols):

    #grid of plots showing how Tests, Cases, Deaths change
    #depending on selected parameters

    params = ['Average temperature per year','Population aged 65 and over (%)',\
              'Median age','Population','GDP/Capita',\
              'Medical doctors per 1000 people',\
              'Hospital beds per 1000 people']
    
    country_max = df.groupby('Entity').median(numeric_only = True)
    country_max = country_max.reset_index()
    rename_params = ['Avg Temp',"% >=65",'Median age','Population','GDP/Capita','MDper1000',\
              "HBper1000"]
    rename_dict = dict(zip(params,rename_params))
    country_max.rename(columns=rename_dict,inplace=True)
    num_rows = len(cols)
    num_cols = len(params)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, col in enumerate(cols):
        for j, param in enumerate(rename_params):
            x = []
            y = []

            for index, row in country_max.iterrows():
                entity = row['Entity']
                val_x = row[param]
                val_y = row[col]
                x.append(val_x)
                y.append(val_y)

            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            trend_line = p(x)

            ax = axs[i, j]
            if param != 'Population':
                if col == 'Daily tests' : y = np.clip(y,0,1e5)
                elif col == 'Cases' : y = np.clip(y,0,0.5*1e6)
                else : y = np.clip(y,0,0.2*1e5)
                ax.scatter(x,y)
            else : ax.scatter(x, y)
            ax.plot(x, trend_line, color='red')
            ax.set_xlabel('\n'+param,fontsize=8)
            if (j==0) : ax.set_ylabel(col,fontsize=8)
            if (j>0) : ax.yaxis.set_ticks([])
            
            ax.tick_params(axis='x', labelsize=5)  
            ax.tick_params(axis='y', labelsize=5)
            ax.yaxis.offsetText.set_fontsize(5)
            ax.xaxis.offsetText.set_fontsize(5) 
    plt.tight_layout()
    plt.show()

    return

stats_df = pd.read_csv("data.csv",index_col=[0])
filled_stats_df = pd.read_csv('filled_data.csv',index_col=[0])
basic_stats(filled_stats_df)
behave_over_time(stats_df,['Daily tests','Cases','Deaths'])
corr_map(filled_stats_df)
country_graphs(filled_stats_df,['Daily tests','Cases','Deaths'])



