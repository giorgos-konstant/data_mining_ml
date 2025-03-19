
import pandas as pd;
import numpy as np;
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

def daily(df):

    # keep daily data from the columns of most interest , used for determining stats_metrics
    df_daily = df.copy(deep=True)
    df_daily = df_daily[['Entity','Population',\
                            'Average temperature per year','Hospital beds per 1000 people',\
                            'Medical doctors per 1000 people','GDP/Capita','Date',\
                            'Daily tests','Cases','Deaths']]
    df_daily = df_daily.reset_index()

    return  df_daily

def stat_metrics(df):

    #calculate metrics per day, all are as percentages

    #PR -> Positivy Rate: positive cases per population
    #TPR -> Test Positivy Rate: positive cases per test
    #MR -> Mortallity Rate: deaths per population
    #CFR -> Case Fatality Rate : deaths per case
    #TR -> Test Rate : tests per population
    df['PR'] = (df['Cases'] / df['Population'])*100
    df['TPR'] = (df['Cases'] / df['Daily tests'])*100
    df['TPR'] = df['TPR'].replace(np.inf,0)
    df['MR'] = (df['Deaths']/df['Population'])*100
    df['CFR'] = (df['Deaths']/df['Cases']*100)
    df['CFR'] = df['TPR'].replace(np.nan,0)
    df['TR'] = (df['Daily tests']/df['Population'])*100
    
    # calculate PR,MR,TR per million people rather than in the whole population

    df['PRM'] = (df['Cases']/1e6)*100
    df['MRM'] = (df['Deaths']/1e6)*100
    df['TRM'] = (df['Daily tests']/1e6)*100

    return df

def success(df):

    # determine success of each country per day depending
    # on values of metrics caluclated in stats_metrics function
    # performance is considered "success" or "1" if its
    # below the median value of the metric

    df['Success'] = 0

    metrics = ['PR','TPR','MR','CFR','TR','PRM','MRM','TRM']

    for metric in metrics:
        condition = df[metric] < df[metric].median()
        df['Success'] += condition.astype(int)

    half = len(metrics)//2
    df['Success'] = (df['Success'] >= half).astype(int)

    return df

def cluster_countries(df):
    

    # clusters countries using kMeans, scatter plot is graphed using
    # country name and positivity rate , colors mapped depending
    # on cluster number
    
    country_df = df.copy(deep=True)
    enc = LabelEncoder()
    country_num = enc.fit_transform(df['Entity'])
    date_num  = enc.fit_transform(df['Date'])
    df.drop(['Entity','Date'],axis=1,inplace=True)
    df['Date'] = date_num
    df['Entity'] = country_num

    temp = df.copy(deep=True)
    scaler = StandardScaler().fit(temp)
    temp = scaler.transform(temp)
    kmeans = KMeans(random_state=125,max_iter=100,n_init='auto').fit(temp)

    country_df['cluster_num'] = kmeans.labels_
    country_df = country_df.groupby('Entity').median(numeric_only=True).reset_index()
    country_df['Success'] = round(country_df['Success'])
    
    #remap if needed
    # remap = [i for i in range(len(np.unique(country_df['cluster_num'])))]
    # u_labels = np.unique(country_df['cluster_num'])
    # cluster_remap = {int(k): v for k,v in zip(u_labels,remap)}

    # country_df['cluster_num'] = country_df['cluster_num'].map(cluster_remap)
    
    x = country_df['Entity']
    y = country_df['PR']
    labels = country_df['cluster_num']
    unique_clusters = np.sort(country_df['cluster_num'].unique())
    cmap = plt.cm.get_cmap('plasma',\
                                      len(unique_clusters))

    colors = [cmap(i) for i in range(len(unique_clusters))]

    for c,i in zip(colors,np.unique(country_df["cluster_num"])) :
      plt.scatter(x[labels==i],y[labels==i],color=c,label=x)
    
    plt.yticks([])
    plt.xticks([])
    plt.title("KMeans Clustering Results")
    
    
    legend_elems = [plt.Line2D([0],[0],marker='o',color='w',\
                                label = "Cluster {}".format(int(cluster)),\
                                markerfacecolor=cmap(i)) \
                                for i,cluster in enumerate(unique_clusters)]
    plt.legend(handles=legend_elems,loc='center right')
    
    #this prints the countries in each cluster
    # cluster_countries = {}
    # for i,row in country_df.iterrows():
    #     cluster = row['cluster_num']
    #     country = row['Entity']
    #     if cluster not in cluster_countries:
    #         cluster_countries[cluster] = [country]
    #     else:
    #         cluster_countries[cluster].append(country)
    
    # for cluster,countries in cluster_countries.items():
    #     print(f"Cluster {int(cluster)}: {', '.join(countries)}")

    plt.show()

    return country_df


stats_df = pd.read_csv("data.csv",index_col=[0])
stats_df['Date'] = pd.to_datetime(stats_df['Date'])
filled_stats_df = pd.read_csv('filled_data.csv',index_col=[0])
filled_stats_df['Date'] = pd.to_datetime(filled_stats_df['Date'])
#------------------Q1--------------------------------------------
basic_stats(filled_stats_df)
behave_over_time(stats_df,['Daily tests','Cases','Deaths'])
corr_map(filled_stats_df)
country_graphs(filled_stats_df,['Daily tests','Cases','Deaths'])
#-----------------------------Q2---------------------------------
daily_data = daily(filled_stats_df)
metrics_df = stat_metrics(daily_data) 
cluster_det_df = success(metrics_df) #df that determines clusters
clust_df = cluster_countries(cluster_det_df)


