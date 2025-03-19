
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import r2_score,explained_variance_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from datetime import datetime
import traceback
import warnings

warnings.filterwarnings('ignore')

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

def cluster_countries(df):
    
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

    # extract the df for the prediction
    pred_df = country_df.copy(deep=True)  
    pred_df = pred_df.reset_index(drop=True)

    country_df['cluster_num'] = kmeans.labels_
    country_df = country_df.groupby('Entity').median().reset_index()
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

    #plt.show()

    return pred_df

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

    df['Success'] = 0

    metrics = ['PR','TPR','MR','CFR','TR','PRM','MRM','TRM']

    for metric in metrics:
        condition = df[metric] < df[metric].median()
        df['Success'] += condition.astype(int)

    half = len(metrics)//2
    df['Success'] = (df['Success'] >= half).astype(int)

    return df
    
def SVM_predict(date,df):
    try:
        
        df = df.drop(columns = ['Cases','TPR','CFR','TRM','PRM','Success'])
        counts = df.nunique()
        const_cols = counts[counts==1].index
        df.drop(const_cols,axis=1,inplace=True)
        date = pd.to_datetime(date)
        end_date = date + pd.DateOffset(days=3)
        pred_pr_df = df.loc[(df['Date'] >= date) & (df['Date'] <= end_date)]
        df = df.loc[df['Date'] < date]

        enc = LabelEncoder()
        date_num  = enc.fit_transform(df['Date'])
        df.drop('Date',axis=1,inplace=True)
        df['Date'] = date_num

        keep = df.columns.difference(['PR'])
        x = df[keep]
        y = df['PR']

        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,\
                                                        random_state=42)
        scaler = StandardScaler()
        x_train_sc = scaler.fit_transform(x_train)
        x_test_sc = scaler.transform(x_test)

        svm = SVR(kernel='linear')
        svm.fit(x_train_sc,y_train)

        y_test = svm.predict(x_test_sc)

        y_real = pred_pr_df['PR']
        
        pred_pr_df['PR'] = np.nan
        pred_pr_df = pred_pr_df.rename(columns={"PR" : "Predicted PR"})
        pred_pr_df = pred_pr_df.copy(deep=True)
        date_num = enc.fit_transform(pred_pr_df['Date'])
        pred_pr_df['Date'] = date_num
        
        keep = pred_pr_df.columns.difference(['Predicted PR'])
        x_prediction = pred_pr_df[keep]
        x_prediction_sc = scaler.transform(x_prediction)

        predicted_pr = svm.predict(x_prediction_sc)

        result_df = create_data(date)
        result_df['Predicted PR'] = predicted_pr
        result_df['Real PR'] = y_real.values

        mae = mean_absolute_error(y_real.values,predicted_pr)
        mse = mean_squared_error(y_real.values,predicted_pr)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_real.values,predicted_pr)
        evs = explained_variance_score(y_real.values,predicted_pr)

        metrics = [mae,mse,rmse,r2,evs]
        metric_names = ["MAE","MSE","RMSE","R2","EVS"]

        print("\n-----SVM PREDICTION-----\n")
        print(result_df.iloc[:, 0:4].tail(4).to_string(index=False))
        for name,metric in zip(metric_names,metrics) :
            print("{} = {:.3f}".format(name,metric))

    except ValueError:
        print("Something went Wrong")
        traceback.print_exc()
    return

def RNN_predict(date,df):

    try:
        
        df = df.drop(columns = ['Cases','TPR','CFR','TRM','PRM','Success'])
        counts = df.nunique()
        const_cols = counts[counts==1].index
        df.drop(const_cols,axis=1,inplace=True)
        date = pd.to_datetime(date)
        end_date = date + pd.DateOffset(days=3)
        df.rename_axis(index=None,axis=1,inplace = True)
        pred_pr_df = df.loc[(df['Date'] >= date) & (df['Date'] <= end_date)]
        df = df.loc[df['Date'] < date]

        enc = LabelEncoder()
        date_num  = enc.fit_transform(df['Date'])
        df.drop('Date',axis=1,inplace=True)
        df['Date'] = date_num
        
        keep = df.columns.difference(['PR'])
        x = df[keep]
        y = df['PR']

        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,\
                                                        random_state=42)
        scaler = StandardScaler()
        x_train_sc = scaler.fit_transform(x_train)
        x_test_sc = scaler.transform(x_test)

        x_train_reshaped = np.reshape(x_train_sc,(x_train_sc.shape[0],\
                                                  x_train_sc.shape[1],1))
        x_test_reshaped = np.reshape(x_test_sc,(x_test_sc.shape[0],\
                                                  x_test_sc.shape[1],1))
        rnn = Sequential()
        rnn.add(LSTM(350,input_shape=(x_train_reshaped.shape[1],1)))
        rnn.add(Dense(1))
        rnn.compile(optimizer='adam',loss='mean_squared_error')
        rnn.fit(x_train_reshaped,y_train,epochs=15,batch_size=32)

        y_test = rnn.predict(x_test_reshaped)
        
        y_real = pred_pr_df['PR']

        pred_pr_df['PR'] = np.nan
        pred_pr_df = pred_pr_df.rename(columns={"PR" : "Predicted PR"})
        pred_pr_df = pred_pr_df.copy(deep=True)
        date_num = enc.fit_transform(pred_pr_df['Date'])
        pred_pr_df['Date'] = date_num
        
        keep = pred_pr_df.columns.difference(['Predicted PR'])
        x_prediction = pred_pr_df[keep]
        x_prediction_sc = scaler.transform(x_prediction)
        x_pred_reshaped = np.reshape(x_prediction_sc,(x_prediction_sc.shape[0],\
                                                  x_prediction_sc.shape[1],1))
        predicted_pr = rnn.predict(x_pred_reshaped)
        
        result_df = create_data(date)
        result_df['Predicted PR'] = predicted_pr
        result_df['Real PR'] = y_real.values

        mae = mean_absolute_error(y_real.values,predicted_pr)
        mse = mean_squared_error(y_real.values,predicted_pr)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_real.values,predicted_pr)
        evs = explained_variance_score(y_real.values,predicted_pr)
        
        metrics = [mae,mse,rmse,r2,evs]
        metric_names = ["MAE","MSE","RMSE","R2","EVS"]


        print("\n-----SVM PREDICTION-----\n")
        print(result_df.iloc[:, 0:4].tail(4).to_string(index=False))
        for name,metric in zip(metric_names,metrics) :
            print("{} = {:.3f}".format(name,metric))

    except ValueError:
        print("Something went Wrong")
        traceback.print_exc()
        
    return

def create_data(date) :

    date = pd.to_datetime(date)
    date_range = pd.date_range(date,periods = 4)
    entity = ['Greece'] * len(date_range)
    date = pd.Series(date_range,name='Date')
    pred_PR = [np.nan] * len(date_range)
    real_PR = [np.nan] * len(date_range)

    df = pd.DataFrame({'Entity':entity,'Date':date,'Predicted PR':pred_PR,\
                       "Real PR":real_PR})
    
    return df

def predict(df):

    print('\nPositivity Rate Predictor Model for Greece after 1/1/2021\n')

    enter = str(input("Do you want to proceed? (y|n) : "))

    while enter.lower() != 'n':

        if enter.lower() == 'y':
            valid_format = '%Y-%m-%d'
            valid_date = datetime.strptime("2020-12-31",'%Y-%m-%d')

            date_input = input("Enter a date for the predicted PR in the format yyyy-mm-dd: ")
            
            try:
                date = datetime.strptime(date_input,valid_format)

                if date > valid_date:
                    print("Valid date entered: {} .".format(date))
                    
                    while True:
                        model = int(input("Enter preferred model for prediction, 1 for SVM and 2 for RNN: "))

                        if model == 1:
                            SVM_predict(date,df)
                            break
                        elif model == 2:
                            RNN_predict(date,df)
                            break
                        else:
                            print("Invalid model number argument. Please try again.")
                else:
                    print("Date is not after 2020-12-31. Please try again.")
                        
            except ValueError:
                print("Invalid date format. Please try again.")
        else:
            print("Invalid input. Enter 'y' or 'n'.")
        enter = str(input("Do you want to use the tool again? (y|n) : "))
        
    print("Exiting Greece Positivity Rate Predictor...\n")

    return

# def real_data(date,df):

#     real_df = df[df['Entity'] == 'Greece']
#     real_df = real_df[real_df['Date'] >= date]
#     real_df['PR'] = (real_df['Cases'] / real_df['Population'])*100
#     real_df = real_df[['Entity','Date','PR']]
#     real_df.reset_index(inplace=True)

#     return real_df

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
pred_df = cluster_countries(cluster_det_df)
#-----------------------Q3-------------------------------------
greece_df = pred_df[pred_df['Entity'] == 'Greece']
predict(greece_df)
    
    

