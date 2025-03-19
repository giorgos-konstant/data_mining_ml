import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time

def preprocessing():
    print("\nStarted Preprocessing...\n")
    start = time.time()
    stats_df = pd.read_csv("data.csv")
    na_cols = ['Daily tests','Cases','Deaths']
    stats_df[na_cols] = stats_df[na_cols].apply(pd.to_numeric)
    stats_df[na_cols] = stats_df[na_cols].interpolate(method='linear',limit =5,limit_direction='both')
    stats_df[na_cols] = stats_df[na_cols].interpolate(method='spline',order=3,limit_direction='both')
    stats_df[na_cols] = stats_df[na_cols].clip(lower=0)
    stats_df[na_cols] = stats_df[na_cols].round().astype(int)

    stats_df.to_csv("filled_data.csv")

    print("\nPre-processing lasted {:.2f} minutes\n".format((time.time()-start)/60))

    return


preprocessing()