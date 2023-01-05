import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def datasave(price_record, time, file_path):
    df = pd.DataFrame(list(zip(price_record[0], price_record[1])),
                      columns=['Firm1', 'Firm2'])
    file_path_csv = file_path + 'out_'+str(time)+'.csv'
    df.to_csv(file_path_csv, index=False)

def finaldatasave(price_record, file_path):
    df = pd.DataFrame(list(zip(price_record[0], price_record[1])),
                      columns=['Firm1', 'Firm2'])
    file_path_csv = file_path + 'finalout.csv'
    df.to_csv(file_path_csv, index=False)

def plot(price_record, time, file_path):
    myFmt = mdates.DateFormatter('%Y')
    colours = sns.color_palette("Set2")
    colours += sns.color_palette("husl", 9)
    colours += sns.color_palette()


    the_title = 'out' + str(time)

    fig, axes = plt.subplots(figsize=(18, 15))

    T  = len(price_record[0])
    mean_window = T // 100
    x = np.array(range(int(T - mean_window + 1)))
    y1 = moving_average(price_record[0], mean_window)
    y2 = moving_average(price_record[1], mean_window)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.ylim(0.1, 1.1)
    plt.grid(visible=True, which='major', axis='both')
    plt.savefig(file_path + the_title + '.png')
    plt.show()