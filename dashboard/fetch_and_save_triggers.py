from dashboard.utils import fetch_trigs, max_time_from_csv, remove_duplicates, convert_to_parquet
import pandas as pd
import gwpy
import time
import numpy as np

tstart_L1 = max_time_from_csv('data/O4_L1.csv', col='time')
tstart_H1 = max_time_from_csv('data/O4_H1.csv', col='time')
now_gps = gwpy.time.tconvert(gpsordate='now').gpsSeconds
tend_L1 = tend_H1 = now_gps


print("Fetching new Omicron triggers now")

trigs_L1 = fetch_trigs(tstart_L1, tend_L1, 'L1')
trigs_L1 = trigs_L1[trigs_L1['time'] > tstart_L1]

if trigs_L1 is not None and len(trigs_L1) > 0:
    # print(tstart_L1)
    # print(list(trigs_L1['time']))
    # Append to the old file
    trigs_L1.to_csv('data/O4_L1.csv', mode='a', index=False, header=False)
    print(f"Appended {len(trigs_L1)} rows to data/O4_L1.csv")
else:
    print("No new triggers found for L1")

trigs_H1 = fetch_trigs(tstart_H1, tend_H1, 'H1')
trigs_H1 = trigs_H1[trigs_H1['time']>tstart_H1]

if trigs_H1 is not None and len(trigs_H1) > 0:
    # print(tstart_H1)
    # print(list(trigs_H1['time']))
    trigs_H1.to_csv('data/O4_H1.csv', mode='a', index=False, header=False)
    print(f"Appended {len(trigs_H1)} rows to data/O4_H1.csv")
else:
    print("No new triggers found for H1")


# Check for duplicates
remove_duplicates('data/O4_L1.csv')
remove_duplicates('data/O4_H1.csv')

# Convert to parquet
convert_to_parquet('data/O4_L1.csv')
convert_to_parquet('data/O4_H1.csv')