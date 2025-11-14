from gwtrigfind import find_trigger_files
from gwpy.time import to_gps, from_gps
from gwpy.table import EventTable
import pandas as pd
from datetime import date


def fetch_trigs(start_time, end_time, ifo):

    t1 = to_gps(start_time)
    t2 = to_gps(end_time)
    channel = f'{ifo}:GDS-CALIB_STRAIN_NOLINES'
    trigs = pd.DataFrame()
    try:
        cache = find_trigger_files(channel, 'omicron',t1, t2, ext='h5')
        trigs = EventTable.read(cache, format='hdf5', path='triggers')
    
        trigs = trigs.to_pandas()
        trigs = trigs[trigs['snr']>6]
        trigs['duration'] = trigs['tend'] - trigs['tstart']
        trigs['bandwidth'] = trigs['fend'] - trigs['fstart']
        trigs.drop(['phase', 'q', 'tstart', 'tend', 'fstart', 'fend'], axis=1, inplace=True)

        trigs['dates'] = trigs['time'].apply(lambda t: from_gps(t).strftime('%Y-%m-%d'))
        trigs = trigs[['time', 'frequency', 'snr', 'bandwidth', 'duration', 'dates']]
        trigs.reset_index(drop=True, inplace=True)

    except Exception as e:
        print(f"Error while fetching triggers: {e}")
        pass


    if len(trigs)>0:
        trigs.reset_index(drop=True, inplace=True)
        return trigs

    return []


def max_time_from_csv(path, col="time", chunksize=1_000_000):
    """Return max value from a single column without loading the whole file."""
    max_val = None
    for chunk in pd.read_csv(path, usecols=[col], chunksize=chunksize):
        m = chunk[col].max()
        if max_val is None or m > max_val:
            max_val = m
    return max_val


def remove_duplicates(path, col="time"):
    df = pd.read_csv(path)

    dups = df[df[col].duplicated()]

    if len(dups)>0:
        df.drop_duplicates([col], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df.to_csv(path, index=None)

    else:
        print("No duplicates found")

    return

def convert_to_parquet(path):

    print("Reading file")
    df = pd.read_csv(path)

    print("Converting file")
    new_path = path.replace('csv', 'parquet')
    df.to_parquet(new_path, compression="zstd")

    print("Done")







