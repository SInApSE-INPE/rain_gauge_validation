import os
import pandas as pd
import csv
import xarray as xr
from scipy.spatial import cKDTree
import glob
from datetime import datetime
import numpy as np
import igdm
from multiprocessing import Pool

def extract_date_from_filename(filename, Pattern):
    """
    Extract the date from the filename using the provided pattern.
    """
    return datetime.strptime(os.path.basename(filename), Pattern)

def add_index_grid(table, Xgrid, Ygrid):
    """
    Add index_x and index_y columns to the table based on the grid coordinates.
    Based on position in the grid (index)
    """
    tree = cKDTree(np.column_stack([Xgrid.ravel(), Ygrid.ravel()]))
    index_x = tree.query(np.column_stack((table['longitude'], table['latitude'])))[1]
    table['index_x'], table['index_y'] = np.unravel_index(index_x, Xgrid.shape)
    return table

def initialize_globals(name_list, read_function_gauge):
    global table_gauge, output_path
    vector_Lon = name_list['longitude_sp']
    vector_Lat = name_list['latitude_sp']
    X_grid, Y_grid = np.meshgrid(vector_Lon, vector_Lat)
    df_gauge = read_function_gauge()
    table_gauge = add_index_grid(df_gauge, X_grid, Y_grid)
    output_path = name_list['output_path']


def save_gaugevalues(row):
    """
    Loads track data from PyForTraCC tracking tables (Parquet format)
    For each track point (array_values), finds matching rain gauge locations
    Saves results in Parquet format organized by:
     * UID (11-digit zero-padded)
     *Timestamp (YYYYMMDD_HHMM format)
    """
    array_x = row['array_x']
    array_y = row['array_y']
    array_values = row['array_values']
    uid = row['uid'] * np.ones(len(array_x))
    table_xy = pd.DataFrame({'index_x': array_x, 'index_y': array_y, 'array_values': array_values, 'uid': uid}).reset_index()
    table_gauge_aux = table_gauge[table_gauge['timestamp'] == row['timestamp']]
    intersection = table_xy.merge(table_gauge_aux, on=['index_x', 'index_y'], how='inner')
    if len(intersection) > 0:
        savename = f"{output_path}/{str(int(row['uid'])).zfill(11)}.{row['timestamp'].strftime('%Y%m%d')}.parquet"
        intersection.to_parquet(savename)

def get_gaugevalues(trk_file):
    track = pd.read_parquet(trk_file)
    track.apply(lambda row: save_gaugevalues(row), axis=1)


def add_gaugevalues(name_list, read_function_gauge, parallel=True):
    """
    Main function to create tables with SPP tracking and gauge values.
    """
    initialize_globals(name_list, read_function_gauge)
    trk_files = sorted(glob.glob(f"{name_list['track_path']}/*.parquet"))
    n_workers = name_list['n_workers']
    if parallel:
        with Pool(n_workers) as pool:
            for _ in pool.imap_unordered(get_gaugevalues, trk_files):
                pass
    else:
        for trk_file in trk_files:
            get_gaugevalues(trk_file)

if __name__ == '__main__':
    def read_function_gauge():
        """
        Any function that returns a pandas dataframe with columns ['timestamp', 'latitude', 'longitude']
        """ 
        table_gauge = pd.DataFrame()
        files_by_station = sorted(glob.glob('/media/milton/Data/Data/Tables/INMET/*.parquet'))
        for station_file in files_by_station:
            station = pd.read_parquet(station_file)
            table_gauge = pd.concat([table_gauge, station], ignore_index=True)
        return table_gauge

    name_list = {}
    name_list['track_path'] = '/media/milton/Data/Data/Tracks/track_gsmap_mvk/track/trackingtable/'
    name_list['output_path'] = '/media/milton/Data/Data/traditional_validation/'
    name_list['longitude_spp'] = np.round(np.append(np.arange(0, 180, 0.1), np.arange(-180, 0, 0.1)), 2)[2936:3227]
    name_list['latitude_spp'] = np.round(np.arange(60, -60, -0.1), 2)[686:918]
    name_list['n_workers'] = 3
    add_gaugevalues(name_list, read_function_gauge)