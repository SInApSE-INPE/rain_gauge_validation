import os
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import glob
from datetime import datetime
from multiprocessing import Pool
from pyfortrace.utilities.utils import get_input_files

def extract_date_from_filename(filename, Pattern):
    """
    Extract the date from the filename using the provided pattern.
    """
    return datetime.strptime(os.path.basename(filename), Pattern)

def add_index_grid(table, Xgrid,Ygrid):
    """
    Add index_x and index_y columns to the table based on the grid coordinates.
    Based on position in the grid (index)
    """
    tree = cKDTree(np.column_stack((Xgrid.ravel(), Ygrid.ravel())))
    index_x=tree.query(np.column_stack((table['longitude'],table['latitude'])))[1]
    table['index_x'],table['index_y']=np.unravel_index(index_x, Xgrid.shape)
    return table

def create_spp_files_df(input_path,date_format):
    """
    Create a DataFrame with the SPP files and their corresponding timestamps.
    """
    table_files=pd.DataFrame([{'files':get_input_files(input_path)} ])#files list
    table_files['timestamp'] = table_files['files'].apply(lambda x: extract_date_from_filename(x,date_format))
    return table_files

def initialize_globals(name_list, read_function_gauge):
    global table_gauge, table_files_SPP, common_timestamps, output_path
    vector_Lon = name_list['longitude_SPP']
    vector_Lat = name_list['latitude_SPP']
    X_grid, Y_grid = np.meshgrid(vector_Lon, vector_Lat)
    df_gauge = read_function_gauge()
    table_gauge = add_index_grid(df_gauge, X_grid, Y_grid)
    table_files_SPP = create_spp_files_df(name_list['input_path'], name_list['timestamp_pattern'])
    common_timestamps = sorted(set(table_files_SPP['timestamp']).intersection(table_gauge['timestamp']))
    output_path = name_list['output_path']

def save_tables(Time):
    """
    Save data frame with SPP rain rate estimatives as SPP and previous columns from table_gauge.
    The tables are saved in parquet format separeted according to year, month, day, hour.
    """
    table_gauge_prov = table_gauge[table_gauge['timestamp'] == Time].copy()
    data_spp = read_function(table_files_SPP[table_files_SPP['timestamp'] == Time]['files'].values[0])
    table_gauge_prov['SPP'] = data_spp[table_gauge_prov['index_y'], table_gauge_prov['index_x']]
    save_parquet_name = f'{output_path}{Time.strftime("%Y%m%d%H%M")}.parquet'
    table_gauge_prov.to_parquet(save_parquet_name, index=False)

def create_tables_spp_gauge(name_list, read_function, read_function_gauge, parallel=True):
    """
    Main function to create tables with SPP and gauge values.
    """
    initialize_globals(name_list, read_function_gauge)
    n_workers = name_list['n_workers']
    if parallel:
        with Pool(n_workers) as pool:
            for _ in pool.imap_unordered(save_tables, common_timestamps):
                pass
    else:
        for Time in common_timestamps:
            save_tables((Time))


if __name__ == '__main__':  
    def read_function(path):  
        """  
        Same as track.py from pyfortrace  
        """  
        data = np.fromfile(path, dtype='float16').reshape(232, 291)  
        data = data.astype('float32')  
        return data  

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

    name_list = {}  # Changed from list to dict to support key assignment
    name_list['input_path'] = 'MatodroksoDoSuL/'  
    name_list['output_path'] = 'traditional_data/'  
    name_list['longitude_spp'] = np.round(np.append(np.arange(0, 180, 1), np.arange(-180, 0, 0.1)), 2)[2936:3227]   #1d array for longitude
    name_list['latitude_spp'] = np.round(np.arange(60, -60, -1), 2)[686:918]  #1d array for latitude
    name_list['timestamp_pattern'] = 'gsmap_mvk.%Y%m%d.%H%M.dat'  
    name_list['n_workers'] = 3  
    create_tables_spp_gauge(name_list, read_function, read_function_gauge)