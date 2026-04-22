from dotenv import load_dotenv
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.cluster import DBSCAN
import odc.geo.xr
import math
import os
import gc

from process_timestep import process_timestep

load_dotenv()

res = 0.003
eps_deg = 0.03
time_scale = 0.003

file = os.getenv('FIRMS')
df = pd.read_csv(file)

df['acq_time_str'] = df['acq_time'].astype(str).str.zfill(4)
df['datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time_str'], 
                                format='%Y-%m-%d %H%M')

df = df.sort_values('datetime')

t0 = df['datetime'].min()
df['time_min'] = (df['datetime'] - t0).dt.total_seconds() / 60.0

X = df[['latitude', 'longitude']].to_numpy()
T = (df['time_min'].to_numpy() * time_scale).reshape(-1, 1)

X_st = np.hstack([X, T])

clustering = DBSCAN(eps=eps_deg, min_samples=3).fit(X_st)
df['cluster'] = clustering.labels_

for cluster_id, cluster in df.groupby('cluster', observed=False):
    if cluster_id == -1:
        continue

    lat_center = cluster['latitude'].mean()
    lon_center = cluster['longitude'].mean()

    half_side_km = 7.5
    half_side_deg_lat = half_side_km / 111.32
    half_side_deg_lon = half_side_km / (111.32 * math.cos(math.radians(lat_center)))

    lon_min = lon_center - half_side_deg_lon
    lon_max = lon_center + half_side_deg_lon
    lat_min = lat_center - half_side_deg_lat
    lat_max = lat_center + half_side_deg_lat

    lon_bins = np.linspace(lon_min, lon_max, 46)
    lat_bins = np.linspace(lat_min, lat_max, 46)

    lon_cell = pd.cut(cluster['longitude'], bins=lon_bins,
                      labels=lon_bins[:-1], include_lowest=True)
    lat_cell = pd.cut(cluster['latitude'], bins=lat_bins,
                      labels=lat_bins[:-1], include_lowest=True)

    grouped_grid = cluster.groupby([lat_cell, lon_cell], observed=False)['frp'].sum()
    grid_2d = grouped_grid.unstack(fill_value=0)

    print(pd.to_numeric(grid_2d.index.values))
    print(pd.to_numeric(grid_2d.columns.values))

    da = xr.DataArray(
        grid_2d.values,
        dims=('lat', 'lon'),
        coords={
            'lat': pd.to_numeric(grid_2d.index.values),
            'lon': pd.to_numeric(grid_2d.columns.values)
        }
    )
    da['lon'] = xr.where(da['lon'] < 0, da['lon'] + 360, da['lon'])
    da = da.odc.assign_crs("EPSG:4326")

    try:
        time = cluster['datetime'].min() - pd.Timedelta(minutes=10)
        output = process_timestep(time, da)
        np.save(f'/mnt/e/Dataset/tmp/{time.isoformat()}_{cluster_id}_input.npy', output)
        np.save(f'/mnt/e/Dataset/tmp/{time.isoformat()}_{cluster_id}_output.npy', da.values)
        del output
    except Exception as e:
        print(e)

    del da
    gc.collect()

