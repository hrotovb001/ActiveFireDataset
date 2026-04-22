from dotenv import load_dotenv
import os
import re
import warnings
from datetime import datetime
import xarray as xr
import odc.geo.xr
import numpy as np
import xesmf as xe
from pyproj import Proj
import pandas as pd
import matplotlib.pyplot as plt
import pickle

warnings.simplefilter(action="ignore")

load_dotenv()


def file_finder(item, dt):
    print(f"Finding file for item: {item}, datetime: {dt.isoformat()}")
    
    match item:
        case "goes-18" | "goes-19":
            folder = os.getenv("GOES")
            if folder == None:
                raise ValueError("Missing folder for GOES in .env")
            year = dt.year
            doy = dt.timetuple().tm_yday
            hour = dt.hour
            folder = f"{folder}/{year}/{doy:03d}/{hour:02d}"
            for file in os.listdir(folder):
                match = re.search(r"G(\d+)_s(\d+)_e(\d+)", file)
                sat, start_raw, end_raw = match.groups()
                start = datetime.strptime(start_raw, "%Y%j%H%M%S%f")
                end = datetime.strptime(end_raw, "%Y%j%H%M%S%f")
                if f'goes-{sat}' == item and start <= dt <= end:
                    return folder + "/" + file
            raise ValueError("GOES file not found")
        case "elv":
            folder = os.getenv("ELV")
            if folder == None:
                raise ValueError("Missing folder for ELV in .env")
            return folder.replace(" ", "") + "/MERIT_DEM.nc"
        case "ast":
            folder = os.getenv("AST")
            if folder == None:
                raise ValueError("Missing folder for AST in .env")
            return folder.replace(" ", "") + "/VIIRS_AST_2024.nc"
        case "fh":
            folder = os.getenv("FH")
            if folder == None:
                raise ValueError("Missing folder for FH in .env")
            return folder.replace(" ", "") + "/GLAD.nc"
        case "vhi":
            folder = os.getenv("VHI")
            if folder == None:
                raise ValueError("Missing folder for VHI in .env")
            week = dt.isocalendar().week
            return (
                folder.replace(" ", "")
                + "/npp/VHP.G04.C07.npp.P2025"
                + ("%03d" % week)
                + ".VH.nc",
                folder.replace(" ", "")
                + "/j01/VHP.G04.C07.j01.P2025"
                + ("%03d" % week)
                + ".VH.nc",
            )
        case "t2m" | "sh2" | "wd" | "ws" | "prate":
            folder = os.getenv("HRRR")
            return (
                folder.replace(" ", "")
                + "/"
                + dt.strftime("%Y%m%d")
                + "/hrrr.t"
                + ("%02d" % dt.hour)
                + "z.wrfsfcf00.grib2"
            )


def save_image_grid(array, filename='output.png', cmap='gray', vmin=None, vmax=None):
    n_images = array.shape[0]
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i in range(n_images):
        ax = axes[i]
        im = ax.imshow(array[i], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.axis('off')
        ax.set_title(f'Slice {i+1}', fontsize=8)

    for j in range(n_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def wind_conversion(
    LAT, U, V
):  # from HRRR FAQ (https://rapidrefresh.noaa.gov/faq/HRRR.faq.html)
    rotcon_p = 0.622515
    lat_tan_p = 38.5

    angle = rotcon_p * (LAT - lat_tan_p) * 0.017453  # LAMBERT CONFORMAL PROJECTION
    sinx2 = np.sin(angle)
    cosx2 = np.cos(angle)

    U_new = cosx2 * U + sinx2 * V
    V_new = (-1) * sinx2 * U + cosx2 * V
    return U_new, V_new


def wind_direction(u, v):
    angle_radians = np.arctan2(v, u)
    angle_degrees = np.rad2deg(angle_radians)
    
    met_dir = (270. - angle_degrees) % 360.
    
    is_calm = (u == 0) & (v == 0)
    return xr.where(is_calm, 0.0, met_dir)


def convert_goes_to_lat_lon(ds):
    dat = ds.metpy.parse_cf('Power')
    crs = dat.metpy.cartopy_crs
    
    x, y = ds.x.values, ds.y.values
    
    dx = (x[1] - x[0])
    dy = (y[1] - y[0])
    xb = np.linspace(x[0] - dx/2, x[-1] + dx/2, len(x) + 1)
    yb = np.linspace(y[0] - dy/2, y[-1] + dy/2, len(y) + 1)
    
    Xb, Yb = np.meshgrid(xb, yb)
    
    h = ds.goes_imager_projection.perspective_point_height
    p = Proj(crs.to_dict()) 
    lon_b, lat_b = p(Xb * h, Yb * h, inverse=True)
    
    X, Y = np.meshgrid(x, y)
    lon, lat = p(X * h, Y * h, inverse=True)

    ds = ds.assign_coords(
        lon=(('y', 'x'), lon),
        lat=(('y', 'x'), lat),
        lon_b=(('y_b', 'x_b'), lon_b),
        lat_b=(('y_b', 'x_b'), lat_b)
    )
    return ds


def build_goes_regridder(file_path):
    ds = xr.open_dataset(file_path, chunks="auto")
    ds = convert_goes_to_lat_lon(ds)
    ds['lon'] = xr.where(ds['lon'] < 0, ds['lon'] + 360, ds['lon'])
    lats = xr.where(np.isfinite(ds.lat), ds.lat, np.nan)
    lons = xr.where(np.isfinite(ds.lon), ds.lon, np.nan)
    lat_min = float(lats.min())
    lat_max = float(lats.max())
    lon_min = max(180, float(lons.min()))
    lon_max = float(lons.max())
    new_lat = np.linspace(lat_min, lat_max, 3000)
    new_lon = np.linspace(lon_min, lon_max, 5000)
    target_ds = xr.Dataset({
        "lat": new_lat, 
        "lon": new_lon
    })
    regridder = xe.Regridder(ds, target_ds, 'conservative_normed', ignore_degenerate=True)
    ds.close()
    return regridder


# initialize hrrr regridder
print("Initializing HRRR regridder")
if os.path.exists("hrrr_regridder.pkl"):
    with open("hrrr_regridder.pkl", "rb") as f:
        hrrr_regridder = pickle.load(f)
else:
    file_path = file_finder("ws", datetime(2025, 1, 1, 0, 0, 0))
    ds = xr.open_dataset(
        file_path,
        engine='cfgrib',
        filter_by_keys={'typeOfLevel': 'heightAboveGround', "level": 10},
        chunks="auto",
    )
    lat_min = float(ds.latitude.min())
    lat_max = float(ds.latitude.max())
    lon_min = float(ds.longitude.min())
    lon_max = float(ds.longitude.max())
    new_lat = np.linspace(lat_min, lat_max, 1100)
    new_lon = np.linspace(lon_min, lon_max, 2400)
    target_ds = xr.Dataset({
        "lat": new_lat, 
        "lon": new_lon
    })
    hrrr_regridder = xe.Regridder(ds, target_ds, 'bilinear')
    ds.close()
    with open("hrrr_regridder.pkl", "wb") as f:
        pickle.dump(hrrr_regridder, f)

# initialize goes regridders
print("Initializing GOES-18 regridder")
if os.path.exists("goes_18_regridder.pkl"):
    with open("goes_18_regridder.pkl", "rb") as f:
        goes_18_regridder = pickle.load(f)
else:
    file_path = file_finder("goes-18", datetime(2025, 1, 1, 8, 27, 0))
    goes_18_regridder = build_goes_regridder(file_path)
    with open("goes_18_regridder.pkl", "wb") as f:
        pickle.dump(goes_18_regridder, f)

print("Initializing GOES-19-1 regridder")
if os.path.exists("goes_19_1_regridder.pkl"):
    with open("goes_19_1_regridder.pkl", "rb") as f:
        goes_19_1_regridder = pickle.load(f)
else:
    file_path = file_finder("goes-19", datetime(2025, 1, 1, 6, 47, 0))
    goes_19_1_regridder = build_goes_regridder(file_path)
    with open("goes_19_1_regridder.pkl", "wb") as f:
        pickle.dump(goes_19_1_regridder, f)

print("Initializing GOES-19-2 regridder")
if os.path.exists("goes_19_2_regridder.pkl"):
    with open("goes_19_2_regridder.pkl", "rb") as f:
        goes_19_2_regridder = pickle.load(f)
else:
    file_path = file_finder("goes-19", datetime(2025, 4, 10, 5, 47, 0))
    goes_19_2_regridder = build_goes_regridder(file_path)
    with open("goes_19_2_regridder.pkl", "wb") as f:
        pickle.dump(goes_19_2_regridder, f)


def process_timestep(dt, target_ds):
    print("Starting input processing for timestep: " + dt.isoformat())

    output = np.empty((12, target_ds.lat.shape[0], target_ds.lon.shape[0]))
    
    target_geobox = target_ds.odc.geobox

    # goes
    if target_ds.lon.max() < 250:
        file_path = file_finder("goes-18", dt)
        goes_regridder = goes_18_regridder
    elif dt < datetime(2025, 4, 7, 15, 0, 0):
        file_path = file_finder("goes-19", dt)
        goes_regridder = goes_19_1_regridder
    else:
        file_path = file_finder("goes-19", dt)
        goes_regridder = goes_19_2_regridder
    ds = xr.open_dataset(file_path, chunks="auto")
    total = ds["Power"].sum().values
    ds = convert_goes_to_lat_lon(ds)
    da = ds["Power"]
    da['lon'] = xr.where(da['lon'] < 0, da['lon'] + 360, da['lon'])
    da = da.fillna(0)
    da = goes_regridder(da)
    da = da.odc.assign_crs("EPSG:4326")
    da = da.odc.reproject(target_geobox, resampling="sum")
    da = da.fillna(0)

    if da.max().values == 0:
        raise ValueError("No GOES data for timestep: " + dt.isoformat())

    output[0] = da
    ds.close()

    # elv
    file_path = file_finder("elv", dt)
    ds = xr.open_dataset(file_path, chunks="auto")
    ds['lon'] = xr.where(ds['lon'] < 0, ds['lon'] + 360, ds['lon'])
    ds = ds.odc.assign_crs("EPSG:4326")
    da = ds["Band1"]
    da = da.odc.reproject(target_geobox, resampling="bilinear")
    da = da.where(da >= 0, 0)

    output[1] = da
    ds.close()

    # ast
    file_path = file_finder("ast", dt)
    ds = xr.open_dataset(file_path, chunks="auto")
    ds['lon'] = xr.where(ds['lon'] < 0, ds['lon'] + 360, ds['lon'])
    ds = ds.odc.assign_crs("EPSG:4326")
    da = ds["surface_type"]
    da = da.odc.reproject(target_geobox, resampling="mode")

    output[2] = da
    ds.close()

    # doy
    da = xr.DataArray(
        data=np.full((target_ds.lon.shape[0], target_ds.lat.shape[0]), dt.timetuple().tm_yday),
        dims=('lon', 'lat'),
        coords={'lon': target_ds.lon, 'lat': target_ds.lat}
    )

    output[3] = da

    # hour
    ds = xr.open_dataset("./fix/timezones_voronoi_1x1.nc")
    ds = ds.squeeze(dim="time")
    ds = ds.odc.assign_crs("EPSG:4326")
    da = ds["UTC_OFFSET"]
    da = da.astype("float64")
    da = da.odc.reproject(target_geobox, resampling="nearest")
    da = da.astype("timedelta64[ns]")
    da = da / np.timedelta64(1, 'h') + dt.hour
    da = xr.where(da < 0, da + 24, da)
    da = xr.where(da >= 24, da - 24, da)

    output[4] = da
    ds.close()

    # fh
    file_path = file_finder("fh", dt)
    ds = xr.open_dataset(file_path, chunks="auto")
    ds['lon'] = xr.where(ds['lon'] < 0, ds['lon'] + 360, ds['lon'])
    da = ds["Band1"]
    da = da.odc.assign_crs("EPSG:4326")
    da = da.odc.reproject(target_geobox, resampling="bilinear")
    da = da.where(da >= 0)
    da = da.fillna(0)

    output[5] = da
    ds.close()

    # vhi
    file_path_npp, file_path_j01 = file_finder("vhi", dt)

    ds_npp = xr.open_dataset(file_path_npp, chunks="auto")
    ds_npp = ds_npp.assign_coords(
        HEIGHT=ds_npp.latitude,
        WIDTH=ds_npp.longitude
    ).rename(HEIGHT="lat", WIDTH="lon")
    ds_npp['lon'] = xr.where(ds_npp['lon'] < 0, ds_npp['lon'] + 360, ds_npp['lon'])
    ds_npp = ds_npp.odc.assign_crs("EPSG:4326")
    ds_npp = ds_npp[["VCI", "TCI"]]
    ds_npp = ds_npp.odc.reproject(target_geobox, resampling="bilinear")

    ds_j01 = xr.open_dataset(file_path_j01, chunks="auto")
    ds_j01 = ds_j01.assign_coords(
        HEIGHT=ds_j01.latitude,
        WIDTH=ds_j01.longitude
    ).rename(HEIGHT="lat", WIDTH="lon")
    ds_j01['lon'] = xr.where(ds_j01['lon'] < 0, ds_j01['lon'] + 360, ds_j01['lon'])
    ds_j01 = ds_j01.odc.assign_crs("EPSG:4326")
    ds_j01 = ds_j01[["VCI", "TCI"]]
    ds_j01 = ds_j01.odc.reproject(target_geobox, resampling="bilinear")

    ds_npp = ds_npp.where(ds_npp != 999)
    ds_j01 = ds_j01.where(ds_j01 != 999)

    ds = xr.concat([ds_npp, ds_j01], dim="source").mean("source", skipna=True)
    ds["VCI"] = ds["VCI"] / 100
    ds["TCI"] = ds["TCI"] / 100
    da = 0.3 * ds["VCI"] + 0.7 * ds["TCI"]
    da = da.fillna(0.5)

    output[6] = da
    ds.close()

    # t2m, sh2
    file_path = file_finder("t2m", dt)
    ds = xr.open_dataset(
        file_path,
        engine="cfgrib",
        filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2},
        chunks="auto"
    )
    ds = ds[["t2m", "sh2"]]
    ds = hrrr_regridder(ds)
    ds = ds.odc.assign_crs("EPSG:4326")
    ds = ds.odc.reproject(target_geobox, resampling="bilinear")
    ds = ds.where(ds >= 0)

    output[7] = ds["t2m"]
    output[8] = ds["sh2"]

    ds.close()

    # prate
    file_path = file_finder("prate", dt)
    ds = xr.open_dataset(
        file_path,
        engine="cfgrib",
        filter_by_keys={"typeOfLevel": "surface", "stepType": "instant"},
        chunks="auto"
    )
    da = ds["prate"]
    da = hrrr_regridder(da)
    da = da.odc.assign_crs("EPSG:4326")
    da = da.odc.reproject(target_geobox, resampling="bilinear")
    da = da.where(da >= 0)
    da *= 3600

    output[9] = da

    ds.close()

    # ws, wd
    file_path = file_finder("ws", dt)
    ds = xr.open_dataset(
        file_path,
        engine="cfgrib",
        filter_by_keys={'typeOfLevel': 'heightAboveGround', "level": 10},
        chunks="auto"
    )
    ds = ds[["u10", "v10"]]
    ds = hrrr_regridder(ds)
    ds = ds.odc.assign_crs("EPSG:4326")
    ds = ds.odc.reproject(target_geobox, resampling="bilinear")

    ds["u10"], ds["v10"] = wind_conversion(ds["latitude"], ds["u10"], ds["v10"])

    ws = np.sqrt(ds["u10"]**2 + ds["v10"]**2)
    wd = wind_direction(ds["u10"], ds["v10"])

    output[10] = ws
    output[11] = wd

    ds.close()

    save_image_grid(output, cmap='viridis')

    return output

if __name__ == "__main__":
    new_lat = np.linspace(32.75004, 34.823302, 100)
    new_lon = np.linspace(241.048279, 242.353626, 100)
    target_ds = xr.Dataset({
        "lat": new_lat, 
        "lon": new_lon
    })
    target_ds = target_ds.odc.assign_crs("EPSG:4326")

    process_timestep(datetime(2025, 1, 21, 7, 12, 0), target_ds)

