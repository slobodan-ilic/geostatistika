"""Code for generating R² graph for OCO-2 SIF vs the CSIF product (Zhange et al).

Author: Slobodan Ilic
Email: slobodan.a.ilic@gmail.com
Date: 03-Oct-2022

This code is intended as an excercise in partially reproducing the results obtained by
Zhang et al (2018). The goal of the excercise is to reproduce the R² score reported in
the paper, for the years 2014 - 2017.

The code operates in the following way:
    1. For a given CSIF file, which covers a specific year, all the available OCO-2
       files are downloaded. The values are then matched by the CSIF grid values of
       latitude and longitude (OCO-2 values are trimmed to the nearest CSIF value for
       both lat and lon).
    2. After matching the OCO-2 soundings to CSIF grid, the OCO-2 data are examined for
       quality based on the cloud flag (has to be 0 - which means no clouds) and for the
       number of soundings (each grid cell has to contain > 5 clear soundings).
    3. The OCO-2 soundings acquired for a particular CSIF grid cell are then averaged.
    4. The matching CSIF and OCO-2 SIF values are used to calculate the R² score.
    5. The score is calculated using the OCO-2 values as originals, since they were
       the de-facto originals, while the graph is "inverted" in the sense that the X
       axis represents the forecasted (generated) CSIF values, while the original OCO-2
       values are represented on the Y-axis.
"""

import datetime as dt
from os import listdir
from os.path import isfile, join
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# --- The folder `OCO_DIR` needs to be populated with files from the following link: ---
# --- https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L2_Lite_SIF.10r/2015/ ---
# --- which contain the OCO-2 satelite recordings of the SIF at 757 nm wavelength. ---
# --- The files need to match the year defined in the CSIF file (e.g. 2015). ---
OCO_DIR = "./data/oco/"

# --- The CSIF file for which we want to perform the R² statistics, for the matching ---
# --- recorded OCO-2 measurements needs to be downloaded from: ---
# --- https://figshare.com/articles/dataset/CSIF/6387494 and needs to match the year ---
# --- (e.g. 2015) for which we want to perform matching with OCO-2 files.
CSIF_DIR = "./data/csif/"
FILE_NAME_CSIF = f"{CSIF_DIR}OCO2.SIF.all.daily.2015.nc"


def get_oco_file(dir_, oco_filename_prefix_with_date) -> Optional[str]:
    """return matching filename from the data directory, for a given date prefix."""
    oco_file_names = [f for f in listdir(dir_) if isfile(join(dir_, f))]
    for fn in oco_file_names:
        if oco_filename_prefix_with_date in fn:
            return join(dir_, fn)
    return None


def get_oco_day_of_year(filename) -> int:
    """Return int representation of the day of year, for current day OCO-2 file."""
    date_from_fn = filename.split("_")[2]  # date info extracxted from file name
    date_str = f"20{date_from_fn[:2]}-{date_from_fn[2:4]}-{date_from_fn[4:]}"
    day_of_year = dt.datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday
    return day_of_year


def get_oco_data(filename) -> tuple:
    """return tuple(sifs, lats, lons, doy, cloud_flags) from OCO-2 SIF file.

    All the data are fetched from the nc4 file content, except for the "day of year",
    which is fetched from the filename itself.
    """
    with h5py.File(filename, mode="r") as file:
        return (
            file["Daily_SIF_757nm"][:],
            file["Latitude"][:],
            file["Longitude"][:],
            get_oco_day_of_year(filename),  # get day of year from filename
            file["Cloud"]["cloud_flag_abp"][:],
        )


def get_csif_data(filename) -> tuple:
    """return tuple(sifs, doys, lats, lons) for CSIF product."""
    with h5py.File(filename, mode="r") as file:
        return file["all_daily_sif"][:], file["doy"][:], file["lat"][:], file["lon"][:]


def aggregate_oco_sif_and_csif_data(oco_data, csif_data):
    """return tuple(oco_sif, csif) of matching data by grid cells."""
    # --- Unpack data for processing ---
    ocos, oco_lats, oco_lons, oco_doy, cloud_flags = oco_data
    csifs, csif_doys, csif_lats, csif_lons = csif_data

    # --- Init empty result containers ---
    csifs_scatter = []
    ocos_scatter = []

    # --- Process (match and aggregate) the OCO-2 SIF and CSIF data ---
    min_lat = oco_lats.min()
    max_lat = oco_lats.max()
    min_lon = oco_lons.min()
    max_lon = oco_lons.max()

    clear_ind = cloud_flags == 0
    doy_csif = csifs[csif_doys == oco_doy][0]
    csif_lats = [  # filter out grid cells not occupied by OCO-2 latitutes
        (i, lat)
        for i, lat in enumerate(csif_lats)
        if min_lat - lat < 0.25 and lat - max_lat < 0.25
    ]
    csif_lons = [  # filter out grid cells not occupied by OCO-2 longitudes
        (i, lon)
        for i, lon in enumerate(csif_lons)
        if min_lon - lon < 0.25 and lon - max_lon < 0.25
    ]

    # --- Iterate over CSIF grid, potentially matching lat, lon to OCO-2 lat, lon ---
    for i, csif_lat in csif_lats:
        oco_lat_inds = np.abs(oco_lats - csif_lat) < 0.25
        if not oco_lat_inds.any():
            continue
        lat_csif = doy_csif[i]
        for j, csif_lon in csif_lons:
            csif = lat_csif[j]
            if np.isnan(csif):
                continue
            oco_lon_inds = np.abs(oco_lons - csif_lon) < 0.25
            ind = np.bitwise_and(oco_lat_inds, oco_lon_inds)
            ind = np.bitwise_and(ind, clear_ind)
            if not ind.any() or not ind.sum() >= 5:
                continue

            matching = ocos[ind]
            mean_oco_sif = matching.mean()

            csifs_scatter.append(csif)
            ocos_scatter.append(mean_oco_sif)

    return ocos_scatter, csifs_scatter


def process_data(filename_csif):
    """return tuple(osif, csif) by processing all OCO-2 files, for a given CSIF file."""
    csif_data = get_csif_data(filename_csif)
    doys = csif_data[1]
    yr = FILE_NAME_CSIF.split(".")[-2][-2:]
    osif, csif = [], []
    for doy in doys.astype("int"):
        date = dt.datetime(int(yr), 1, 1) + dt.timedelta(int(doy) - 1)
        date_str = date.strftime("%y%m%d")
        oco_prefix_with_date = f"oco2_LtSIF_{date_str}"
        filename_oco2 = get_oco_file(OCO_DIR, oco_prefix_with_date)
        if filename_oco2 is None:
            print(f"Couldn't fine file with prefix: {oco_prefix_with_date}")
            continue
        oco_data = get_oco_data(filename_oco2)
        osifs, csifs = aggregate_oco_sif_and_csif_data(oco_data, csif_data)
        print(f"\nfilename: {filename_oco2}")
        print(r2_score(osifs, csifs))
        osif.extend(osifs)
        csif.extend(csifs)
    return osif, csif


def draw_scatter(csifs, osifs, date_str):
    """draw scatter plot and save figure."""
    plt.plot(csifs, osifs, "o")
    plt.xlabel("Predicted CSIF from Yao Zhang's paper")
    plt.ylabel("OCO-2 MEAN SPATIAL SIF (averaged over CSIF grid)")
    plt.title(f"R² (for {date_str}): {r2_score(osifs, csifs)}")
    plt.plot([-0.5, 0, 1.5], [-0.5, 0, 1.5])  # straight line y = x
    fig = plt.gcf()

    pngfile = f"{date_str}.py.png"
    fig.savefig(pngfile)


if __name__ == "__main__":
    osif, csif = process_data(FILE_NAME_CSIF)
    date_str = FILE_NAME_CSIF.split(".")[-2]
    draw_scatter(csif, osif, date_str)
