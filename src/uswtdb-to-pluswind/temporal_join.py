import pandas as pd
import geopandas as gpd
import shapely
import os
from pathlib import Path

df_uswtdb = pd.read_csv("uswtdb_v5.csv")
geoms = [shapely.Point(xy) for xy in zip(df_uswtdb["xlong"], df_uswtdb["ylat"])]
uswtdb = gpd.GeoDataFrame(df_uswtdb, geometry=geoms)
uswtdb.set_crs(epsg=4326, inplace=True)

df_pluswind = pd.read_csv("pluswind-plant-list-20221213.csv")
geoms = [shapely.Point(xy) for xy in zip(df_pluswind["lon"], df_pluswind["lat"])]
pluswind = gpd.GeoDataFrame(df_pluswind, geometry=geoms)
pluswind.set_crs(epsg=4326, inplace=True)

joined = pluswind[["EIA_ID", "geometry"]].sjoin_nearest(
    uswtdb, how="inner", distance_col="proximity", max_distance=0.015
)
joined.rename(columns={"EIA_ID": "pluswind_eia_id"}, inplace=True)

print(f"Spatial join complete: {len(joined)} turbine locations matched")

weather_data_dir = "data"
all_temporal_records = []

print("Building EIA_ID to file mapping...")
eia_to_file = {}
for weather_file in Path(weather_data_dir).glob("plant.c0.2021.*.csv"):
    # Extract EIA_ID from filename (last number before .csv)
    # Example: plant.c0.2021.central-spp.7771.csv -> 7771
    eia_id = int(weather_file.stem.split('.')[-1])
    eia_to_file[eia_id] = weather_file

print(f"Found {len(eia_to_file)} weather data files")

output_file = "turbine_weather_temporal_dataset.csv"
batch_size = 100
write_header = True
total_records_written = 0

for idx, row in joined.iterrows():
    pluswind_eia_id = row["pluswind_eia_id"]
    
    if pluswind_eia_id not in eia_to_file:
        print(f"Warning: Weather file not found for EIA_ID {pluswind_eia_id}")
        continue
    
    weather_file = eia_to_file[pluswind_eia_id]
    
    try:
        weather_df = pd.read_csv(weather_file)
        
        weather_df["timestamp"] = pd.to_datetime(weather_df["gmt"])
        
        for _, weather_row in weather_df.iterrows():
            # Combine turbine metadata with weather data
            temporal_record = {
                "pluswind_eia_id": row["pluswind_eia_id"],
                "index_right": row["index_right"],
                "case_id": row["case_id"],
                "faa_ors": row["faa_ors"],
                "faa_asn": row["faa_asn"],
                "usgs_pr_id": row["usgs_pr_id"],
                "eia_id": row["eia_id"],
                "t_state": row["t_state"],
                "t_county": row["t_county"],
                "t_fips": row["t_fips"],
                "p_name": row["p_name"],
                "p_year": row["p_year"],
                "p_tnum": row["p_tnum"],
                "p_cap": row["p_cap"],
                "t_manu": row["t_manu"],
                "t_model": row["t_model"],
                "t_cap": row["t_cap"],
                "t_hh": row["t_hh"],
                "t_rd": row["t_rd"],
                "t_rsa": row["t_rsa"],
                "t_ttlh": row["t_ttlh"],
                "t_retrofit": row["t_retrofit"],
                "t_retro_yr": row["t_retro_yr"],
                "t_offshore": row["t_offshore"],
                "t_conf_atr": row["t_conf_atr"],
                "t_conf_loc": row["t_conf_loc"],
                "t_img_date": row["t_img_date"],
                "t_img_src": row["t_img_src"],
                "xlong": row["xlong"],
                "ylat": row["ylat"],
                "proximity": row["proximity"],
                
                "timestamp": weather_row["timestamp"],
                "wind_speed": weather_row["MERRA2 density-corrected wind speed (m/s)"],
                "capacity_factor": weather_row["MERRA2 CF (density and loss adjusted)"],
            }
            
            all_temporal_records.append(temporal_record)
    
    except Exception as e:
        print(f"Error processing EIA_ID {pluswind_eia_id}: {e}")
        continue
    
    # Write to CSV every batch_size locations to avoid memory issues
    if (idx + 1) % batch_size == 0:
        batch_df = pd.DataFrame(all_temporal_records)
        batch_df.to_csv(output_file, mode='a', header=write_header, index=False)
        total_records_written += len(all_temporal_records)
        print(f"Processed {idx + 1}/{len(joined)} locations, {total_records_written} total records written")
        
        all_temporal_records = []
        write_header = False

if all_temporal_records:
    batch_df = pd.DataFrame(all_temporal_records)
    batch_df.to_csv(output_file, mode='a', header=write_header, index=False)
    total_records_written += len(all_temporal_records)
    print(f"Processed {len(joined)}/{len(joined)} locations, {total_records_written} total records written")

print(f"\nDataset saved to {output_file}")
print(f"Total records written: {total_records_written}")

print("\nReading sample from saved file:")
sample = pd.read_csv(output_file, nrows=5)
print("\nFinal columns:")
print(sample.columns.tolist())
print("\nSample of final dataset:")
print(sample)