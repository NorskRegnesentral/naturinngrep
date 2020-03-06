import json
import os
import shutil
from sentinel_dataset._utils import parse_eodata_folder_name
from sentinel_data_preparation.data_preparation import DataPreparation

# Input data are level-1
# Prepare data for Sentinel-2 processing (COGSAT files)
config = {
    "sensor": "sentinel-2",
    "eocloud_path_list": "s2_files.txt", #"s2_files.txt",
    "rad_cor": "toa",
    "resolution": "10m",
    "bands": ["b02", "b03", "b04", "b08",  # 10m
              "b05", "b06", "b07", "b8a", "b11", "b12",  # 20m
              "b01", "b09","b10"],  # 60m
    "process_clouds": "yes",
    "process_target_data": "yes", #"yes",
    "target_dir": "/nr/samba/jodata2/pro/naturinngrep/usr/arnt/targets",
    # "target_dir": "/nr/samba/jodata2/pro/naturinngrep/s2_granules/test4",
    "target_postfix": "mask",
    "target_prefix": "L1C",
    # "target_basename": "mask", #TODO remove this
    "target_names": ['lbl_roads'],
    "n_threads": 4,
    "outdir": "/nr/samba/jodata2/pro/naturinngrep/usr/arnt/mmap",
    'tile_ids':[
        'T33WXT_20190718T113853',
    ]
}

s2_dp = DataPreparation(config)
s2_dp.run_all()
