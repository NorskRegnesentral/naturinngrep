import torch
import numpy as np
import os

from dlt.basic.predict_on_large_tile import apply_net_to_large_data
from dlt.basic.unet import UNet
from sentinel_dataset import Dataset

#Configuration
network_path = 'saved_models/saved_model.pt'
target_is_classes = True #Put to false for regression problems
n_output_channels = 2
model_name = network_path.split('/')[-1].replace('.pt','')

data_bands = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
pred_win_size = [1024, 1024]
window_overlap = [50, 50]

#Load model with weights
net = UNet(n_output_channels, len(data_bands), use_bn=True)
weights = torch.load(network_path, map_location=lambda storage, loc: storage)
net.load_state_dict(weights)

#Move model to GPU to enable GPU-computing
net.cuda()

#Put model in evaluation mode
net.eval()

tiles = [
    'T34WFD_20150730T103016',
    'T34WFD_20150805T105026',
    'T34WFD_20160806T104026',
    'T34WFD_20160819T105029',
    'T34WFD_20170701T102023',
    'T34WFD_20170930T104019',
    'T34WFD_20180729T143053',
    'T34WFD_20180910T161121',
]
for test_tile in tiles:
    #Load test-tile
    dataset = Dataset('/home/salberg/naturinngrep/mmap/test4/' + test_tile, band_identifiers=data_bands ) #T32VNR
    # Loop through tiles
    for tile in dataset.tiles:
        print('Predicting for tile', tile.tile_id, tile.file_name)

        #Get data
        data = tile.get_data(data_bands)
        data = [np.expand_dims(d,-1) for d in data]
        data = np.concatenate(data,-1)

        #Run through network
        output = apply_net_to_large_data(data, net, pred_win_size, window_overlap, apply_classifier=target_is_classes)

        #Save output as GeoTiff
        tile.export_prediction_to_tif("results/" + tile.file_name +'_' + model_name +'.tif', output)
