## Naturinngrep

Python code for applying convolutional neural networks (CNN) to Sentinel-2 data using python and PyTorch.

The code provides a simple framework for (1) converting Sentinel-2 data to an efficient format for deep learning, (2) training CNNs, and (3) applying trained networks to new data. We provide a simple example, classifiying road pixels in Sentinel-2 data. The code can easily be modified for other solving other problems by adding your own training data in the prepare_data.py-code. 

### Setup:
- Make sure GDAL is installed. 
- Download code and setup python:
```console
git clone https://github.com/NorskRegnesentral/naturinngrep.git
cd naturinngrep
virtualenv -p python3 env
source env/bin/activate
pip3 install -r REQUIREMENTS.txt
``` 

- You need to manually download the SAFE-files listed in s2_files*.txt.  Also, rename the paths in these files to point to the location where the SAFE-files are downloaded. If you have your own files, these should be listed similarly in a text file. 

### Main files:
- **prepare_data.py** - Move data from SAFE-format to a local storage with an efficient np.memmap-format suitable for training CNNs. Both training-data and test-data should be prepared. The file contains the processing configuration. Change the config paths to match your system.
- **train.py** - Train the CNN. In the script, the training and validation are specified. The trainied network are saved in the directory saved_models.
- **predict.py** - Apply the trained network to new data specified by the tiles list, and store as results as GEOtiff in the results directory. The

### Credit:
- UNet implementation by Jackson Huang: https://github.com/jaxony/unet-pytorch 
- SAR geocoding library by the Norwegian Computing Center

### Authors:
Anders U. Waldeland (anders@nr.no), Arnt-Børre Salberg (salberg@nr.no), Øivind Due Trier (trier@nr.no) 

Norwegian Computing Center (https://www.nr.no/)


 
