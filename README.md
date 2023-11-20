# what it does
Simple code to extract fire perimeter from FDS Level set fire front simulation. perimeters are save as shape files that can be imported in QGIS. 

# available data
Topography and 'LEVEL SET VALUE' slice files of a coarse simulation setup with the [qgis2fds](https://github.com/firetools/qgis2fds) plugin of QGIS are available in `testData/LS4/`

# to run the extraction
```
python ./src/fds2frontShp.py -p ./testData/LS4/ -sF 50
```
output shape file is saved in the simulation directory. 

For input parameters definition run
```
python ./src/fds2frontShp.py -h
usage: fds2frontShp.py [-h] -p PATH2FDSDIR [-slcPhiId SLCPHIID]
                       [-sF SKIPFRAME]

extract fire perimeters from FDS level set simulation

optional arguments:
  -h, --help            show this help message and exit
  -p PATH2FDSDIR, --path2FDSdir PATH2FDSDIR
                        fds simulation directory path
  -slcPhiId SLCPHIID, --slcPhiId SLCPHIID
                        id number of the slc level set variable
  -sF SKIPFRAME, --skipFrame SKIPFRAME
                        number of frame to skip
```

# requirements
* cv2
* geopandas 
* scipy
* shapely
