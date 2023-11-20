# what it does
Simple code to extract fire perimeter from FDS Level set fire front simulation. perimeters are save as shape files that can be imported in QGIS. 

# available data
Topography and 'LEVEL SET VALUE' slice files of a coarse simulation setup with the [qgis2fds](https://github.com/firetools/qgis2fds) plugin of QGIS are available in `testData/LS4/`

# to run the extraction
```
python ./src/fds2frontShp.py -p ./testData/LS4/ -sF 50
```
output files of the shape file are saved in the simulation directory.

# requirements
* cv2
* geopandas 
* scipy
* shapely
