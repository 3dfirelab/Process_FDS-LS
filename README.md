# FDS-LS 2 GIS 
This code process outputs from FDS-Level set (FDS-LS) simualtion into data array format that can be open in GIS software. It is expecting simulation set using the `qgis2fds` pluging of `QGIS`.

# How to run:
To run the code:
```
$python fdsls2gis.py -p path2your-FDSLS-Simulation
```

for help see 
```
$pyhton fdsls2gis.py -h
usage: fdsls2gis.py [-h] -p PATH2FDSDIR [-slcPhiId SLCPHIID] [-bfMLRId BFMLRID] [-sF SKIPFRAME] [-tE TIMEEND]

extracts fire perimeters from slive file of FDS level set, and compute ROS. It also computes biomass burnt from
boundary file of MASS FLUX

options:
  -h, --help            show this help message and exit
  -p PATH2FDSDIR, --path2FDSdir PATH2FDSDIR
                        fds simulation directory path
  -slcPhiId SLCPHIID, --slcPhiId SLCPHIID
                        id number of the slc level set variable
  -bfMLRId BFMLRID, --bfMLRId BFMLRID
                        id number of the bndf MASS FLUX variable
  -sF SKIPFRAME, --skipFrame SKIPFRAME
                        number of frame to skip
  -tE TIMEEND, --timeEnd TIMEEND
                        timeEnd, to for example if the simulation crashes before the end

```


# Outputs:
The outputs are: 
- `BB`: Biomass Burnt, the amount of vegetation burnt (kg) as prescribed form by the rothermel-Albini model used in FDS-LS. To compute BB you need to add in the default `qgis2fds` configuration file a line to save `MASS FLUX` in the boundary files, 
```&BNDF QUANTITY='MASS FLUX' /``` 
- `arrivalTimeFront`: the map of time of arrival extracted from the level set variable (phi).
- `arrivalTime`: the map of arrival time interpolated.
- `ROS`: Rate of Spread in (m/s) computed from the interpolated arrival time map.
- `burntarea`:  the final level set variable, showing the burnt area.
- `terrainFDS`: the terrain imported in FDS-LS.

