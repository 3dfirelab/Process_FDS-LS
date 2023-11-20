import numpy as np 
import slread
import importlib 
import struct 
import pdb 
from scipy import interpolate
import matplotlib.pyplot as plt
import sys 
from shapely.geometry import polygon
import cv2 
import geopandas as gpd 
import argparse
import glob 


def getTerrain(fname):
    f = open(fname,'rb')

    #zmin
    struct.unpack('i',f.read(4))
    zmin = struct.unpack('f',f.read(4))
    struct.unpack('i',f.read(4))

    #nx, ny
    struct.unpack('i',f.read(4))
    nx = struct.unpack('i',f.read(4))[0]
    ny = struct.unpack('i',f.read(4))[0]
    struct.unpack('i',f.read(4))

    #x
    lenx = struct.unpack('i',f.read(4))[0]
    x = np.array(struct.unpack('{:d}f'.format(lenx//4),f.read(lenx)))
    struct.unpack('i',f.read(4))

    #y
    leny=struct.unpack('i',f.read(4))[0]
    y=np.array(struct.unpack('{:d}f'.format(leny//4),f.read(leny)))
    struct.unpack('i',f.read(4))

    #z
    struct.unpack('i',f.read(4))
    zz = np.array(struct.unpack('304f',f.read(1216)))
    struct.unpack('i',f.read(4))

    f.close()
    

    zz = zz.reshape((nx,ny))

    return x,y,zz,zmin

#################################
if __name__ == '__main__':
#################################

    '''
    example to run 
    run fds2frontShp.py -p ./LS4 -sF 50
    where ./LS4 is the path to my fds-levelSet simulation
    '''

    importlib.reload(slread)

    parser = argparse.ArgumentParser(description='map WII')
    parser.add_argument('-p','--path2FDSdir', help='fds simulation directory path',required=True)
    parser.add_argument('-slcPhiId','--slcPhiId', help='id number of the slc level set variable',required=False)
    parser.add_argument('-sF','--skipFrame', help='number of frame to skip',required=False)
    args = parser.parse_args()

    inputDir = args.path2FDSdir
    
    if args.slcPhiId == None:
        levelSetIdSlc = 1
    else:
        levelSetIdSlc = int(args.slcPhiId)

    if args.skipFrame == None:
        skipFrame = 1
    else:
        skipFrame = int(args.skipFrame)


    #load simulation info
    fdsConfigFile = glob.glob(inputDir+'/*.fds')[0]
    print('reading FDS config file', fdsConfigFile)
    simName = None
    crs_fds = None
    x0,y0 = None, None
    with open(fdsConfigFile,'r') as f: 
        lines = f.readlines()
        for line_ in lines: 
            if 'Selected UTM CRS:' in line_: 
                crs_fds = line_.split(':')[1].strip()
            if 'Domain origin:' in line_: 
                x0,y0 = np.array(line_.strip().replace('E','').replace('N','').split(': ')[1].split(' '), dtype=float)
                
            if 'HEAD' in line_: 
                simName = line_.split("HID='")[1].split("'")[0]

    if (simName == None) | (crs_fds == None) | (x0 == None) | (y0 == None):
        print('could not find info in fds config file, stop here')
        sys.exit()


    xa = []
    ya = []
    zza = []
    phia  = []
    for imesh in range(6):
        fname = '{:s}/{:}_{:d}_{:d}.sf'.format(inputDir,simName,imesh+1,levelSetIdSlc)
        phi, times = slread.slread(fname, 0, 3000, 1000, gridskip=1, timeskip=1)
    
        fname = '{:s}/{:}_{:d}.ter'.format(inputDir,simName,imesh+1)
        x,y,zz,zmin = getTerrain(fname)

        xa.append(x)
        ya.append(y)
        zza.append(zz)
        phia.append(phi)

    nx = xa[0].shape[0]
    ny = ya[0].shape[0]
    nt = phia[0].shape[-1]

    #Big domain
    dx = np.diff(xa).mean()/2
    dy = np.diff(ya).mean()/2
    
    xmin = np.array(xa).min()
    xmax = np.array(xa).max()+dx
    ymin = np.array(ya).min()
    ymax = np.array(ya).max()+dy


    xbig = np.arange(xmin,xmax,dx)
    ybig = np.arange(ymin,ymax,dy)


    grid_y, grid_x = np.meshgrid(ybig, xbig)


    polygons = []
    times_polygon = []
    for it in range(nt)[::skipFrame]:
        points = []
        values = []
        for imesh in range(6):
            for i in range(nx):
                for j in range(ny):
                    points.append([xa[imesh][i],ya[imesh][j]])
                    values.append(phia[imesh][i,j,it])

        phi = interpolate.griddata(points, values, (grid_x, grid_y), method='linear', fill_value=-1)

        firemask = np.where(phi>0, 1, 0)
  
        contours, _ = cv2.findContours(firemask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        contoursXY = []
        for c in contours:
            idx = c.reshape(-1,2) 
            contoursXY.append( np.dstack(( grid_x[(np.dstack(idx)[0][1], np.dstack(idx)[0][0])] + x0, 
                                           grid_y[(np.dstack(idx)[0][1], np.dstack(idx)[0][0])] + y0) )[0] )

        for c in contoursXY: 
            if len(c) >= 3:
                polygon_ = polygon.Polygon( c.reshape(-1, 2) )
                polygons.append( polygon_) #.buffer(10, join_style=1).buffer(-10.0, join_style=1) )
   
                times_polygon.append(times[it])
        
        
    gdf = gpd.GeoDataFrame( geometry=polygons, crs=crs_fds)
    gdf['time'] = times_polygon

    gdf.plot(facecolor='none')
    plt.show()

    gdf.to_file('{:s}/{:}_perimeter.shp'.format(inputDir,simName))
    #    if it>500: 
    #        plt.imshow(firemask.T, origin='lower')
    #        plt.show()
