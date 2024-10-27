import numpy as np 
import matplotlib.pyplot as plt
import geopandas as gpd
import pyproj
import argparse
import importlib
import glob 
from scipy import interpolate
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from affine import Affine
import f90nml 

import getTerrain
import slread
import tools 

#################################
if __name__ == '__main__':
#################################

    '''
    example to run 
    run fds2frontShp.py -p ./LS4 -sF 50
    where ./LS4 is the path to my fds-levelSet simulation
    '''

    importlib.reload(slread)

    parser = argparse.ArgumentParser(description='extract fire perimeters from FDS level set simulation')
    parser.add_argument('-p','--path2FDSdir', help='fds simulation directory path',required=True)
    parser.add_argument('-shp','--path2ShpFile', help='shp file to convert to xb',required=True)
    args = parser.parse_args()

    inputDir = args.path2FDSdir + '/'
    outputDir = args.path2FDSdir + '/Preproc/'
    tools.ensure_dir(outputDir)
    
    #load simulation info
    fdsConfigFile = glob.glob(inputDir+'/*.fds')[0]
    print('reading FDS config file', fdsConfigFile)
   
    simName = None
    crs_fds = None
    x0,y0 = None, None
    nproc = None
    with open(fdsConfigFile,'r') as f: 
        lines = f.readlines()
        for line_ in lines: 
            if 'Selected UTM CRS:' in line_: 
                crs_fds = line_.split(':')[1].strip()
            if 'Domain origin:' in line_: 
                x0,y0 = np.array(line_.strip().replace('E','').replace('N','').split(': ')[1].split(' '), dtype=float)
                
            if 'HEAD' in line_: 
                simName = line_.split("HID='")[1].split("'")[0]
           
            if 'meshes of' in line_: 
                nproc = int(np.prod(np.array(line_.split("meshes of")[0].split("·")).astype(float)))

    if (simName == None) | (crs_fds == None) | (x0 == None) | (y0 == None):
        print('could not find info in fds config file, stop here')
        sys.exit()
   
    nml=f90nml.read(fdsConfigFile)

    # Define the CRS from the string
    crs_fds = pyproj.CRS.from_string("WGS 84 / UTM zone 31N")
    
    WGS84proj  = pyproj.Proj('EPSG:4326')
    UTMproj    = crs_fds
    lonlat2xy = pyproj.Transformer.from_proj(WGS84proj,UTMproj)
    xy2lonlat = pyproj.Transformer.from_proj(UTMproj,WGS84proj)

    #check number of proc
    files_ = glob.glob('{:s}/{:}_*.ter'.format(inputDir,simName))
    if nproc > len(files_): 
        print('missing files')
        sys.exit()


    xa = []
    ya = []
    zza = []
    for imesh in range(nproc):
        fname = '{:s}/{:}_{:d}.ter'.format(inputDir,simName,imesh+1)
        x,y,zz,zmin = getTerrain.getTerrain(fname)

        xa.append(x)
        ya.append(y)
        zza.append(zz)

    nx = [xx.shape[0]  for xx in xa]
    ny = [yy.shape[0]  for yy in ya]
    
    #Big domain # assume all mesh have same resolution
    dxfds = np.diff(xa[0]).mean()
    dyfds = np.diff(ya[0]).mean()
    
    dx = dxfds
    dy = dyfds
    dz = (nml['mesh']['xb'][-1]-nml['mesh']['xb'][-2])/nml['mesh']['ijk'][-1]

    xamin = min([xx.min() for xx in xa ]) 
    xamax = max([xx.max() for xx in xa ]) 
    yamin = min([yy.min() for yy in ya ]) 
    yamax = max([yy.max() for yy in ya ]) 

    xmin = np.round(xamin,3)
    xmax = np.round(xamax,3)
    ymin = np.round(yamin,3)
    ymax = np.round(yamax,3)

    xbig = np.arange(xmin+dxfds/2,xmax,dx)
    ybig = np.arange(ymin+dyfds/2,ymax,dy)

    if xbig.shape[0] %2 != 0: 
        nxb = xbig.shape[0]+1
        xbig = np.linspace(xmin+dxfds/2,xmax,nxb)
    if ybig.shape[0] %2 != 0: 
        nyb = ybig.shape[0]+1
        ybig = np.linspace(ymin+dyfds/2,ymax,nyb)
    
    dx=xbig[1]-xbig[0] 
    dy=ybig[1]-ybig[0] 
    print('resolution = ',dx,dy)

    grid_y, grid_x = np.meshgrid(ybig, xbig)

    grid_x = np.round(grid_x,3)
    grid_y = np.round(grid_y,3)
    
    grid_e = grid_x + x0
    grid_n = grid_y + y0
    
    polygons = []
    times_polygon = []
    
    points = []
    valuesPhi = []
    valuesTer = []
    for imesh in range(nproc):
        for i in range(nx[imesh]):
            for j in range(ny[imesh]):
                points.append([xa[imesh][i],ya[imesh][j]])
                valuesTer.append(zza[imesh][i,j])
        
    terrain = interpolate.griddata(points, valuesTer, (grid_x, grid_y), method='linear', fill_value=-999)
  

    #load shapefile
    gdf = gpd.read_file(args.path2ShpFile)
    gdf = gdf.to_crs(crs_fds)
    
    print('   ')
    for index, row in gdf.iterrows():
        geom = row.geometry
        
        min_x = grid_e.min()
        max_y = grid_n.min()
        height, width = grid_x.shape[1], grid_x.shape[0]
        transform = Affine.translation(min_x, max_y) * Affine.scale(dx, dy)

        if geom.geom_type == 'Point':
            ii,jj = ~transform*[geom.x,geom.y]
            ii = int(np.round(ii,0))
            jj = int(np.round(jj,0))
           
            print( row['name'] )
            print( "XB={:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(grid_x[ii,jj],grid_x[ii+1,jj+1] , 
                                                                          grid_y[ii,jj],grid_y[ii+1,jj+1], 
                                                                          terrain[ii,jj], terrain[ii,jj]+dz  ))
            print('   ')
            print('   ')

        if geom.geom_type == 'Polygon':
            shapes = [geom for geom in devc_gdf.geometry]

            mask = geometry_mask(shapes, transform=transform, invert=True, 
                                 out_shape=(height, width),all_touched=True).astype(np.uint8).T

            idx = np.where(mask==1)
            print( "XB={:.2f},{:.2f},{:.2f},{:.2f},,{:.2f},{:.2f}".format(grid_x[idx].min(),grid_x[idx].max()+dx ,grid_y[idx].min(),grid_y[idx].max()+dy, terrain[idx].min(), terrain[idx].max()+dz  ))
