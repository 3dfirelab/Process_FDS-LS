#Paugam Ronan
#2023-11-20

import numpy as np 
import importlib 
import struct 
import pdb 
from scipy import interpolate
import matplotlib.pyplot as plt
import sys 
from shapely.geometry import polygon
import cv2 
import geopandas as gpd 
import pandas as pd 
import argparse
import glob 
import xarray as xr
from pyproj import CRS
import importlib 
import rioxarray  # This is required to add the .rio functionality
from pyproj import Transformer
import pyproj
import shapely 
import f90nml
import os 

#homebrewed
import getbmap
import ros 
#import map_georefImage as tools
import tools
import slread

importlib.reload(getbmap)
importlib.reload(ros)
importlib.reload(slread)

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
    lenz = struct.unpack('i',f.read(4))[0]
    zz = np.array(struct.unpack('{:d}f'.format(lenz//4),f.read(lenz)))
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

    parser = argparse.ArgumentParser(description='extracts fire perimeters from slive file of FDS level set, and compute ROS. It also computes biomass burnt from boundary file of MASS FLUX')
    parser.add_argument('-p','--path2FDSdir', help='fds simulation directory path',required=True)
    parser.add_argument('-slcPhiId','--slcPhiId', help='id number of the slc level set variable',required=False)
    parser.add_argument('-bfMLRId','--bfMLRId', help='id number of the bndf MASS FLUX variable',required=False)
    #parser.add_argument('-slcHrrId','--slcHrrId', help='id number of the slc HRRPUV variable',required=False)
    parser.add_argument('-sF','--skipFrame', help='number of frame to skip',required=False)
    parser.add_argument('-tE','--timeEnd', help='timeEnd, to for example if the simulation crashes before the end',required=False)
    args = parser.parse_args()

    inputDir = args.path2FDSdir + '/'
    outputDir = args.path2FDSdir + '/Postproc/'
    tools.ensure_dir(outputDir)
    tools.ensure_dir(outputDir+'shp/')
    
    if args.slcPhiId == None:
        levelSetIdSlc = 1
    else:
        levelSetIdSlc = int(args.slcPhiId)
    
    if args.bfMLRId == None:
        bfMLRId = 1
    else:
        bfMLRId = int(args.bfMLRId)
    
    #if args.slcHrrId == None:
    #    hrrIdSlc = 2
    #else:
    #    hrrIdSlc = int(args.slcHrrId)
    
    if args.skipFrame == None:
        skipFrame = 1
    else:
        skipFrame = int(args.skipFrame)
    
    if args.timeEnd == None:
        timeEnd = None
    else:
        timeEnd = float(args.timeEnd)

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
           
            if timeEnd == None:
                if 'T_END' in line_: 
                    timeEnd = float(line_.split("T_END=")[1].split("/")[0]) - 1
            
            if 'meshes of' in line_: 
                nproc = int(np.prod(np.array(line_.split("meshes of")[0].split("·")).astype(float)))

    if (simName == None) | (crs_fds == None) | (x0 == None) | (y0 == None):
        print('could not find info in fds config file, stop here')
        sys.exit()
    
    nml=f90nml.read(fdsConfigFile)
    
    # Define the CRS from the string
    crs_fds = CRS.from_string("WGS 84 / UTM zone 31N")
    
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
    phia  = []
    #hrra = []
    mlra = []
    timesa = []
    print ('load FDS data:')
    for imesh in range(nproc):
        print('  {:.1f}% \r'.format(100*imesh/nproc), end='')
        
        fname = '{:s}/{:}_{:d}_{:d}.sf'.format(inputDir,simName,imesh+1,levelSetIdSlc)
        phi, times = slread.slread(fname, 0, timeEnd, gridskip=1, timeskip=1)
        
        #fname = '{:s}/{:}_{:d}_{:d}.sf'.format(inputDir,simName,imesh+1,hrrIdSlc)
        #hrr, times = slread.slread(fname, 0, timeEnd, gridskip=1, timeskip=1)
    
        fname = '{:s}/{:}_{:d}.ter'.format(inputDir,simName,imesh+1)
        x,y,zz,zmin = getTerrain(fname)

        fname = '{:s}/{:}_{:d}_{:d}.bf'.format(inputDir,simName,imesh+1,bfMLRId)
        if os.path.isfile(fname):
            mlr, times_mlr  =  slread.bfread(fname, 0, timeEnd, gridskip=1, timeskip=1)
        else: 
            mlr = None

        xa.append(x)
        ya.append(y)
        zza.append(zz)
        phia.append(phi)
        #hrra.append(hrr)
        timesa.append(times)
        if mlr is not None:
            mlra.append(mlr)

        #print ('{:d} mesh with time max = {:.2f} with dim {:d}'.format(imesh,times.max(),times.shape[0]) )
        #print ('{:d} mesh with phi dim {:d}'.format(imesh,phi.shape[-1]) )
    print ('  done        ')
    
    nx = [xx.shape[0]  for xx in xa]
    ny = [yy.shape[0]  for yy in ya]
    nt = [tt.shape[-1] for tt in phia]
    if np.diff(nt).sum()==0: # all same, this is what we expect
        nt = nt[0]
    else: 
        print ('all meshes do not have same time dimension')
        print ('stop')
        sys.exit()
    
    #Big domain # assume all mesh have same resolution
    print ('create grid:')
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

    #xbig = np.arange(xmin+dxfds/2,xmax,dx)
    #ybig = np.arange(ymin+dyfds/2,ymax,dy)
    xbig = np.arange(xmin,xmax,dx)
    ybig = np.arange(ymin,ymax,dy)

    #if xbig.shape[0] %2 != 0: 
    #    nxb = xbig.shape[0]+1
    #    xbig = np.linspace(xmin+dxfds/2,xmax,nxb)
    #if ybig.shape[0] %2 != 0: 
    #    nyb = ybig.shape[0]+1
    #    ybig = np.linspace(ymin+dyfds/2,ymax,nyb)
    
    dx=xbig[1]-xbig[0] 
    dy=ybig[1]-ybig[0] 
    print('  resolution = ',dx,dy)

    grid_y, grid_x = np.meshgrid(ybig, xbig)

    grid_x = np.round(grid_x,3)
    grid_y = np.round(grid_y,3)
    print('  done')
    
    #Interpolate FDS data on 2D horizontal grid
    arrivalTime = np.zeros_like(grid_x) - 999
    endTime = np.zeros_like(grid_x) - 999
    BB = np.zeros_like(grid_x)  # Fire Emited Energy
    polygons = []
    times_polygon = []
    nbFrame = nt/skipFrame
    print ('interpolate FDS data:')
    for it in range(nt)[::skipFrame]:
        points = []
        valuesPhi = []
        valuesMlr = []
        #valuesHrr = []
        if it == 0: valuesTer = []
        for imesh in range(nproc):
            for i in range(nx[imesh]):
                for j in range(ny[imesh]):
                    points.append([xa[imesh][i],ya[imesh][j]])
                    valuesPhi.append(phia[imesh][i,j,it])
                    if len(mlra)>0:  
                        valuesMlr.append(mlra[imesh][i,j,it])
                    else: 
                        valuesMlr.append(-999)
                    #valuesHrr.append(hrra[imesh][i,j,it])
                    if it ==0 : valuesTer.append(zza[imesh][i,j])
        
        print('  ({:.1f}%) t = {:.1f}\r'.format(100*it/nbFrame,times[it]), end='')

        
        phi = np.zeros_like(grid_x) - 999
        if it == 0: terrain = np.zeros_like(grid_x) - 999
        if len(mlra)>0: mlr = np.zeros_like(grid_x) - 999
        for pt_,phi_,mlr_,terrain_ in zip(points,valuesPhi,valuesMlr,valuesTer):
            #idx_ =  np.unravel_index(((grid_x-pt_[0]+dx/2)**2+ (grid_y-pt_[1]+dy/2)**2).argmin() , grid_x.shape)
            idx_ =  np.unravel_index(((grid_x-pt_[0])**2+ (grid_y-pt_[1])**2).argmin() , grid_x.shape)
            phi[idx_] = phi_
            if len(mlra)>0: mlr[idx_] = mlr_
            if it == 0 : terrain[idx_] = terrain_
        #phi = interpolate.griddata(points, valuesPhi, (grid_x, grid_y), method='linear', fill_value=-1)
        '''if (it%100 == 0) | ((times[it]>700)&(times[it]<720)):
            plt.imshow(phi.T, origin='lower')
            plt.show()
            pdb.set_trace()
        '''
        #hrr = interpolate.griddata(points, valuesHrr, (grid_x, grid_y), method='linear', fill_value=-1)
        
        #if it == 0 : terrain = interpolate.griddata(points, valuesTer, (grid_x, grid_y), method='linear', fill_value=-999)
        firemask = np.where(phi>=0, 1, 0)
  
        arrivalTime[np.where((firemask>0)&(arrivalTime<0))] = times[it]
        if len(mlra)>0:
            if it > 0:
                idx = np.where((mlr>0))
                BB[idx] += mlr[idx]*(times[it]-times[it-1])*(dx*dy)

                idx = np.where( (arrivalTime >0 ) & (mlr <= 0) & (endTime < 0))
                endTime[idx] = times[it]

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
    print ('  done             ')
    
    #FEE *= 1.e-3 # MJ
    #save front    
    gdf = gpd.GeoDataFrame( geometry=polygons, crs=crs_fds)
    gdf['time'] = times_polygon
    gdf.to_file('{:s}/shp/{:}_perimeter.shp'.format(outputDir,simName))
    
    #burntmap
    burntarea = firemask.copy()
    ignitionTime = arrivalTime[np.where(burntarea==1)].min()+1
    ignitionAera = np.where((burntarea==1)&(arrivalTime>=0)&(arrivalTime<=ignitionTime),1,0)
    burntarea = np.where((burntarea==1)&(arrivalTime>ignitionTime), 1, 0)

    #compute ROS
    print ('compute ROS')
    #first interpolation
    params = {}
    params['projection'] = 'epsg:{:d}'.format(crs_fds.to_epsg())
    #params['reso']       = 10.
    params['buffer']     = 1000.
    params['plot_name']  = 'test'                       
    params['flag_parallel'] = True 
    params['ros_max']    = 10.
    params['flag_use_time_reso_constraint'] = False
    params['interpolate_betwee_ff'] = 'griddata'
    #params['interpolate_betwee_ff'] = 'fsrbf'
    params['distance_interaction_rbf'] = 100.
      
    maps_fire = np.zeros(grid_x.shape, dtype=np.dtype([('grid_e',float),('grid_n',float),('plotMask',int),('terrain',float)]))
    maps_fire = maps_fire.view(np.recarray)
    maps_fire.grid_e= grid_x + x0
    maps_fire.grid_n= grid_y + y0
    maps_fire.plotMask= np.where(burntarea==1,2,0)
    maps_fire.terrain = terrain 
    
    arrivalTime_edgepixel = np.zeros_like(maps_fire.grid_e)
    arrivalTime_ignition_seed = None

    arrivalTime_interp, arrivalTime_FS, arrivalTime_clean = getbmap.interpolate_arrivalTime( outputDir, maps_fire,  gdf,                         \
                                                                                     arrivalTime, arrivalTime_edgepixel,                      \
                                                                                     params,                                               \
                                                                                     #flag_interpolate_betwee_ff= 'rbf',                        \
                                                                                     flag_plot                 = False,                    \
                                                                                     frame_time_period         = 0,            \
                                                                                     set_rbf_subset            = None,                         \
                                                                                     arrivalTime_ignition_seed =  arrivalTime_ignition_seed,   \
                                                                        )
    #then ros
    normal_x, normal_y, ros, ros_qc = ros.compute_ros( maps_fire, arrivalTime_interp, arrivalTime_clean) 

    '''
    #save ros
    rosxr = xr.DataArray(
                            ros.T,
                            coords={
                                'y': maps_fire.grid_n[0,:],
                                'x': maps_fire.grid_e[:,0]
                            },
                            dims=['y', 'x']
                        )
    '''

    # Create the dataset
    transformer = Transformer.from_crs(crs_fds, "EPSG:4326", always_xy=True)

    # Perform the transformation to UTM
    lon, lat = transformer.transform(maps_fire.grid_e, maps_fire.grid_n)
    x, y  = (grid_x+x0)[:,0]+dx/2, (grid_y+y0)[0,:]+dy/2

    #lats = np.linspace(lat[0,0],lat[0,-1],lat.shape[1])
    #lons = np.linspace(lon[0,0],lon[-1,0],lon.shape[0])
    
    #lats = np.round(lat[0,:],10)
    #lons = np.round(lon[:,0],10)

    #create da
    daTerrain = xr.DataArray(
          data = terrain.T,
          dims = ['y','x'],
          coords = {'x': x, 'y': y},
          attrs = {'long_name': 'terrain', 'units': 'm', '_fillValue': -999}
          )
    daTerrain = daTerrain.rio.write_crs(crs_fds, inplace=True)
    
    daBB = xr.DataArray(
          data = BB.T,
          dims = ['y','x'],
          coords = {'x': x, 'y': y},
          attrs = {'long_name': 'BIOMASS burnt', 'units': 'kg', '_fillValue': -999}
          )
    daBB= daBB.rio.write_crs(crs_fds, inplace=True)

    daArrivalTime = xr.DataArray(
          data = arrivalTime_interp.T,
          dims = ['y','x'],
          coords = {'x': x, 'y': y},
          attrs = {'long_name': 'arrivalTime', 'units': 's', '_fillValue': -999}
          )
    daArrivalTime = daArrivalTime.rio.write_crs(crs_fds, inplace=True)

    daRos = xr.DataArray(
          data = ros.T,
          dims = ['y','x'],
          coords = {'x': x, 'y': y},
          attrs = {'long_name': 'ROS', 'units': 'm/s', '_fillValue': -999}
          )
    daRos = daRos.rio.write_crs(crs_fds, inplace=True)

    daBA = xr.DataArray(
          data = burntarea.T,
          dims = ['y','x'],
          coords = {'x': x, 'y': y},
          attrs = {'long_name': 'burn area', 'units': '-', '_fillValue': -999}
          )
    daBA = daBA.rio.write_crs(crs_fds, inplace=True)

    daArrivalTimeFront = xr.DataArray(
          data = arrivalTime_clean.T,
          dims = ['y','x'],
          coords = {'x': x, 'y': y},
          attrs = {'long_name': 'arrivalTime cleaned', 'units': 's', '_fillValue': -999}
          )
    daArrivalTimeFront = daArrivalTimeFront.rio.write_crs(crs_fds, inplace=True)

    daIgnitionArea = xr.DataArray(
          data = ignitionAera.T,
          dims = ['y','x'],
          coords = {'x': x, 'y': y},
          attrs = {'long_name': 'arrivalTime cleaned', 'units': 's', '_fillValue': -999}
          )
    daIgnitionArea = daIgnitionArea.rio.write_crs(crs_fds, inplace=True)
    
    daDurationTime = xr.DataArray(
          data = (endTime-arrivalTime_interp).T,
          dims = ['y','x'],
          coords = {'x': x, 'y': y},
          attrs = {'long_name': 'durationTime', 'units': 's', '_fillValue': -999}
          )
    daDurationTime = daDurationTime.rio.write_crs(crs_fds, inplace=True)

    #ds
    ds = xr.Dataset({
          'ros':  daRos.rio.reproject("EPSG:4326"),
          'BB':  daBB.rio.reproject("EPSG:4326"),
          'arrivalTime': daArrivalTime.rio.reproject("EPSG:4326"),
          'arrivalTimeFront': daArrivalTimeFront.rio.reproject("EPSG:4326"),
          'burntarea': daBA.rio.reproject("EPSG:4326"),
          'terrainFDS':daTerrain.rio.reproject("EPSG:4326"),
          },
          attrs = {'simu origin': 'FDS-LS output from simulation :{:s}'.format(simName)}
          )
      
    # save datasets
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)
    ds.to_netcdf('{:}/{:}_FDSLS-FBM.nc'.format(outputDir,simName),format='NETCDF4')


    # Create a DataFrame from lat and lon with point measurement of ROS
    df = pd.DataFrame({'ros': ros.flatten(), 
                       'ros_qc':ros_qc.flatten(), 
                       'normal_x':normal_x.flatten(), 
                       'normal_y':normal_y.flatten()})

    #lon lat at the pixel center
    lon,lat = xy2lonlat.transform(maps_fire.grid_e.flatten()+ dx/2, maps_fire.grid_n.flatten()+ dy/2)

    # Create geometry points from lat/lon
    geometry = [shapely.geometry.Point(xy) for xy in zip(lat, lon)]

    gdfros = gpd.GeoDataFrame(df, geometry=geometry)
    
    #keep only >0 ros
    gdfros = gdfros[gdfros['ros']>0]
    
    #add direction
    ros_direction = np.arctan2(gdfros['normal_x'], gdfros['normal_y'])*180./3.14
    gdfros['ros_direct'] = ros_direction

    #save with wgs84 crs
    gdfros = gdfros.set_crs(epsg=4326)
    gdfros.to_file(outputDir+'/shp/'+simName+'_ros.shp')

    #plot
    #gdf.plot(cmap='viridis', column='time',facecolor='none')
    #plt.show()

    #    if it>500: 
    #        plt.imshow(firemask.T, origin='lower')
    #        plt.show()

    #load ignition shp file

    ignitionline = gpd.read_file('/data/paugam/FDS/LS-elpontdeVilomara/practical2024/2-qgis2fds/ignition.shp')

    fig = plt.figure(figsize=(15,6))
    
    ax = plt.subplot(131)
    daTerrain.plot(ax=ax)
    ax.set_aspect(1)
    ax.set_title('terrain')

    ax = plt.subplot(132)
    daRos.where(daRos>=0).plot(vmin=0,vmax=3)
    daIgnitionArea.where(daIgnitionArea==1).plot(cmap='Reds',vmin=0,vmax=1.1,add_colorbar=False)
    ignitionline.to_crs(daRos.rio.crs).plot(ax=ax,color='k')
    ax.set_aspect(1)
    ax.set_title('ROS')
   
    #file_path = "/data/paugam/FDS/LS-elpontdeVilomara/practical2024/2-qgis2fds/fuelmap.tif"
    #data_array = rioxarray.open_rasterio(file_path)
    ax = plt.subplot(133)
    #data_array.plot()
    daBB.where(daBB>0).plot()
    ignitionline.to_crs(daRos.rio.crs).plot(ax=ax,color='k')
    ax.set_aspect(1)
    ax.set_title('Biomass Burning')

    plt.show()

