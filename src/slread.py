#!/usr/bin/python3
#McDermott
#2019-10-08

import struct
import numpy as np
import pdb 
import matplotlib.pyplot  as plt
import time 

def slread( fname, Tstart, Tend, *args, **kwargs ):

    """
    Reads FDS slice file
    Based on slread.m by Simo Hostikka
    https://github.com/firemodels/fds/blob/master/Utilities/Matlab/scripts/slread.m
    (QQ,Time)=slread(fname,Tstart,Tend [,Nframes, gridskip, timeskip]);
      Tstart  is start time
      Tend    is end time
      Nframes is number of slice file frames (FDS default is 1000 between Tstart and Tend)
      gridskip is a skip rate for reading cells: =2 --> read every other cell, etc.
      timeskip is a skip rate for frames: =2 --> read every other frame, etc.
      QQ      contains the data
      Time    contains the time points
    """

    #print(fname)
    #print(Tstart)
    #print(Tend)

    if len(args)==0:
        Nframes = 10000
    else:
        Nframes = args[0]
        
    gridskip = kwargs['gridskip'] if 'gridskip' in kwargs else 1
    timeskip = kwargs['timeskip'] if 'timeskip' in kwargs else 1
        
    f = open(fname,'rb')

    #print('#######')
    f.read(4)
    Str1 = f.read(30)       # quantity
    #print(Str1)
    f.read(8)
    Str2 = f.read(30)       # short name
    #print(Str2)
    f.read(8)
    Str3 = f.read(30)       # units
    #print(Str3)
    f.read(8)
    Indx = struct.unpack('6i',f.read(24))  # index bounds i,j,k
    #print(Indx)
    f.read(4)

    # allocate arrays for QQ and Time

    Isize = Indx[1]-Indx[0]+1
    Jsize = Indx[3]-Indx[2]+1
    Ksize = Indx[5]-Indx[4]+1
    if Isize==1:
       M = Jsize
       N = Ksize
    elif Jsize==1:
       M = Isize
       N = Ksize
    else:
       M = Isize
       N = Jsize

    Nframes = max(1,Nframes)
    QQ = np.zeros((M,N))
    Time = np.zeros(Nframes+1)
    
    ii = np.arange(0,M,gridskip)
    jj = np.arange(0,N,gridskip)
    tt = np.arange(0,Nframes+1,timeskip)
    Qskip  = np.zeros((len(ii),len(jj),len(tt)))

    st = 0

    while Time[st] < Tstart:

        f.read(4)
        Time_list = struct.unpack('f',f.read(4))
        Time[st] = Time_list[0]
        f.read(8)
        for n in range(N):
            for m in range(M):
                QQ_list = struct.unpack('f',f.read(4))
                QQ[m,n] = QQ_list[0]
        f.read(4)

    while Time[st-1] < Tend:

        f.read(4)
        try: 
            Time_list = struct.unpack('f',f.read(4))
        except: 
            pdb.set_trace()
        Time[st] = Time_list[0]
        f.read(8)
        for n in range(N):
            for m in range(M):
                QQ_list = struct.unpack('f',f.read(4))
                QQ[m,n] = QQ_list[0]
        if st%timeskip==0:
            #print(st,int(st/timeskip))
            Qskip[:,:,int(st/timeskip)] = QQ[np.ix_(ii,jj)]
        f.read(4)
        st = st + 1
        if st>Nframes:
            break

    return(Qskip[:,:,:st],Time[:st])

#################################
def bfread( fname, Time_start, Time_end, *args, **kwargs ):

    """
    Reads FDS boundaty file
    Based on slread.m by Simo Hostikka
    https://github.com/firemodels/fds/blob/master/Utilities/Matlab/scripts/slread.m
    (QQ,Time)=slread(fname,Tstart,Tend [,Nframes, gridskip, timeskip]);
      Tstart  is start time
      Tend    is end time
      Nframes is number of slice file frames (FDS default is 1000 between Tstart and Tend)
      gridskip is a skip rate for reading cells: =2 --> read every other cell, etc.
      timeskip is a skip rate for frames: =2 --> read every other frame, etc.
      QQ      contains the data
      Time    contains the time points
    """

    #print(fname)
    #print(Tstart)
    #print(Tend)

    if len(args)==0:
        Nframes = 10000
    else:
        Nframes = args[0]
        
    gridskip = kwargs['gridskip'] if 'gridskip' in kwargs else 1
    timeskip = kwargs['timeskip'] if 'timeskip' in kwargs else 1
        
    f = open(fname,'rb')

    #print('#######')
    f.read(4)
    Str1 = f.read(30)       # quantity
    #print(Str1)
    f.read(8)
    Str2 = f.read(30)       # short name
    #print(Str2)
    f.read(8)
    Str3 = f.read(30)       # units
    #print(Str3)
    f.read(8)
    npatch = struct.unpack('1i', f.read(4))       # units
    #print('npatch = ', npatch)
    
    Indx = []
    for inp in range(npatch[0]):
        f.read(8)
        Indx.append(struct.unpack('9i',f.read(36)))  # index bounds i,j,k
        #print(Indx[-1])
    
    Nframes = max(1,Nframes)
    
    time = -999
    data_time = []
    flag_time_plot = True
    data_bf = []
    time_bf = []
    st = 0
    Qskip = None
    while time < Time_end:
        f.read(8)
        time = struct.unpack('1f',f.read(4))[0]
        #print(time)
        time_bf.append(time)
        #print(time)
        for inp in range(npatch[0]):
           
            Isize = max(Indx[inp][1]-Indx[inp][0]+1,1)
            Jsize = max(Indx[inp][3]-Indx[inp][2]+1,1)
            Ksize = max(Indx[inp][5]-Indx[inp][4]+1,1)
            
            if (st == 0) and (Qskip is None) : 
                M =np.array( Indx )[:,0:2].max()+1
                N =np.array( Indx )[:,2:4].max()+1
                ii = np.arange(0,M,gridskip)
                jj = np.arange(0,N,gridskip)
                Qskip  = np.zeros((len(ii),len(jj),Nframes))
            
            if Isize==1:
               M = Jsize
               N = Ksize
            elif Jsize==1:
               M = Isize
               N = Ksize
            else:
               M = Isize
               N = Jsize   
            
            ii = np.arange(0,M,gridskip)
            jj = np.arange(0,N,gridskip)
            
            nbre_float = Isize*Jsize*Ksize
            f.read(8)
            QQ = np.zeros((M,N))
            
            #print(Indx[inp])
            try:
                 QQ_list = struct.unpack('{:d}f'.format(nbre_float),f.read(nbre_float*4)) 
            except: 
                pdb.set_trace()
            if Indx[inp][-3]!=3: continue
            if Isize * Jsize > 4: continue
            try: 
                ii_= 0
                for n in range(N):
                    for m in range(M):
                        QQ[m,n] = QQ_list[ii_]
                        ii_ += 1
            except: 
                pdb.set_trace()
            try:
                Qskip[Indx[inp][0],Indx[inp][2],int(st/timeskip)] = QQ[0,0] # QQ[np.ix_(ii,jj)][0,0] 
            except: 
                pdb.set_trace()
        st = st + 1

    f.close()
    return Qskip[:,:,:st], time_bf

    '''
        Indx=np.array(Indx)
        imax =   Indx[:,0:2].max()
        jmax =   Indx[:,2:4].max()
        kmax =   Indx[:,3:5].max()

        data = np.zeros([imax+1,jmax+1])
        for inp in range(npatch[0]):
            #print(time, Indx[inp])
            Isize = max(Indx[inp][1]-Indx[inp][0]+1,1)
            Jsize = max(Indx[inp][3]-Indx[inp][2]+1,1)
            Ksize = max(Indx[inp][5]-Indx[inp][4]+1,1)
            
            if Isize==1:
               M = Jsize
               N = Ksize
            elif Jsize==1:
               M = Isize
               N = Ksize
            else:
               M = Isize
               N = Jsize


            ior = Indx[inp][6]
            if ior != 3: continue
            if Isize*Jsize > 4: continue
            data[Indx[inp][0]:Indx[inp][1],Indx[inp][2]:Indx[inp][3]] = np.array(data_bf[inp]).reshape(Isize,Jsize, order='F')[0,0]
            #if  np.array(data_bf[inp]).max()> 0: 
            #    pdb.set_trace()
        data_time.append(data)
        
        if (flag_time_plot) & (data.max()>0): 
            plt.imshow(data.T, origin='lower')
            plt.title('{:s} | {:.1f}'.format(fname.split('/')[-1], time))
            plt.show()
            pdb.set_trace()
            flag_time_plot = False

    return np.array(data_time)
    '''

if __name__ == '__main__':
   
    databf = []
    for nproc in range(40):
        fname = '/data/paugam/FDS/LS-elpontdeVilomara/practical2024/3-fdsLS-LS4-fireAtm/hill_{:d}_1.bf'.format(nproc+1)
        databf.append( bfread(fname , 0, 2000) )

    # Create an empty array of shape 120x40
    idxMesh = np.zeros((40, 120), dtype=int)

    # Initialize the block value
    block_value = 1

    # Iterate over 12x10 blocks
    for j in range(0, 120, 12):   # Rows from 0 to 120 with step of 12
        for i in range(0, 40, 10): # Columns from 0 to 40 with step of 10
            idxMesh[i:i+10, j:j+12] = block_value  # Fill the block
            block_value += 1  # Increment the block value
    
    dataAll = np.zeros(idxMesh.shape)
    plt.ion()
    fig, ax = plt.subplots()

    for it in np.arange(200,len(databf[0]),10):
        print(it)
        for idx in np.unique(idxMesh):
            idx_ = np.where(idxMesh == idx)

            dataAll[idx_] = databf[idx-1][it].flatten()
            ax.clear()
            ax.imshow(dataAll.T, origin='lower')
            if it == 0 :
                plt.show()
            else: 
                plt.draw()
            plt.pause(0.1)
            
