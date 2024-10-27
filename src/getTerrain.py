import struct 
import numpy as np 


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
