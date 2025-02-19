## plot one simple
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# from pylib.texnet import create_catalog
# from pylib.texnet import win_catalog
import obspy.core.utcdatetime as utc

#
# column meaning: 
# all columns in GC file
# allcols = ["yr","mon","dy","hr","min","sec","evid","rlat","rlon","rdep","mag",
# "qix", "cix", "nb", "npair", "ndiffP","ndiffS","rmsP","rmsS","eh","ez","et",
# "clat","clon","cdep"]


lon1=-104.7
lon2=-103.7-0.25
lat1=31.3+0.2
lat2=31.9-0.15
fig=plt.figure(figsize=(18,10))
### 202208
## From initial EQCCT catalog
fname='../data/catalogs/delaware11140_growclust-cat-cc0.65-ngood6.txt'
fid=open(fname,'r')
lines=fid.readlines()
## Catalog
lats0=[float(ii.split()[22]) for ii in lines]
lons0=[float(ii.split()[23]) for ii in lines]
deps0=[float(ii.split()[24]) for ii in lines]
## Relocation refined
lats1=[float(ii.split()[7]) for ii in lines if float(ii.split()[13])>=2]
lons1=[float(ii.split()[8]) for ii in lines if float(ii.split()[13])>=2]
deps1=[float(ii.split()[9]) for ii in lines if float(ii.split()[13])>=2]
mags1=[float(ii.split()[10]) for ii in lines if float(ii.split()[13])>=2]
utcs1=[utc.UTCDateTime(int(ii.split()[0]), int(ii.split()[1]),  int(ii.split()[2]), int(ii.split()[3]), int(ii.split()[4]),  int(ii.split()[5].split('.')[0]), int(ii.split()[5].split('.')[1])) for ii in lines if float(ii.split()[13])>=2]

## From Relocated TexNet Catalog
fname='../data/catalogs/texnet_events_20220801_20220901_growclust-cat-cc0.65-ngood6.txt'
fid=open(fname,'r')
lines=fid.readlines()
# lats2=[float(ii.split()[22]) for ii in lines] #oy: one year
# lons2=[float(ii.split()[23]) for ii in lines]
# deps2=[float(ii.split()[24]) for ii in lines]
lats2=[float(ii.split()[7]) for ii in lines if float(ii.split()[13])>=2] #oyr: one year reloc
lons2=[float(ii.split()[8]) for ii in lines if float(ii.split()[13])>=2]
deps2=[float(ii.split()[9]) for ii in lines if float(ii.split()[13])>=2]
mags2=[float(ii.split()[10]) for ii in lines if float(ii.split()[13])>=2]

## From depth-corrected EQCCT catalog
fname='../data/catalogs/delaware202209xml_growclust-cat-cc0.65-ngood6.txt'
fid=open(fname,'r')
lines=fid.readlines()
# lats2=[float(ii.split()[22]) for ii in lines] #oy: one year
# lons2=[float(ii.split()[23]) for ii in lines]
# deps2=[float(ii.split()[24]) for ii in lines]
lats3=[float(ii.split()[7]) for ii in lines if float(ii.split()[13])>=2] #oyr: one year reloc
lons3=[float(ii.split()[8]) for ii in lines if float(ii.split()[13])>=2]
deps3=[float(ii.split()[9]) for ii in lines if float(ii.split()[13])>=2]
mags3=[float(ii.split()[10]) for ii in lines if float(ii.split()[13])>=2]
mg=mags3
mg=np.array(mg,dtype='float')
mg=(mg-mg.min())/mg.max()*20
utcs3=[utc.UTCDateTime(int(ii.split()[0]), int(ii.split()[1]),  int(ii.split()[2]), int(ii.split()[3]), int(ii.split()[4]),  int(ii.split()[5].split('.')[0]), int(ii.split()[5].split('.')[1])) for ii in lines if float(ii.split()[13])>=2]
t=utcs3
t=np.array(t,dtype='float');

ax = plt.subplot(3,2,1)
## Initial
l1=plt.scatter(lons1,deps1,10,marker='s',facecolor='blue',edgecolor='k',alpha=0.5,label='Initial')
plt.gca().set_ylim(ymin=0,ymax=10);
plt.gca().invert_yaxis();
plt.gca().set_xlim(xmin=lon1,xmax=lon2)
## Depth-corrected
l3=plt.scatter(lons3,deps3,s=mg,c=t,cmap='spring',marker='o',facecolor='blue',edgecolor='k',alpha=0.9,label='Depth-corrected')
# cbar=plt.colorbar(l3,ticks=[t.min(), (t.min()+t.max())/2, t.max()])
# cbar.ax.set_yticklabels(['Oct 1 2022', 'Oct 16 2022', 'Oct 31 2022'])
## TexNet
l2=plt.scatter(lons2,deps2,20,marker='s',facecolor='red',edgecolor='k',alpha=0.9,label='TexNet')
plt.gca().set_ylim(ymin=0,ymax=10);
plt.gca().invert_yaxis();
plt.gca().set_xlim(xmin=lon1,xmax=lon2)
# plt.xlabel('Longitude (deg)');
plt.ylabel('Depth (km)');
plt.gca().legend(loc='lower left', fontsize = 8)
plt.gca().text(lon1-0.1,-0.5,'(a)',size=20,weight='normal')

ax = plt.subplot(3,2,2)
## Initial
l1=plt.scatter(lats1,deps1,10,marker='s',facecolor='blue',edgecolor='k',alpha=0.5,label='Initial')
plt.gca().set_ylim(ymin=0,ymax=10);
plt.gca().invert_yaxis();
plt.gca().set_xlim(xmin=lat1,xmax=lat2)
## Depth-corrected
l3=plt.scatter(lats3,deps3,s=mg,c=t,cmap='spring',marker='o',facecolor='blue',edgecolor='k',alpha=0.9,label='Depth-corrected')
cbar=plt.colorbar(l3,ticks=[t.min(), (t.min()+t.max())/2, t.max()])
cbar.ax.set_yticklabels(['Aug 1 2022', 'Aug 16 2022', 'Aug 31 2022'])
## TexNet
l2=plt.scatter(lats2,deps2,20,marker='s',facecolor='red',edgecolor='k',alpha=0.9,label='TexNet')
plt.gca().set_ylim(ymin=0,ymax=10);
plt.gca().invert_yaxis();
plt.gca().set_xlim(xmin=lat1,xmax=lat2)
# plt.xlabel('Latitude (deg)');
plt.ylabel('Depth (km)');
# plt.gca().legend(loc='lower left', fontsize = 8)

### 202209
## From initial EQCCT catalog
fname='../data/catalogs/delaware10867_growclust-cat-cc0.65-ngood6.txt'
fid=open(fname,'r')
lines=fid.readlines()
## Catalog
lats0=[float(ii.split()[22]) for ii in lines]
lons0=[float(ii.split()[23]) for ii in lines]
deps0=[float(ii.split()[24]) for ii in lines]
## Relocation refined
lats1=[float(ii.split()[7]) for ii in lines if float(ii.split()[13])>=2]
lons1=[float(ii.split()[8]) for ii in lines if float(ii.split()[13])>=2]
deps1=[float(ii.split()[9]) for ii in lines if float(ii.split()[13])>=2]
mags1=[float(ii.split()[10]) for ii in lines if float(ii.split()[13])>=2]
utcs1=[utc.UTCDateTime(int(ii.split()[0]), int(ii.split()[1]),  int(ii.split()[2]), int(ii.split()[3]), int(ii.split()[4]),  int(ii.split()[5].split('.')[0]), int(ii.split()[5].split('.')[1])) for ii in lines if float(ii.split()[13])>=2]

## From Relocated TexNet Catalog
fname='../data/catalogs/texnet_events_20220901_20221001_growclust-cat-cc0.65-ngood6.txt'
fid=open(fname,'r')
lines=fid.readlines()
# lats2=[float(ii.split()[22]) for ii in lines] #oy: one year
# lons2=[float(ii.split()[23]) for ii in lines]
# deps2=[float(ii.split()[24]) for ii in lines]
lats2=[float(ii.split()[7]) for ii in lines if float(ii.split()[13])>=2] #oyr: one year reloc
lons2=[float(ii.split()[8]) for ii in lines if float(ii.split()[13])>=2]
deps2=[float(ii.split()[9]) for ii in lines if float(ii.split()[13])>=2]
mags2=[float(ii.split()[10]) for ii in lines if float(ii.split()[13])>=2]

## From depth-corrected EQCCT catalog
fname='../data/catalogs/delaware202209xml_growclust-cat-cc0.65-ngood6.txt'
fid=open(fname,'r')
lines=fid.readlines()
# lats2=[float(ii.split()[22]) for ii in lines] #oy: one year
# lons2=[float(ii.split()[23]) for ii in lines]
# deps2=[float(ii.split()[24]) for ii in lines]
lats3=[float(ii.split()[7]) for ii in lines if float(ii.split()[13])>=2] #oyr: one year reloc
lons3=[float(ii.split()[8]) for ii in lines if float(ii.split()[13])>=2]
deps3=[float(ii.split()[9]) for ii in lines if float(ii.split()[13])>=2]
mags3=[float(ii.split()[10]) for ii in lines if float(ii.split()[13])>=2]
mg=mags3
mg=np.array(mg,dtype='float')
mg=(mg-mg.min())/mg.max()*20
utcs3=[utc.UTCDateTime(int(ii.split()[0]), int(ii.split()[1]),  int(ii.split()[2]), int(ii.split()[3]), int(ii.split()[4]),  int(ii.split()[5].split('.')[0]), int(ii.split()[5].split('.')[1])) for ii in lines if float(ii.split()[13])>=2]
t=utcs3
t=np.array(t,dtype='float');
plt.gca().text(lat1-0.04,-0.5,'(b)',size=20,weight='normal')

ax = plt.subplot(3,2,3)
## Initial
l1=plt.scatter(lons1,deps1,10,marker='s',facecolor='blue',edgecolor='k',alpha=0.5,label='Initial')
plt.gca().set_ylim(ymin=0,ymax=10);
plt.gca().invert_yaxis();
plt.gca().set_xlim(xmin=lon1,xmax=lon2)
## Depth-corrected
l3=plt.scatter(lons3,deps3,s=mg,c=t,cmap='spring',marker='o',facecolor='blue',edgecolor='k',alpha=0.9,label='Depth-corrected')
# cbar=plt.colorbar(l3,ticks=[t.min(), (t.min()+t.max())/2, t.max()])
# cbar.ax.set_yticklabels(['Oct 1 2022', 'Oct 16 2022', 'Oct 31 2022'])
## TexNet
l2=plt.scatter(lons2,deps2,20,marker='s',facecolor='red',edgecolor='k',alpha=0.9,label='TexNet')
plt.gca().set_ylim(ymin=0,ymax=10);
plt.gca().invert_yaxis();
plt.gca().set_xlim(xmin=lon1,xmax=lon2)
#plt.xlabel('Longitude (deg)');
plt.ylabel('Depth (km)');
plt.gca().legend(loc='lower left', fontsize = 8)

plt.gca().text(lon1-0.1,-0.5,'(c)',size=20,weight='normal')


ax = plt.subplot(3,2,4)
## Initial
l1=plt.scatter(lats1,deps1,10,marker='s',facecolor='blue',edgecolor='k',alpha=0.5,label='Initial')
plt.gca().set_ylim(ymin=0,ymax=10);
plt.gca().invert_yaxis();
plt.gca().set_xlim(xmin=lat1,xmax=lat2)
## Depth-corrected
l3=plt.scatter(lats3,deps3,s=mg,c=t,cmap='spring',marker='o',facecolor='blue',edgecolor='k',alpha=0.9,label='Depth-corrected')
cbar=plt.colorbar(l3,ticks=[t.min(), (t.min()+t.max())/2, t.max()])
cbar.ax.set_yticklabels(['Sep 1 2022', 'Sep 16 2022', 'Sep 30 2022'])
## TexNet
l2=plt.scatter(lats2,deps2,20,marker='s',facecolor='red',edgecolor='k',alpha=0.9,label='TexNet')
plt.gca().set_ylim(ymin=0,ymax=10);
plt.gca().invert_yaxis();
plt.gca().set_xlim(xmin=lat1,xmax=lat2)
#plt.xlabel('Latitude (deg)');
plt.ylabel('Depth (km)');
# plt.gca().legend(loc='lower left', fontsize = 8)

###202210
## From initial EQCCT catalog
fname='../data/catalogs/delaware11687_growclust-cat-cc0.65-ngood6.txt'
fid=open(fname,'r')
lines=fid.readlines()
## Catalog
lats0=[float(ii.split()[22]) for ii in lines]
lons0=[float(ii.split()[23]) for ii in lines]
deps0=[float(ii.split()[24]) for ii in lines]
## Relocation refined
lats1=[float(ii.split()[7]) for ii in lines if float(ii.split()[13])>=2]
lons1=[float(ii.split()[8]) for ii in lines if float(ii.split()[13])>=2]
deps1=[float(ii.split()[9]) for ii in lines if float(ii.split()[13])>=2]
mags1=[float(ii.split()[10]) for ii in lines if float(ii.split()[13])>=2]
utcs1=[utc.UTCDateTime(int(ii.split()[0]), int(ii.split()[1]),  int(ii.split()[2]), int(ii.split()[3]), int(ii.split()[4]),  int(ii.split()[5].split('.')[0]), int(ii.split()[5].split('.')[1])) for ii in lines if float(ii.split()[13])>=2]

## From Relocated TexNet Catalog
fname='../data/catalogs/texnet_events_20221001_20221101_growclust-cat-cc0.65-ngood6.txt'
fid=open(fname,'r')
lines=fid.readlines()
# lats2=[float(ii.split()[22]) for ii in lines] #oy: one year
# lons2=[float(ii.split()[23]) for ii in lines]
# deps2=[float(ii.split()[24]) for ii in lines]
lats2=[float(ii.split()[7]) for ii in lines if float(ii.split()[13])>=2] #oyr: one year reloc
lons2=[float(ii.split()[8]) for ii in lines if float(ii.split()[13])>=2]
deps2=[float(ii.split()[9]) for ii in lines if float(ii.split()[13])>=2]
mags2=[float(ii.split()[10]) for ii in lines if float(ii.split()[13])>=2]

## From depth-corrected EQCCT catalog
fname='../data/catalogs/delaware202210xml_growclust-cat-cc0.65-ngood6.txt'
fid=open(fname,'r')
lines=fid.readlines()
# lats2=[float(ii.split()[22]) for ii in lines] #oy: one year
# lons2=[float(ii.split()[23]) for ii in lines]
# deps2=[float(ii.split()[24]) for ii in lines]
lats3=[float(ii.split()[7]) for ii in lines if float(ii.split()[13])>=2] #oyr: one year reloc
lons3=[float(ii.split()[8]) for ii in lines if float(ii.split()[13])>=2]
deps3=[float(ii.split()[9]) for ii in lines if float(ii.split()[13])>=2]
mags3=[float(ii.split()[10]) for ii in lines if float(ii.split()[13])>=2]
mg=mags3
mg=np.array(mg,dtype='float')
mg=(mg-mg.min())/mg.max()*20
utcs3=[utc.UTCDateTime(int(ii.split()[0]), int(ii.split()[1]),  int(ii.split()[2]), int(ii.split()[3]), int(ii.split()[4]),  int(ii.split()[5].split('.')[0]), int(ii.split()[5].split('.')[1])) for ii in lines if float(ii.split()[13])>=2]
t=utcs3
t=np.array(t,dtype='float');
plt.gca().text(lat1-0.04,-0.5,'(d)',size=20,weight='normal')

ax = plt.subplot(3,2,5)
## Initial
l1=plt.scatter(lons1,deps1,10,marker='s',facecolor='blue',edgecolor='k',alpha=0.5,label='Initial')
plt.gca().set_ylim(ymin=0,ymax=10);
plt.gca().invert_yaxis();
plt.gca().set_xlim(xmin=lon1,xmax=lon2)
## Depth-corrected
l3=plt.scatter(lons3,deps3,s=mg,c=t,cmap='spring',marker='o',facecolor='blue',edgecolor='k',alpha=0.9,label='Depth-corrected')
# cbar=plt.colorbar(l3,ticks=[t.min(), (t.min()+t.max())/2, t.max()])
# cbar.ax.set_yticklabels(['Oct 1 2022', 'Oct 16 2022', 'Oct 31 2022'])
## TexNet
l2=plt.scatter(lons2,deps2,20,marker='s',facecolor='red',edgecolor='k',alpha=0.9,label='TexNet')
plt.gca().set_ylim(ymin=0,ymax=10);
plt.gca().invert_yaxis();
plt.gca().set_xlim(xmin=lon1,xmax=lon2)
plt.xlabel('Longitude (deg)');
plt.ylabel('Depth (km)');
plt.gca().legend(loc='lower left', fontsize = 8)
plt.gca().text(lon1-0.1,-0.5,'(e)',size=20,weight='normal')

ax = plt.subplot(3,2,6)
## Initial
l1=plt.scatter(lats1,deps1,10,marker='s',facecolor='blue',edgecolor='k',alpha=0.5,label='Initial')
plt.gca().set_ylim(ymin=0,ymax=10);
plt.gca().invert_yaxis();
plt.gca().set_xlim(xmin=lat1,xmax=lat2)
## Depth-corrected
l3=plt.scatter(lats3,deps3,s=mg,c=t,cmap='spring',marker='o',facecolor='blue',edgecolor='k',alpha=0.9,label='Depth-corrected')
cbar=plt.colorbar(l3,ticks=[t.min(), (t.min()+t.max())/2, t.max()])
cbar.ax.set_yticklabels(['Oct 1 2022', 'Oct 16 2022', 'Oct 31 2022'])
## TexNet
l2=plt.scatter(lats2,deps2,20,marker='s',facecolor='red',edgecolor='k',alpha=0.9,label='TexNet')
plt.gca().set_ylim(ymin=0,ymax=10);
plt.gca().invert_yaxis();
plt.gca().set_xlim(xmin=lat1,xmax=lat2)
plt.xlabel('Latitude (deg)');
plt.ylabel('Depth (km)');
# plt.gca().legend(loc='lower left', fontsize = 8)
plt.gca().text(lat1-0.04,-0.5,'(f)',size=20,weight='normal')


plt.savefig('delaware202210depthreloc.png',format='png',dpi=300,bbox_inches='tight')
plt.savefig('delaware202210depthreloc.pdf',format='pdf',dpi=300,bbox_inches='tight')


plt.show()








