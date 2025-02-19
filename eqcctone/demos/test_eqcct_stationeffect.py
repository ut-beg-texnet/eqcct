## Prepared by Yangkang Chen
# Originally in 2023
# Cleaned up in 2024

import numpy as np
import glob,os,obspy

from eqcct.io import asciiread
from eqcct.texnet import get_sta_loc
from eqcct.texnet import get_sta_time

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

stafile='../data/stations/texnet_stations_2022_1019.csv'

lines0=asciiread('../data/catalogs/stationeffect/EQCCTNLL_catalog_202210sta21.dat')
names=[ii.split()[-1].split(":") for ii in lines0 if len(ii.split()[-1].split(":")) == 21][0]


lonlats=get_sta_loc(names,stafile=stafile)
times=get_sta_time(names,stafile=stafile)
print(times)

lons=[float(ii[0]) for ii in lonlats]
lats=[float(ii[1]) for ii in lonlats]

## Sorting according installation time
inds=np.argsort(np.array(times))
names=[names[ii] for ii in inds]
times=[float(times[ii]) for ii in inds]
lons=[lons[ii] for ii in inds]
lats=[lats[ii] for ii in inds]
t=np.array(times);

lonsr=lons;
latsr=lats;

## assume only 0.8 stations are there
nsta=int(0.7*len(names))
stanames=names[:nsta]



### plot station-effect figures

def addcounty():
##red county lines
	f=open('../data/texas/Culberson_XY.txt','r')
	# f=open('Culberson_XY.txt','r')
	clines=f.readlines()
	clines=[ii.rstrip().split(' ') for ii in clines]
	##add county lines
	tmp=[]
	for ii in range(len(clines)):
		if clines[ii] != ['']:
			tmp.append(clines[ii])
		else:
			tmp=np.array(tmp,dtype='float32')
			plt.plot(tmp[:,0],tmp[:,1],'-',color='#929591')
			ii=ii+1;
			tmp=[];
	plt.text(-104.4,31.3,'Culberson',fontsize=10,color='#929591')
	plt.text(-104.0,31.3,'Reeves',fontsize=10,color='#929591')
	scalebar = ScaleBar(110000,location='lower left');plt.gca().add_artist(scalebar)


## AOI
lon1=-104.7
lon2=-103.7
lat1=31.3
lat2=31.9

lon1=-104.7-0.1
lon2=-103.7+0.1
lat1=31.3-0.05
lat2=31.9+0.05

edge=0.1
nrow=3
ratio=2.0 #left vs right part

## Adding axes in the following ways is more flexible (and sometimes more beautiful)
fig=plt.figure(figsize=(10,8))
irow=1
ax = fig.add_axes([edge,edge+(1-edge)/nrow*(nrow-irow),(1-edge*2-edge)*ratio/(ratio+1),(1-edge-edge-(nrow-1)*edge/2)/nrow])
# ax = plt.subplot(3,1,1)
nsta=21
lines=asciiread('../data/catalogs/stationeffect/EQCCTNLL_catalog_202210sta21.dat')
lons1=[float(ii.split()[2]) for ii in lines]
lats1=[float(ii.split()[3]) for ii in lines]

plt.scatter(lons1,lats1,1,marker='.',color='k',alpha=0.5)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latiitude (deg)",fontsize='large', fontweight='normal')

plt.title('2022/10-2022/11 %d stations %d events'%(nsta,len(lats1)))
addcounty()
l=plt.scatter(lonsr[0:nsta],latsr[0:nsta],50,c=t,vmin=t.min(),vmax=t.max(),cmap='jet',marker='v',facecolor='gray',edgecolor='k',alpha=0.8)
# for ii in range(len(names)):
#     plt.text(lonsr[ii],latsr[ii]+0.02,names[ii],color='k',horizontalalignment='center')
cbar=plt.colorbar(l,ticks=[t.min(), (t.min()+t.max())/2, t.max()])
cbar.ax.set_yticklabels(['02/01/2017', '11/27/2019', '09/23/2022'])
plt.gca().text(lon1-0.25,lat2+0.03,'(a)',size=20,weight='normal')


# ax = plt.subplot(3,1,2)
irow=2
ax = fig.add_axes([edge,edge+(1-edge)/nrow*(nrow-irow),(1-edge*2-edge)*ratio/(ratio+1),(1-edge-edge-(nrow-1)*edge/2)/nrow])

nsta=12
lines=asciiread('../data/catalogs/stationeffect/EQCCTNLL_catalog_202210sta12.dat')
lons1=[float(ii.split()[2]) for ii in lines]
lats1=[float(ii.split()[3]) for ii in lines]

plt.scatter(lons1,lats1,1,marker='.',color='k',alpha=0.5)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latiitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/10-2022/11 %d stations %d events'%(nsta,len(lats1)))
addcounty()
l=plt.scatter(lonsr[0:nsta],latsr[0:nsta],50,c=t[0:nsta],vmin=t.min(),vmax=t.max(),cmap='jet',marker='v',facecolor='gray',edgecolor='k',alpha=0.8)
cbar=plt.colorbar(l,ticks=[t.min(), (t.min()+t.max())/2, t.max()])
cbar.ax.set_yticklabels(['02/01/2017', '11/27/2019', '09/23/2022'])
plt.gca().text(lon1-0.25,lat2+0.03,'(c)',size=20,weight='normal')

# ax = plt.subplot(3,1,3)
irow=3
ax = fig.add_axes([edge,edge+(1-edge)/nrow*(nrow-irow),(1-edge*2-edge)*ratio/(ratio+1),(1-edge-edge-(nrow-1)*edge/2)/nrow])

nsta=8
lines=asciiread('../data/catalogs/stationeffect/EQCCTNLL_catalog_202210sta8.dat')
lons1=[float(ii.split()[2]) for ii in lines]
lats1=[float(ii.split()[3]) for ii in lines]

plt.scatter(lons1,lats1,1,marker='.',color='k',alpha=0.5)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
# plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latiitude (deg)",fontsize='large', fontweight='normal')
plt.gca().set_xlabel("longitude (deg)",fontsize='large', fontweight='normal')

plt.title('2022/10-2022/11 %d stations %d events'%(nsta,len(lats1)))

addcounty()
l=plt.scatter(lonsr[0:nsta],latsr[0:nsta],50,c=t[0:nsta], vmin=t.min(),vmax=t.max(),cmap='jet',marker='v',facecolor='gray',edgecolor='k',alpha=0.8)
# for ii in range(len(names)):
#     plt.text(lonsr[ii],latsr[ii]+0.02,names[ii],color='k',horizontalalignment='center')
cbar=plt.colorbar(l,ticks=[t.min(), (t.min()+t.max())/2, t.max()])
cbar.ax.set_yticklabels(['02/01/2017', '11/27/2019', '09/23/2022'])
plt.gca().text(lon1-0.25,lat2+0.03,'(e)',size=20,weight='normal')


### Mc
# plt.figure(figsize=(3,8))
# ax = plt.subplot(3,1,1)
irow=1
ax = fig.add_axes([edge+(1-edge*2-edge)*ratio/(ratio+1)+edge,edge+(1-edge)/nrow*(nrow-irow),(1-edge*2-edge)*1/(ratio+1),(1-edge-edge-(nrow-1)*edge/2)/nrow])

lines=asciiread('../data/catalogs/stationeffect/EQCCTNLL_catalog_202210sta21.dat')

mags1=[float(ii.split()[5]) for ii in lines]
plt.hist(mags1,30,range=(np.min(mags1),np.max(mags1)),label='EQCCT (one month)',color='lightgray',edgecolor='black',log=True)
plt.gca().set_xlim(xmin=0,xmax=4);
plt.gca().set_ylim(ymin=0,ymax=2500);
plt.gca().legend(loc='lower left');
plt.gca().set_ylabel("Count",fontsize='large', fontweight='normal')
plt.gca().text(-1,3500,'(b)',size=20,weight='normal')


# ax = plt.subplot(3,1,2)
irow=2
ax = fig.add_axes([edge+(1-edge*2-edge)*ratio/(ratio+1)+edge,edge+(1-edge)/nrow*(nrow-irow),(1-edge*2-edge)*1/(ratio+1),(1-edge-edge-(nrow-1)*edge/2)/nrow])

lines=asciiread('../data/catalogs/stationeffect/EQCCTNLL_catalog_202210sta12.dat')
mags1=[float(ii.split()[5]) for ii in lines]
plt.hist(mags1,30,range=(np.min(mags1),np.max(mags1)),label='EQCCT (one month)',color='lightgray',edgecolor='black',log=True)
plt.gca().set_xlim(xmin=0,xmax=4);
plt.gca().set_ylim(ymin=0,ymax=2500);
plt.gca().legend(loc='lower left');
plt.gca().set_ylabel("Count",fontsize='large', fontweight='normal')
# plt.gca().set_xlabel("Magnitude (Ml)",fontsize='large', fontweight='normal')
plt.gca().text(-1,3500,'(d)',size=20,weight='normal')


# ax = plt.subplot(3,1,3)
irow=3
ax = fig.add_axes([edge+(1-edge*2-edge)*ratio/(ratio+1)+edge,edge+(1-edge)/nrow*(nrow-irow),(1-edge*2-edge)*1/(ratio+1),(1-edge-edge-(nrow-1)*edge/2)/nrow])

lines=asciiread('../data/catalogs/stationeffect/EQCCTNLL_catalog_202210sta8.dat')
mags1=[float(ii.split()[5]) for ii in lines]
plt.hist(mags1,30,range=(np.min(mags1),np.max(mags1)),label='EQCCT (one month)',color='lightgray',edgecolor='black',log=True)
plt.gca().set_xlim(xmin=0,xmax=4);
plt.gca().set_ylim(ymin=0,ymax=2500);
plt.gca().legend(loc='lower left');
plt.gca().set_ylabel("Count",fontsize='large', fontweight='normal')
plt.gca().set_xlabel("Magnitude (Ml)",fontsize='large', fontweight='normal')
plt.gca().text(-1,3500,'(f)',size=20,weight='normal')

plt.savefig('test_eqcct_stationeffect.png',format='png',dpi=300,bbox_inches='tight')

plt.show()

