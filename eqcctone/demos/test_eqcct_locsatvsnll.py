import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from eqcct.io import asciiread
# pip install matplotlib-scalebar
from matplotlib_scalebar.scalebar import ScaleBar
import obspy.core.utcdatetime as utc
import obspy
from obspy.imaging.beachball import beach

e=obspy.read_events('../data/qmls/texnet2020galz.qml');lonm=e[0].origins[0].longitude;latm=e[0].origins[0].latitude;
nop=[110.0, 47.0, -108.0];

e=obspy.read_events('../data/qmls/texnet2022wmmd.qml');lonc=e[0].origins[0].longitude;latc=e[0].origins[0].latitude;
nopm=[96.0, 43.0, -100.0];

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
	

lines=asciiread('../data/catalogs/EQCCTNLL_catalog_202208.dat')
lonsom08=[float(ii.split()[2]) for ii in lines]
latsom08=[float(ii.split()[3]) for ii in lines]
depsom08=[float(ii.split()[4]) for ii in lines]
magsom08=[float(ii.split()[5]) for ii in lines]
utcsom08=[utc.UTCDateTime(ii.split()[1]) for ii in lines]


lines=asciiread('../data/catalogs/EQCCTNLL_catalog_202209.dat')
lonsom09=[float(ii.split()[2]) for ii in lines]
latsom09=[float(ii.split()[3]) for ii in lines]
depsom09=[float(ii.split()[4]) for ii in lines]
magsom09=[float(ii.split()[5]) for ii in lines]
utcsom09=[utc.UTCDateTime(ii.split()[1]) for ii in lines]


lines=asciiread('../data/catalogs/EQCCTNLL_catalog_202210.dat')
lonsom10=[float(ii.split()[2]) for ii in lines]
latsom10=[float(ii.split()[3]) for ii in lines]
depsom10=[float(ii.split()[4]) for ii in lines]
magsom10=[float(ii.split()[5]) for ii in lines]
utcsom10=[utc.UTCDateTime(ii.split()[1]) for ii in lines]

lines=asciiread('../data/catalogs/delaware_11140_events.csv') #08
lines.pop(0);
lons0om08=[float(ii.split(',')[9]) for ii in lines]
lats0om08=[float(ii.split(',')[8]) for ii in lines]
deps0om08=[float(ii.split(',')[10]) for ii in lines]
mags0om08=[float(ii.split(',')[7]) for ii in lines]
utcs0om08=[utc.UTCDateTime(ii.split(',')[2].split('/')[-1]) for ii in lines]

lines=asciiread('../data/catalogs/delaware_10867_events.csv') #09
lines.pop(0);
lons0om09=[float(ii.split(',')[9]) for ii in lines]
lats0om09=[float(ii.split(',')[8]) for ii in lines]
deps0om09=[float(ii.split(',')[10]) for ii in lines]
mags0om09=[float(ii.split(',')[7]) for ii in lines]
utcs0om09=[utc.UTCDateTime(ii.split(',')[2].split('/')[-1]) for ii in lines]

lines=asciiread('../data/catalogs/delaware_11687_events.csv') #10
lines.pop(0);
lons0om10=[float(ii.split(',')[9]) for ii in lines]
lats0om10=[float(ii.split(',')[8]) for ii in lines]
deps0om10=[float(ii.split(',')[10]) for ii in lines]
mags0om10=[float(ii.split(',')[7]) for ii in lines]
utcs0om10=[utc.UTCDateTime(ii.split(',')[2].split('/')[-1]) for ii in lines]

## AOI
lon1=-104.7
lon2=-103.7
lat1=31.3
lat2=31.9

lon1=-104.7-0.1
lon2=-103.7+0.1
lat1=31.3-0.05
lat2=31.9+0.05

plt.figure(figsize=(12,9))
ax = plt.subplot(3,2,1)
plt.plot(lons0om08,lats0om08,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/08-2022/09 LOCSAT (%d events)'%(len(lats0om08)))
ax.set_aspect(1)
beach2022wmmd=beach(nop, xy=(lonc+0.05,latc+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2022wmmd)
beach2020galz=beach(nopm, xy=(lonm+0.05,latm+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2020galz)  
plt.plot(lonm,latm,'*g',markersize=10);plt.text(lonm,latm,'M4.9',color='g',fontsize=10);
plt.plot(lonc,latc,'*g',markersize=10);plt.text(lonc,latc,'M5.4',color='g',fontsize=10);
addcounty()
plt.gca().text(lon1-0.3,lat2+0.05,'(a)',size=20,weight='normal')

ax = plt.subplot(3,2,2)
plt.plot(lonsom08,latsom08,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/08-2022/09 NLL (%d events)'%(len(latsom08)))
ax.set_aspect(1)
beach2022wmmd=beach(nop, xy=(lonc+0.05,latc+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2022wmmd)
beach2020galz=beach(nopm, xy=(lonm+0.05,latm+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2020galz)  
plt.plot(lonm,latm,'*g',markersize=10);plt.text(lonm,latm,'M4.9',color='g',fontsize=10);
plt.plot(lonc,latc,'*g',markersize=10);plt.text(lonc,latc,'M5.4',color='g',fontsize=10);
addcounty()
plt.gca().text(lon1-0.3,lat2+0.05,'(b)',size=20,weight='normal')

ax = plt.subplot(3,2,3)
plt.plot(lons0om09,lats0om09,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/09-2022/10 LOCSAT (%d events)'%(len(lats0om09)))
ax.set_aspect(1)
beach2022wmmd=beach(nop, xy=(lonc+0.05,latc+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2022wmmd)
beach2020galz=beach(nopm, xy=(lonm+0.05,latm+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2020galz)  
plt.plot(lonm,latm,'*g',markersize=10);plt.text(lonm,latm,'M4.9',color='g',fontsize=10);
plt.plot(lonc,latc,'*g',markersize=10);plt.text(lonc,latc,'M5.4',color='g',fontsize=10);
addcounty()
plt.gca().text(lon1-0.3,lat2+0.05,'(c)',size=20,weight='normal')

ax = plt.subplot(3,2,4)
plt.plot(lonsom09,latsom09,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/09-2022/10 NLL (%d events)'%(len(latsom09)))
ax.set_aspect(1)
beach2022wmmd=beach(nop, xy=(lonc+0.05,latc+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2022wmmd)
beach2020galz=beach(nopm, xy=(lonm+0.05,latm+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2020galz)  
plt.plot(lonm,latm,'*g',markersize=10);plt.text(lonm,latm,'M4.9',color='g',fontsize=10);
plt.plot(lonc,latc,'*g',markersize=10);plt.text(lonc,latc,'M5.4',color='g',fontsize=10);
addcounty()
plt.gca().text(lon1-0.3,lat2+0.05,'(d)',size=20,weight='normal')

ax = plt.subplot(3,2,5)
plt.plot(lons0om10,lats0om10,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
# plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/10-2022/11 LOCSAT (%d events)'%(len(lats0om10)))
ax.set_aspect(1)
beach2022wmmd=beach(nop, xy=(lonc+0.05,latc+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2022wmmd)
beach2020galz=beach(nopm, xy=(lonm+0.05,latm+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2020galz)  
plt.plot(lonm,latm,'*g',markersize=10);plt.text(lonm,latm,'M4.9',color='g',fontsize=10);
plt.plot(lonc,latc,'*g',markersize=10);plt.text(lonc,latc,'M5.4',color='g',fontsize=10);
addcounty()
plt.gca().text(lon1-0.3,lat2+0.05,'(e)',size=20,weight='normal')

ax = plt.subplot(3,2,6)
plt.plot(lonsom10,latsom10,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
# plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/10-2022/11 NLL (%d events)'%(len(lonsom10)))
ax.set_aspect(1)
beach2022wmmd=beach(nop, xy=(lonc+0.05,latc+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2022wmmd)
beach2020galz=beach(nopm, xy=(lonm+0.05,latm+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2020galz)  
plt.plot(lonm,latm,'*g',markersize=10);plt.text(lonm,latm,'M4.9',color='g',fontsize=10);
plt.plot(lonc,latc,'*g',markersize=10);plt.text(lonc,latc,'M5.4',color='g',fontsize=10);
addcounty()
plt.gca().text(lon1-0.3,lat2+0.05,'(f)',size=20,weight='normal')

plt.savefig('test_eqcct_locsatvsnll.png',format='png',dpi=300)
plt.show()

