import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
# pip install matplotlib-scalebar
from matplotlib_scalebar.scalebar import ScaleBar
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

## 2022/08
## From depth-corrected EQCCT catalog
fname='../data/catalogs/delaware202208xml_growclust-cat-cc0.65-ngood6.txt'

fid=open(fname,'r')
lines=fid.readlines()

## Magnitude
mags=np.array([float(ii.split()[10]) for ii in lines])
nb=np.array([float(ii.split()[13]) for ii in lines])

## Relocated 
lats=[float(ii.split()[7]) for ii in lines]
lons=[float(ii.split()[8]) for ii in lines]
deps=[float(ii.split()[9]) for ii in lines]

## Catalog
lats1=[float(ii.split()[22]) for ii in lines]
lons1=[float(ii.split()[23]) for ii in lines]
deps1=[float(ii.split()[24]) for ii in lines]

## Relocation refined
lats2=[float(ii.split()[7]) for ii in lines if float(ii.split()[13])>=2]
lons2=[float(ii.split()[8]) for ii in lines if float(ii.split()[13])>=2]
deps2=[float(ii.split()[9]) for ii in lines if float(ii.split()[13])>=2]

## 2022/09
## From depth-corrected EQCCT catalog
fname='../data/catalogs/delaware202209xml_growclust-cat-cc0.65-ngood6.txt'
fid=open(fname,'r')
lines=fid.readlines()
lats11=[float(ii.split()[22]) for ii in lines]
lons11=[float(ii.split()[23]) for ii in lines]
deps11=[float(ii.split()[24]) for ii in lines]

lats22=[float(ii.split()[7]) for ii in lines if float(ii.split()[13])>=2]
lons22=[float(ii.split()[8]) for ii in lines if float(ii.split()[13])>=2]
deps22=[float(ii.split()[9]) for ii in lines if float(ii.split()[13])>=2]

## 2022/10
## From depth-corrected EQCCT catalog
fname='../data/catalogs/delaware202210xml_growclust-cat-cc0.65-ngood6.txt'
fid=open(fname,'r')
lines=fid.readlines()
lats111=[float(ii.split()[22]) for ii in lines]
lons111=[float(ii.split()[23]) for ii in lines]
deps111=[float(ii.split()[24]) for ii in lines]

lats222=[float(ii.split()[7]) for ii in lines if float(ii.split()[13])>=2]
lons222=[float(ii.split()[8]) for ii in lines if float(ii.split()[13])>=2]
deps222=[float(ii.split()[9]) for ii in lines if float(ii.split()[13])>=2]

## Texnet 1 year
fname='../data/catalogs/texnet_events_20211201_20221201_growclust-cat-cc0.65-ngood6.txt'
fid=open(fname,'r')
lines=fid.readlines()
latsoy=[float(ii.split()[22]) for ii in lines] #oy: one year
lonsoy=[float(ii.split()[23]) for ii in lines]
depsoy=[float(ii.split()[24]) for ii in lines]

latsoyr=[float(ii.split()[7]) for ii in lines if float(ii.split()[13])>=2] #oyr: one year reloc
lonsoyr=[float(ii.split()[8]) for ii in lines if float(ii.split()[13])>=2]
depsoyr=[float(ii.split()[9]) for ii in lines if float(ii.split()[13])>=2]

## From Relocated TexNet Catalog
# Texnet 1 month (10)
fname='../data/catalogs/texnet_events_20221001_20221101_growclust-cat-cc0.65-ngood6.txt'
fid=open(fname,'r')
lines=fid.readlines()
latsom10=[float(ii.split()[22]) for ii in lines] #oy: one year
lonsom10=[float(ii.split()[23]) for ii in lines]
depsom10=[float(ii.split()[24]) for ii in lines]
latsomr10=[float(ii.split()[7]) for ii in lines if float(ii.split()[13])>=2] #oyr: one year reloc
lonsomr10=[float(ii.split()[8]) for ii in lines if float(ii.split()[13])>=2]
depsomr10=[float(ii.split()[9]) for ii in lines if float(ii.split()[13])>=2]

# Texnet 1 month (09)
fname='../data/catalogs/texnet_events_20220901_20221001_growclust-cat-cc0.65-ngood6.txt'
fid=open(fname,'r')
lines=fid.readlines()
latsom09=[float(ii.split()[22]) for ii in lines] #oy: one year
lonsom09=[float(ii.split()[23]) for ii in lines]
depsom09=[float(ii.split()[24]) for ii in lines]
latsomr09=[float(ii.split()[7]) for ii in lines if float(ii.split()[13])>=2] #oyr: one year reloc
lonsomr09=[float(ii.split()[8]) for ii in lines if float(ii.split()[13])>=2]
depsomr09=[float(ii.split()[9]) for ii in lines if float(ii.split()[13])>=2]


# Texnet 1 month (08)
fname='../data/catalogs/texnet_events_20220801_20220901_growclust-cat-cc0.65-ngood6.txt'
fid=open(fname,'r')
lines=fid.readlines()
latsom08=[float(ii.split()[22]) for ii in lines] #oy: one year
lonsom08=[float(ii.split()[23]) for ii in lines]
depsom08=[float(ii.split()[24]) for ii in lines]
latsomr08=[float(ii.split()[7]) for ii in lines if float(ii.split()[13])>=2] #oyr: one year reloc
lonsomr08=[float(ii.split()[8]) for ii in lines if float(ii.split()[13])>=2]
depsomr08=[float(ii.split()[9]) for ii in lines if float(ii.split()[13])>=2]

## AOI
lon1=-104.7-0.1
lon2=-103.7+0.1
lat1=31.3-0.05
lat2=31.9+0.05

plt.figure(figsize=(12,6))
ax = plt.subplot(2,2,1)
plt.plot(lonsoy,latsoy,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.title('2021/12-2022/12 TexNet (%d events)'%(len(latsoy)))
ax.set_aspect(1)
beach2022wmmd=beach(nop, xy=(lonc+0.05,latc+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2022wmmd)
beach2020galz=beach(nopm, xy=(lonm+0.05,latm+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2020galz)  
plt.plot(lonm,latm,'*g',markersize=10);plt.text(lonm,latm,'M4.9',color='g',fontsize=10);
plt.plot(lonc,latc,'*g',markersize=10);plt.text(lonc,latc,'M5.4',color='g',fontsize=10);
addcounty()

ax = plt.subplot(2,2,3)
plt.plot(lonsoyr,latsoyr,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=True)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
plt.title('2021/12-2022/12 TexNet Relocated (%d events)'%(len(latsoyr)))
ax.set_aspect(1)
beach2022wmmd=beach(nop, xy=(lonc+0.05,latc+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2022wmmd)
beach2020galz=beach(nopm, xy=(lonm+0.05,latm+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2020galz)  
plt.plot(lonm,latm,'*g',markersize=10);plt.text(lonm,latm,'M4.9',color='g',fontsize=10);
plt.plot(lonc,latc,'*g',markersize=10);plt.text(lonc,latc,'M5.4',color='g',fontsize=10);
addcounty()

ax = plt.subplot(2,2,2)
plt.plot(lons11,lats11,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/09-2022/10 EQCCT (%d events)'%(len(lats11)))
ax.set_aspect(1)
beach2022wmmd=beach(nop, xy=(lonc+0.05,latc+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2022wmmd)
beach2020galz=beach(nopm, xy=(lonm+0.05,latm+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2020galz)  
plt.plot(lonm,latm,'*g',markersize=10);plt.text(lonm,latm,'M4.9',color='g',fontsize=10);
plt.plot(lonc,latc,'*g',markersize=10);plt.text(lonc,latc,'M5.4',color='g',fontsize=10);
addcounty()

ax = plt.subplot(2,2,4)
plt.plot(lons22,lats22,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=True)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/09-2022/10 Relocated (%d events)'%(len(lats22)))
ax.set_aspect(1)
beach2022wmmd=beach(nop, xy=(lonc+0.05,latc+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2022wmmd)
beach2020galz=beach(nopm, xy=(lonm+0.05,latm+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2020galz)  
plt.plot(lonm,latm,'*g',markersize=10);plt.text(lonm,latm,'M4.9',color='g',fontsize=10);
plt.plot(lonc,latc,'*g',markersize=10);plt.text(lonc,latc,'M5.4',color='g',fontsize=10);
addcounty()
plt.savefig('test_eqcct_relo_manualvsAI_oneyear.png',format='png',dpi=300)
plt.show()




plt.figure(figsize=(12,6))
ax = plt.subplot(2,2,1)
plt.plot(lonsom10,latsom10,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/10-2022/11 TexNet (%d events)'%(len(latsom10)))
ax.set_aspect(1)
beach2022wmmd=beach(nop, xy=(lonc+0.05,latc+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2022wmmd)
beach2020galz=beach(nopm, xy=(lonm+0.05,latm+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2020galz)  
plt.plot(lonm,latm,'*g',markersize=10);plt.text(lonm,latm,'M4.9',color='g',fontsize=10);
plt.plot(lonc,latc,'*g',markersize=10);plt.text(lonc,latc,'M5.4',color='g',fontsize=10);
addcounty()

ax = plt.subplot(2,2,3)
plt.plot(lonsomr10,latsomr10,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=True)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/10-2022/11 TexNet Relocated (%d events)'%(len(latsomr10)))
ax.set_aspect(1)
beach2022wmmd=beach(nop, xy=(lonc+0.05,latc+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2022wmmd)
beach2020galz=beach(nopm, xy=(lonm+0.05,latm+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2020galz)  
plt.plot(lonm,latm,'*g',markersize=10);plt.text(lonm,latm,'M4.9',color='g',fontsize=10);
plt.plot(lonc,latc,'*g',markersize=10);plt.text(lonc,latc,'M5.4',color='g',fontsize=10);
addcounty()

ax = plt.subplot(2,2,2)
plt.plot(lons111,lats111,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/10-2022/11 EQCCT (%d events)'%(len(lats111)))
ax.set_aspect(1)
beach2022wmmd=beach(nop, xy=(lonc+0.05,latc+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2022wmmd)
beach2020galz=beach(nopm, xy=(lonm+0.05,latm+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2020galz)  
plt.plot(lonm,latm,'*g',markersize=10);plt.text(lonm,latm,'M4.9',color='g',fontsize=10);
plt.plot(lonc,latc,'*g',markersize=10);plt.text(lonc,latc,'M5.4',color='g',fontsize=10);
addcounty()

ax = plt.subplot(2,2,4)
plt.plot(lons222,lats222,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=True)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/10-2022/11 Relocated (%d events)'%(len(lats222)))
ax.set_aspect(1)
beach2022wmmd=beach(nop, xy=(lonc+0.05,latc+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2022wmmd)
beach2020galz=beach(nopm, xy=(lonm+0.05,latm+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2020galz)  
plt.plot(lonm,latm,'*g',markersize=10);plt.text(lonm,latm,'M4.9',color='g',fontsize=10);
plt.plot(lonc,latc,'*g',markersize=10);plt.text(lonc,latc,'M5.4',color='g',fontsize=10);
addcounty()
plt.savefig('test_eqcct_relo_manualvsAI_locvsreloc_onemonth.png',format='png',dpi=300)
plt.show()


plt.figure(figsize=(12,9))
ax = plt.subplot(3,2,1)
plt.plot(lonsomr08,latsomr08,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/08-2022/09 TexNet (%d events)'%(len(latsomr08)))
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
plt.plot(lons2,lats2,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/08-2022/09 EQCCT (%d events)'%(len(lats2)))
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
plt.plot(lonsomr09,latsomr09,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/09-2022/10 TexNet (%d events)'%(len(latsomr09)))
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
plt.plot(lons22,lats22,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/09-2022/10 EQCCT (%d events)'%(len(lats22)))
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
plt.plot(lonsomr10,latsomr10,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
# plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/10-2022/11 TexNet (%d events)'%(len(latsomr10)))
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
plt.plot(lons222,lats222,'.',color='k',markersize=1)
plt.gca().set_ylim(ymin=lat1,ymax=lat2);
plt.gca().set_xlim(xmin=lon1,xmax=lon2);
# plt.setp(ax.get_xticklabels(), visible=False)
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
plt.title('2022/10-2022/11 EQCCT (%d events)'%(len(lats222)))
ax.set_aspect(1)
beach2022wmmd=beach(nop, xy=(lonc+0.05,latc+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2022wmmd)
beach2020galz=beach(nopm, xy=(lonm+0.05,latm+0.05), width=0.05, facecolor='r', linewidth=1)
ax.add_collection(beach2020galz)  
plt.plot(lonm,latm,'*g',markersize=10);plt.text(lonm,latm,'M4.9',color='g',fontsize=10);
plt.plot(lonc,latc,'*g',markersize=10);plt.text(lonc,latc,'M5.4',color='g',fontsize=10);
addcounty()
plt.gca().text(lon1-0.3,lat2+0.05,'(f)',size=20,weight='normal')

plt.savefig('test_eqcct_relo_manualvsAI.png',format='png',dpi=300)
plt.show()

