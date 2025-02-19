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
	

# lines=asciiread('../data/catalogs/EQCCTNLL_catalog_202208.dat')
# lonsom08=[float(ii.split()[2]) for ii in lines]
# latsom08=[float(ii.split()[3]) for ii in lines]
# depsom08=[float(ii.split()[4]) for ii in lines]
# magsom08=[float(ii.split()[5]) for ii in lines]
# utcsom08=[utc.UTCDateTime(ii.split()[1]) for ii in lines]
# 
# 
# lines=asciiread('../data/catalogs/EQCCTNLL_catalog_202209.dat')
# lonsom09=[float(ii.split()[2]) for ii in lines]
# latsom09=[float(ii.split()[3]) for ii in lines]
# depsom09=[float(ii.split()[4]) for ii in lines]
# magsom09=[float(ii.split()[5]) for ii in lines]
# utcsom09=[utc.UTCDateTime(ii.split()[1]) for ii in lines]
# 

lines=asciiread('../data/catalogs/EQCCTNLL_catalog_202210.dat')
lonsom10=[float(ii.split()[2]) for ii in lines]
latsom10=[float(ii.split()[3]) for ii in lines]
depsom10=[float(ii.split()[4]) for ii in lines]
magsom10=[float(ii.split()[5]) for ii in lines]
utcsom10=[utc.UTCDateTime(ii.split()[1]) for ii in lines]

# lines=asciiread('../data/catalogs/delaware_11140_events.csv') #08
# lines.pop(0);
# lons0om08=[float(ii.split(',')[9]) for ii in lines]
# lats0om08=[float(ii.split(',')[8]) for ii in lines]
# deps0om08=[float(ii.split(',')[10]) for ii in lines]
# mags0om08=[float(ii.split(',')[7]) for ii in lines]
# utcs0om08=[utc.UTCDateTime(ii.split(',')[2].split('/')[-1]) for ii in lines]
# 
# lines=asciiread('../data/catalogs/delaware_10867_events.csv') #09
# lines.pop(0);
# lons0om09=[float(ii.split(',')[9]) for ii in lines]
# lats0om09=[float(ii.split(',')[8]) for ii in lines]
# deps0om09=[float(ii.split(',')[10]) for ii in lines]
# mags0om09=[float(ii.split(',')[7]) for ii in lines]
# utcs0om09=[utc.UTCDateTime(ii.split(',')[2].split('/')[-1]) for ii in lines]
# 
# lines=asciiread('../data/catalogs/delaware_11687_events.csv') #10
# lines.pop(0);
# lons0om10=[float(ii.split(',')[9]) for ii in lines]
# lats0om10=[float(ii.split(',')[8]) for ii in lines]
# deps0om10=[float(ii.split(',')[10]) for ii in lines]
# mags0om10=[float(ii.split(',')[7]) for ii in lines]
# utcs0om10=[utc.UTCDateTime(ii.split(',')[2].split('/')[-1]) for ii in lines]



## AOI
lon1=-104.7
lon2=-103.7
lat1=31.3
lat2=31.9

lon1=-104.7-0.1
lon2=-103.7+0.1
lat1=31.3-0.05
lat2=31.9+0.05

#split the earthquakes to dates
t1=utc.UTCDateTime(2022, 8, 1, 00, 00, 00, 000000)
t2=utc.UTCDateTime(2022, 9, 1, 00, 00, 00, 000000)
t3=utc.UTCDateTime(2022, 10, 1, 00, 00, 00, 000000)
t4=utc.UTCDateTime(2022, 11, 1, 00, 00, 00, 000000)


t=t3;
inds2=[]
files=[]
ii=0
while t>=t3 and t<t4:
	dt=60*60*24
	inds=[ii for ii in range(len(utcsom10)) if utcsom10[ii]>t and utcsom10[ii]<t+dt]
	t=t+dt;ii=ii+1;
	print(t)
	
	inds2=inds2+inds
	print(len(inds))
	plt.figure(figsize=(12,6))
	ax=plt.gca()
	plt.plot([lonsom10[jj] for jj in inds2],[latsom10[jj] for jj in inds2],'.',color='k',markersize=1)
	plt.gca().set_ylim(ymin=lat1,ymax=lat2);
	plt.gca().set_xlim(xmin=lon1,xmax=lon2);
	plt.setp(ax.get_xticklabels(), visible=False)
	plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
	plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
	plt.title('%s - %s (day %d) cumulative seismicity (%d events)'%(str(t3).split('T')[0],str(t).split('T')[0],ii,len(inds2)))
	plt.plot(lonm,latm,'*g',markersize=10);plt.text(lonm,latm,'M4.9',color='g',fontsize=10);
	plt.plot(lonc,latc,'*g',markersize=10);plt.text(lonc,latc,'M5.4',color='g',fontsize=10);
	addcounty()
	file='day-%d.png'%ii
	files.append(file)
	plt.savefig(file,format='png',dpi=300)
	plt.close()
	
from pyseistr import gengif #pip install git+https://github.com/aaspip/pyseistr
gengif(files,'test_eqcct_nll_gif.gif')




