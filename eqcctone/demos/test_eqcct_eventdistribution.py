
import numpy as np
import sys,os
from eqcct.texnet import create_catalog,get_sta_loc,get_sta_locelv
from eqcct.io import asciiread,asciiwrite
import obspy.core.utcdatetime as utc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

if3month=False

if if3month==True: #three-month version or one-month version
	## Read EQCCT result
	f=open('../data/catalogs/delaware_11140_events.csv') #202008
	lines1=f.readlines()
	f=open('../data/catalogs/delaware_10867_events.csv') #202009
	lines2=f.readlines()
	f=open('../data/catalogs/delaware_11687_events.csv') #202010
	lines3=f.readlines()
	lines1=lines1[1:]
	lines2=lines2[1:]
	lines3=lines3[1:]
	lines=lines1+lines2+lines3;
else:
	f=open('../data/catalogs/delaware_11687_events.csv') #202010
	lines=f.readlines()[1:]

## Read reported events
fname='../data/texnetreported/texnet__20170101-20230315_on20230315.asc'

lons1=[float(ii.split(',')[9]) for ii in lines]
lats1=[float(ii.split(',')[8]) for ii in lines]
deps1=[float(ii.split(',')[10]) for ii in lines]
ids1=[ii.split(',')[0].split('/')[-1] for ii in lines]

mags1=[float(ii.split(',')[7]) for ii in lines]
rmss1=[float(ii.split(',')[-1]) for ii in lines]

utcs1=[utc.UTCDateTime(ii.split(',')[6].split(' ')[0]+'T'+ii.split(',')[6].split('+')[0].split(' ')[1]) for ii in lines]

t=[]
for ii in rmss1:
	if ii<1.0:
		t.append(ii)

x=[];
y=[];

mag1=1.0
for ii in range(len(lons1)):
	if mags1[ii] >= mag1:
		x.append(lons1[ii])
		y.append(lats1[ii])

for ii in range(len(lons1)):
	if rmss1[ii] <= 0.2:
		x.append(lons1[ii])
		y.append(lats1[ii])

if if3month == True:
	nmonths=[12, 6, 1, 3]
else:
	nmonths=[12, 6, 3, 1]

for nmonth in nmonths:
    #Midland: -102.0779,31.9973
    xM=-102.0779
    yM=31.9973
    #Odessa: 
    # xO=-102.3676
    # yO=31.8457
    #Pecos:
    xP=-103.4932;
    yP=31.4229;
    #Van Horn
    xV=-104.8308
    yV=31.0399
    #Orla
    xO_Orla=-103.9085
    yO_Orla=31.8248
    
    f=open(fname)
    lines=f.readlines()
    lines=lines[1:-8]
    lats=[float(ii.split()[8]) for ii in lines]
    lons=[float(ii.split()[9]) for ii in lines]
    deps=[float(ii.split()[-1]) for ii in lines]
    eid0=[ii.split()[0] for ii in lines]
    flags=[ii.split()[3] for ii in lines]
    utcs=[utc.UTCDateTime(ii.split('  ')[1]) for ii in lines]
    
    if nmonth == 12:
        t1=utc.UTCDateTime(2021, 12, 1, 00, 00, 00, 000000)
        t2=utc.UTCDateTime(2022, 12, 1, 00, 00, 00, 000000)
    if nmonth == 1:
        t1=utc.UTCDateTime(2022, 10, 1, 00, 00, 00, 000000)
        t2=utc.UTCDateTime(2022, 11, 1, 00, 00, 00, 000000)  
    if nmonth == 6:
        t1=utc.UTCDateTime(2022, 6,  1, 00, 00, 00, 000000)
        t2=utc.UTCDateTime(2022, 12, 1, 00, 00, 00, 000000)   
    if nmonth == 3:
        t1=utc.UTCDateTime(2022, 8,  1, 00, 00, 00, 000000)
        t2=utc.UTCDateTime(2022, 11, 1, 00, 00, 00, 000000) 
        
    lon1=-104.7
    lon2=-103.7
    
    lat1=31.3
    lat2=31.9
    
    eids=[]
    lines3=[]
    for ii in range(len(lines)):
        if utcs[ii]>=t1 and utcs[ii]<=t2 and lons[ii]>=lon1 and lons[ii]<=lon2 and lats[ii]>=lat1 and lats[ii]<=lat2:
            eids.append(eid0[ii])
            lines3.append(lines[ii])
    print('Number of Reported events in AOI is',len(eids))
    
    ### plot Reported events
    lats3=[float(ii.split()[8]) for ii in lines3]
    lons3=[float(ii.split()[9]) for ii in lines3]
    deps3=[float(ii.split()[-1]) for ii in lines3]
    mags3=[float(ii.split()[7]) for ii in lines3]
    utcs3=[utc.UTCDateTime(ii.split('  ')[1]) for ii in lines3]
    
    eids=[]
    lines2=[]
    for ii in range(len(lines)):
        if utcs[ii]>=t1 and utcs[ii]<=t2 and lons[ii]>=lon1 and lons[ii]<=lon2 and lats[ii]>=lat1 and lats[ii]<=lat2 and (flags[ii] != 'REPORTED'):
            eids.append(eid0[ii])
            lines2.append(lines[ii])
    print('Number of Catalog events in AOI is',len(eids))
    create_catalog(eids,fincatalog='../data/catalogs/texnet_events_20221220.csv',foutcatalog='texnet_events_tmp.csv');
    
    ### plot Catalog events
    f=open('texnet_events_tmp.csv')
    lines=f.readlines()
    lines=lines[1:]
    lats2=[float(ii.split(',')[6]) for ii in lines]
    lons2=[float(ii.split(',')[8]) for ii in lines]
    deps2=[float(ii.split(',')[10]) for ii in lines]
    mags2=[float(ii.split(',')[4]) for ii in lines]
    rmss2=[float(ii.split(',')[13]) for ii in lines]
    utcs2=[utc.UTCDateTime(ii.split(',')[2]+'T'+ii.split(',')[3]) for ii in lines]
        
    #%% compute the density of EQCCT events
    elon_p=x
    elat_p=y
    
    gridsize=0.025
    lons=np.arange(np.floor(min(elon_p)),np.ceil(max(elon_p)),gridsize)
    lats=np.arange(np.floor(min(elat_p)),np.ceil(max(elat_p)),gridsize)
    
    [LON,LAT]=np.meshgrid(lons,lats)
    V=np.nan*np.ones_like(LON)
    S=np.nan*np.ones_like(LON)
    index_hv=[]
    radii=gridsize*.5
    
    D=np.nan*np.ones_like(LON)
    # assign value to grid nodes
    for i,lat in enumerate(lats):
        print(i)
        for j,lon in enumerate(lons):
    #        print(j)
            glon0=lon-radii;
            glon1=lon+radii;
            glat0=lat-radii;
            glat1=lat+radii;
            keep=np.where((elon_p>=glon0) & (elon_p<=glon1) & (elat_p>=glat0) & (elat_p<=glat1))
            D[i,j]=len(keep[0])

    if nmonth == 12:
        fig=plt.figure(figsize=(14,10))
        ax4=fig.add_subplot(224)
        plt.pcolor(LON,LAT,D,cmap='pink_r',vmin=0,vmax=20)
        cb=plt.colorbar()
        cb.set_label('Count')
        # plt.plot(x,y,'r.',label='EQCCT');
        l1=plt.scatter(lons2,lats2,10,facecolor='red',edgecolor='k',alpha=0.9)
        # plt.xlim([-104.7,-103.6])
        # plt.ylim([31.3,31.9])
        
        plt.gca().set_ylim(ymin=lat1,ymax=lat2)
        plt.gca().set_xlim(xmin=lon1,xmax=lon2)
        
        plt.plot(xO_Orla,yO_Orla,'*b',markersize=12);
        plt.text(xO_Orla,yO_Orla+0.025,'Orla, TX',color='b',fontsize=14,horizontalalignment='center')
        
        plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
        plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
        plt.legend({l1,},{'Catalog'},loc='lower right')
    
        plt.gca().set_title("2367 Catalog events, Period: 2021/12/01-2022/12/01",fontsize='large', fontweight='normal')
        ax4.text(-0.25,1,'(d)',transform=ax4.transAxes,size=20,weight='normal')
    
    if nmonth == 6:
        ax3=fig.add_subplot(223)
        plt.pcolor(LON,LAT,D,cmap='pink_r',vmin=0,vmax=20)
        cb=plt.colorbar()
        cb.set_label('Count')
        # plt.plot(x,y,'r.',label='EQCCT');
        l1=plt.scatter(lons2,lats2,10,facecolor='red',edgecolor='k',alpha=0.9)
        # plt.xlim([-104.7,-103.6])
        # plt.ylim([31.3,31.9])
        
        plt.gca().set_ylim(ymin=lat1,ymax=lat2)
        plt.gca().set_xlim(xmin=lon1,xmax=lon2)
        
        plt.plot(xO_Orla,yO_Orla,'*b',markersize=12);
        plt.text(xO_Orla,yO_Orla+0.025,'Orla, TX',color='b',fontsize=14,horizontalalignment='center')
        
        plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
        plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
        plt.legend({l1,},{'Catalog'},loc='lower right')
        
        plt.gca().set_title("1379 Catalog events, Period: 2022/06/01-2022/12/01",fontsize='large', fontweight='normal')
        ax3.text(-0.25,1,'(c)',transform=ax3.transAxes,size=20,weight='normal')
        
    if nmonth == 3:
        ax2=fig.add_subplot(222)
        plt.pcolor(LON,LAT,D,cmap='pink_r',vmin=0,vmax=20)
        cb=plt.colorbar()
        cb.set_label('Count')
        # plt.plot(x,y,'r.',label='EQCCT');
        l1=plt.scatter(lons2,lats2,10,facecolor='red',edgecolor='k',alpha=0.9)
        # plt.xlim([-104.7,-103.6])
        # plt.ylim([31.3,31.9])
        
        plt.gca().set_ylim(ymin=lat1,ymax=lat2)
        plt.gca().set_xlim(xmin=lon1,xmax=lon2)
        
        plt.plot(xO_Orla,yO_Orla,'*b',markersize=12);
        plt.text(xO_Orla,yO_Orla+0.025,'Orla, TX',color='b',fontsize=14,horizontalalignment='center')
        
        plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
        plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
        plt.legend({l1,},{'Catalog'},loc='lower right')
        
        plt.gca().set_title("567 Catalog events, Period: 2022/08/01-2022/11/01",fontsize='large', fontweight='normal')
        ax2.text(-0.25,1,'(b)',transform=ax2.transAxes,size=20,weight='normal')
        
    if nmonth == 1:
        ax1=fig.add_subplot(221)
        plt.pcolor(LON,LAT,D,cmap='pink_r',vmin=0,vmax=20)
        cb=plt.colorbar()
        cb.set_label('Count')
        # plt.plot(x,y,'r.',label='EQCCT');
        l1=plt.scatter(lons2,lats2,10,facecolor='red',edgecolor='k',alpha=0.9)
        # plt.xlim([-104.7,-103.6])
        # plt.ylim([31.3,31.9])
        
        plt.gca().set_ylim(ymin=lat1,ymax=lat2)
        plt.gca().set_xlim(xmin=lon1,xmax=lon2)
        
        plt.plot(xO_Orla,yO_Orla,'*b',markersize=12);
        plt.text(xO_Orla,yO_Orla+0.025,'Orla, TX',color='b',fontsize=14,horizontalalignment='center')
        
        plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
        plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
        plt.legend({l1,},{'Catalog'},loc='lower right')
        
        plt.gca().set_title("215 Catalog events, Period: 2022/10/01-2022/11/01",fontsize='large', fontweight='normal')
        ax1.text(-0.25,1,'(a)',transform=ax1.transAxes,size=20,weight='normal')

plt.savefig('test_eqcct_eventdistribution1.png',format='png',dpi=300,bbox_inches='tight')
plt.show()

#Midland: -102.0779,31.9973
xM=-102.0779
yM=31.9973


#Odessa: 
xO=-102.3676
yO=31.8457
#%% Plot Figure 10
#AOI
lon1=-104.7
lon2=-103.7

lat1=31.5
lat2=31.9


stafile='../data/stations/texnet_stations_2022_1019.csv'
lines=asciiread(stafile)[1:]
stnames = [line.strip().split(',')[0]+'.'+line.strip().split(',')[1] for line in lines]

lonlats=get_sta_loc(stnames,stafile=stafile)


lines=[]
for ii in range(len(lonlats)):
# 	print('stnames is',stnames[ii])
	lon=float(lonlats[ii][0])
	lat=float(lonlats[ii][1])
	if lon>=lon1 and lon<=lon2 and lat>=lat1 and lat<=lat2:
		lines.append(stnames[ii]+' '+str(lon)+' '+str(lat))

lons=[float(ii.split(" ")[1]) for ii in lines];
lats=[float(ii.split(" ")[2].split("\n")[0]) for ii in lines];
names=[ii.split(" ")[0] for ii in lines];


if if3month == True:
	period='Three months'
else:
	period='One month'

fig=plt.figure(figsize=(14,10))
ax1=fig.add_subplot(221)
for ii in range(len(lines)):
	plt.scatter(lons[ii],lats[ii],200,marker='v',facecolor='gray',edgecolor='k')
		
for ii in range(len(names)):
    plt.text(lons[ii],lats[ii]+0.02,names[ii],color='k',horizontalalignment='center')

plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
ax1.text(-0.15,1,'(a)',transform=ax1.transAxes,size=20,weight='normal')
   
ax2=fig.add_subplot(222)
print("mags1:",len(mags1),"mags2",len(mags2),"mags3",len(mags2))
plt.hist(mags1,30,range=(np.min(mags1),np.max(mags1)),label='EQCCT (%s)'%period,color='lightgray',edgecolor='black',log=True)
plt.hist(mags3,30,range=(np.min(mags1),np.max(mags1)),label='Reported (%s)'%period,color='lightyellow',edgecolor='black',log=True)
plt.hist(mags2,30,range=(np.min(mags1),np.max(mags1)),label='Catalog (%s)'%period,color='red',edgecolor='black',log=True)
plt.gca().set_xlim(xmin=0,xmax=4);
plt.gca().legend(loc='upper right');
plt.gca().set_ylabel("Count",fontsize='large', fontweight='normal')
plt.gca().set_xlabel("Magnitude (Ml)",fontsize='large', fontweight='normal')
ax2.text(-0.15,1,'(b)',transform=ax2.transAxes,size=20,weight='normal')

ax3=fig.add_subplot(223)
print("deps1:",len(deps1),"deps2",len(deps2))

plt.hist(deps1,100,range=(np.min(deps1),np.max(deps1)),label='EQCCT (%s)'%period,color='lightgray',edgecolor='black',log=True)
plt.hist(deps3,100,range=(np.min(deps1),np.max(deps1)),label='Reported (%s)'%period,color='lightyellow',edgecolor='black',log=True)
plt.hist(deps2,100,range=(np.min(deps1),np.max(deps1)),label='Catalog (%s)'%period,color='red',edgecolor='black',log=True)
plt.gca().set_xlim(xmin=0,xmax=20);
plt.gca().legend(loc='upper right');
plt.gca().set_ylabel("Count",fontsize='large', fontweight='normal')
plt.gca().set_xlabel("Depth (km)",fontsize='large', fontweight='normal')
ax3.text(-0.15,1,'(c)',transform=ax3.transAxes,size=20,weight='normal')

utcs1=[float(ii) for ii in utcs1]
utcs2=[float(ii) for ii in utcs2]
utcs3=[float(ii) for ii in utcs3]

ax4=fig.add_subplot(224)
print("utcs1:",len(utcs1),"utcs2",len(utcs2))
plt.hist(utcs1,31,range=(np.min(utcs1),np.max(utcs1)),label='EQCCT (%s)'%period,color='lightgray',edgecolor='black',log=True)
plt.hist(utcs3,31,range=(np.min(utcs1),np.max(utcs1)),label='Reported (%s)'%period,color='lightyellow',edgecolor='black',log=True)
plt.hist(utcs2,31,range=(np.min(utcs1),np.max(utcs1)),label='Catalog (%s)'%period,color='red',edgecolor='black',log=True)
# plt.gca().set_xlim(xmin=0,xmax=30);
plt.gca().legend(loc='upper right');
plt.gca().set_ylabel("Number of earthquakes by days",fontsize='large', fontweight='normal')
# plt.gca().set_xlabel("Time",fontsize='large', fontweight='normal')
plt.gca().set_xlim(xmin=min(utcs1),xmax=max(utcs1))

x1,x2=plt.gca().get_xlim()
if if3month == True:
	plt.xticks([x1, (x1+x2)/2, x2], ['Aug 1 2022', 'Sep 16 2022', 'Oct 31 2022'])
else:
	plt.xticks([x1, (x1+x2)/2, x2], ['Oct 1 2022', 'Oct 16 2022', 'Oct 31 2022'])
ax4.text(-0.15,1,'(d)',transform=ax4.transAxes,size=20,weight='normal')
plt.savefig('test_eqcct_eventdistribution2.png',format='png',dpi=300,bbox_inches='tight')
plt.show()

