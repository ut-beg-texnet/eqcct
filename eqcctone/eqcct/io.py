def asciiread(fname):
	'''
	fname: file name
	din:   a list of lines
	withnewline: if with the newline symbol '\n': True: with; False: without
	
	Example:
	
	from eqcct.io import asciiread
	import os
	
	lines=asciiread(os.getenv('HOME')+'/chenyk.data2/various/cyksmall/texnet_stations_2022_1019.csv');
	'''
	
	f=open(fname,'r')
	lines=f.readlines()
	lines=[ii.strip() for ii in lines]
	
	return lines
	
	
def asciiwrite(fname,din,withnewline=False):
	'''
	fname: file name
	din:   a list of lines
	withnewline: if with the newline symbol '\n': True: with; False: without
	
	Example:
	
	from eqcct.io import asciiwrite
	import os
	
	f=open(os.getenv('HOME')+'/chenyk.data2/various/cyksmall/texnet_stations_2022_1019.csv')
	lines=f.readlines();
	
	asciiwrite('stations.txt',lines,withnewline=True);
	'''
	
	f=open(fname,'w')
	if withnewline:
		for ii in range(len(din)):
			f.write(str(din[ii]))
	else:
		for ii in range(len(din)):
			f.write(str(din[ii])+"\n")