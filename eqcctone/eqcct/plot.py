from matplotlib import pyplot as plt
import numpy as np

def plot_traces(st,axoff=1,ayoff=1,titleoff=1,picks=None,ptime=None,stime=None,figname=None,showf=True,**kwargs):
	"""
	plot_traces: plot all traces in a stream quickly side-by-side from top to bottom
	
	INPUT
	st: Stream
	axoff: axes X off
	ayoff: axes Y off
	title: title off
	picks: P/S picks object
		in this subroutine, picks is a list of two-entries Dict 
		initialized by {'P':None,'S':None}
		(compared with obspy pick object obspy.core.event.origin.Pick)
	showf: if show figure interatively
	
	EXAMPLES
	from eqcct.plot.waveforms import plot_traces
	plot_traces(st);
	
	from eqcct.plot.waveforms import plot_traces
	plot_traces(st,axoff=0,ayoff=0,titleoff=0);
	
	from eqcct.plot import plot_traces; 
	plot_traces(obspy.read("newwaveforms/texnet2022ugad.mseed")[0:10]);
	

	"""


	ntr=len(st)
	
	fig = plt.figure(figsize=(6, 8))
	
	for ii in range(ntr):
		ax = plt.subplot(ntr,1,ii+1)
		nt=len(st[ii].data);
		twin=(nt-1)*1.0/st[ii].stats.sampling_rate;
		staname=st[ii].stats.network+'.'+st[ii].stats.station
		t=np.linspace(0,twin,nt)
		plt.plot(t,st[ii].data,color='k',label = st[ii].id, linewidth = 1, markersize=1)
		
		if ii==0 and titleoff != 1:
			plt.title(st[ii].stats.starttime,fontsize='large', fontweight='normal')
		
		if ii==ntr-1:
			if axoff == 1:
				plt.setp(ax.get_xticklabels(), visible=False)
			else:
				plt.setp(ax.get_xticklabels(), visible=True)
				ax.set_xlabel("Time (s)",fontsize='large', fontweight='normal')
		else:
			plt.setp(ax.get_xticklabels(), visible=False)
		ax.set_xlim(xmin=0)
		ax.set_xlim(xmax=t[-1])
		ymin, ymax = ax.get_ylim()
		
		if ayoff:
			plt.setp(ax.get_yticklabels(), visible=False)
			
		if picks is not None:
			if picks[ii]['P'] is not None:
				tp=picks[ii]['P']-st[ii].stats.starttime
				plt.vlines(tp, ymin, ymax, color = 'r', linewidth = 1, label="EQCCT P") #for P
			
			if picks[ii]['S'] is not None:
				ts=picks[ii]['S']-st[ii].stats.starttime
				plt.vlines(ts, ymin, ymax, color = 'g', linewidth = 1, label="EQCCT S") #for S

		if ptime is not None:
			tp=ptime[ii]-st[ii].stats.starttime
			plt.vlines(tp, ymin, ymax, color = 'r', linewidth = 1, label="EQCCT P") #for P
			
		if stime is not None:
			ts=stime[ii]-st[ii].stats.starttime
			plt.vlines(ts, ymin, ymax, color = 'g', linewidth = 1, label="EQCCT S") #for P
		
		plt.gca().legend(loc='lower left', fontsize = 5/(ntr/4))
		
	if figname is not None:
		plt.savefig(figname,**kwargs)
	
	if showf:
		plt.show()
	else:
		plt.close() #or plt.clear() ?
	