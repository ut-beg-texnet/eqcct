## Example for applying EQCCT to a random input data (stream)
#
# By Yangkang Chen
# March, 2023


import obspy.core.utcdatetime as utc
from obspy.clients.fdsn import Client
from eqcct.stream import st_predictor
from eqcct import plot_traces
import pandas as pd
import os

os.system('rm -rf ./detections_ALPN')

cl=Client('http://rtserve.beg.utexas.edu')
st = cl.get_waveforms('TX', 'ALPN', '00', '*', utc.UTCDateTime(2022, 12, 16, 23, 35, 27, 000000), utc.UTCDateTime(2022, 12, 16, 23, 37, 27, 000000))


st_predictor(input_modelP='../ModelPS/test_trainer_024.h5',
         input_modelS='../ModelPS/test_trainer_021.h5',
         stinput = st,
         output_dir='detections_ALPN',
         P_threshold=0.1,
         S_threshold=0.1, 
         number_of_plots=10,
         normalization_mode='std',
         batch_size=1,
         overlap=0.8,
         gpuid=None,
         gpu_limit=None) 


data = pd.read_csv('detections_ALPN/ALPN_outputs/X_prediction_results.csv', low_memory=False)
data.p_arrival_time

pat=[ii for ii in list(data.p_arrival_time) if type(ii) is str]; #P arrival time
sat=[ii for ii in list(data.s_arrival_time) if type(ii) is str]; #S arrival time

import obspy
plot_traces(st,axoff=1,titleoff=1,ptime= [obspy.UTCDateTime(pat[0]),obspy.UTCDateTime(pat[0]),obspy.UTCDateTime(pat[0])],stime=[obspy.UTCDateTime(sat[0]),obspy.UTCDateTime(sat[0]),obspy.UTCDateTime(sat[0])],figname='test_eqcct_texnet2022yplg.png',dpi=500);

