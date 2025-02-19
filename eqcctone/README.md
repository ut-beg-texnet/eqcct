<img src='https://github.com/chenyk1990/gallery/blob/main/eqcct/logo2.png' alt='Slicing'>

**EQCCT**
======

## Description

**EQCCT** package is a production-ready EarthQuake detection and phase-picking method using the Compact Convolutional Transformer

-----------
## Reference
If you find this package useful, please do not forget to cite the following paper.

    Saad, O.M., Chen, Y.F., Siervo, D., Zhang, F., Savvaidis, A., Huang, G., Igonin, N., Fomel, S., and Chen, Y., (2023). EQCCT: A production-ready EarthQuake detection and phase picking method using the Compact Convolutional Transformer, IEEE Transactions on Geoscience and Remote Sensing, 61, 4507015, doi:10.1109/TGRS.2023.3319440.
    
    Chen, Y., Savvaidis, A., Saad, O.M., Siervo, D., Huang, G., Chen, Y.F., Grigoratos, I., Fomel, S., and Breton, C., (2024). Thousands of Induced Earthquakes per Month in West Texas Detected Using EQCCT, Geosciences, 14(5), 114, doi: 10.3390/geosciences14050114.
    
BibTeX:
	
	@article{eqcct,
	  author={Omar M. Saad and Yunfeng Chen and Daniel Siervo and Fangxue Zhang and Alexandros Savvaidis and Guo-chin Huang and Nadine Igonin and Sergey Fomel and Yangkang Chen},
	  title = {EQCCT: A production-ready EarthQuake detection and phase picking method using the Compact Convolutional Transformer},
	  journal={IEEE Transactions on Geoscience and Remote Sensing},
	  year=2023,
	  volume=61,
	  issue=12,
	  pages={4507015},
	  doi={10.1109/TGRS.2023.3319440},
	}

	@Article{eqcct2024catalog,
	  author={Yangkang Chen and Alexandros Savvaidis and Omar M. Saad and Daniel Siervo and Guo-chin Dino Huang and Yunfeng Chen and Iason Grigoratos and Sergey Fomel and Caroline Breton},
	  title = {Thousands of Induced Earthquakes per Month in West Texas Detected Using EQCCT},
	  journal={Geosciences},
	  year=2024,
	  volume=14,
	  issue=5,
	  number=5,
	  pages={114},
	  doi={10.3390/geosciences14050114},
}

The accepted version of the paper can be downloaded from:

	https://drive.google.com/drive/folders/1v9akXtnNy6b3gCWzUn9wTBiAsKPgHlKM?usp=sharing
	
-----------

## Copyright
    Developers of the EQCCT package, 2021-present

-----------

## License
    MIT License

-----------

## Install
First set up the environment and install the dependency packages

	conda create -n eqcct python=3.7.16
	conda activate eqcct
	conda install ipython notebook
	pip install obspy tqdm matplotlib-scalebar tensorflow==2.8.0 protobuf==3.20.1 pandas==1.3.5 scikit-learn==1.0.2
	
Then install EQCCT using the latest version

    git clone https://github.com/chenyk1990/eqcct
    cd eqcct
    pip install -v -e .
    
Or using Pypi

	pip install eqcct
	
Or using pip directly from Github

	pip install git+https://github.com/chenyk1990/eqcct

-----------
## Examples
1 Pick the P/S arrivals from an arbitrary waveform (stream) containing an earthquake (please run in the demos/ folder)
	
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

    
2 Japanese data example (The Ipython Notebooks are examples for playing with the models.)
    
    Please first download the mseed data from
    https://drive.google.com/drive/folders/1jAkW4kOvwUDYxXW-ty3BTY81fjLF7sKE?usp=share_link
    
    Then download the manual picks from
    https://drive.google.com/drive/folders/1KK16j1-WbwqKfTvh5gJbzJrX0fiayhl6?usp=share_link
    
    Then run the notebook 
    notebooks/Test_Predict_MSEED_Japan.ipynb for the single-model (P and S together) example
    notebooks/Test_Predict_MSEED_Japan_PS.ipynb for the double-model (P and S separately) example

    
-----------
## Development
    The development team welcomes voluntary contributions from any open-source enthusiast. 
    If you want to make contribution to this project, feel free to contact the development team. 

-----------
## Contact
    Regarding any questions, bugs, developments, collaborations, please contact  
    Yangkang Chen
    chenyk2016@gmail.com

-----------
## NOTES:

-----------
## Gallery
The gallery figures of the eqcct package can be found at
    https://github.com/chenyk1990/gallery/tree/main/eqcct

Each figure in the gallery directory corresponds to a DEMO script in the "demo" directory. These gallery figures are also presented below. 

DEMO1 
The following figure shows an example of a comparison between different catalogs. Generated by [demos/test_eqcct_texnet2022yplg.py](https://github.com/chenyk1990/eqcct/blob/main/demos/test_eqcct_texnet2022yplg.py)

<img src='https://github.com/chenyk1990/gallery/blob/main/eqcct/test_eqcct_texnet2022yplg.png' alt='Slicing' width=960/>

DEMO2
The following figure shows an example of a comparison between initial LOCSAT and refined NonLinLoc catalogs. Generated by [demos/test_eqcct_locsatvsnll.py](https://github.com/chenyk1990/eqcct/blob/main/demos/test_eqcct_locsatvsnll.py)

<img src='https://github.com/chenyk1990/gallery/blob/main/eqcct/test_eqcct_locsatvsnll.png' alt='Slicing' width=960/>


DEMO3
The following figure shows an example of a comparison between different catalogs. Generated by [demos/test_eqcct_nll_growclust.py](https://github.com/chenyk1990/eqcct/blob/main/demos/test_eqcct_nll_growclust.py)

<img src='https://github.com/chenyk1990/gallery/blob/main/eqcct/delaware202210depthreloc.png' alt='Slicing' width=960/>

DEMO4
The following figure shows an example of a comparison between the one-month catalog before (up) and after (down) relocation, manual (left), and EQCCT (right). Generated by [demos/test_eqcct_relo_manualvsAI.py](https://github.com/chenyk1990/eqcct/blob/main/demos/test_eqcct_relo_manualvsAI.py)

<img src='https://github.com/chenyk1990/gallery/blob/main/eqcct/test_eqcct_relo_manualvsAI_locvsreloc_onemonth.png' alt='Slicing' width=960/>

DEMO5
The following figure shows an example of comparison between one-year (manual, left) and one-month (EQCCT, right) catalog before (up) and after (down) relocation. Generated by [demos/test_eqcct_relo_manualvsAI.py](https://github.com/chenyk1990/eqcct/blob/main/demos/test_eqcct_relo_manualvsAI.py)

<img src='https://github.com/chenyk1990/gallery/blob/main/eqcct/test_eqcct_relo_manualvsAI_oneyear.png' alt='Slicing' width=960/>

DEMO6
The following figure shows an example of a comparison between the three-month (manual, left) and three-month (EQCCT, right) catalog after relocation. Generated by [demos/test_eqcct_relo_manualvsAI.py](https://github.com/chenyk1990/eqcct/blob/main/demos/test_eqcct_relo_manualvsAI.py)

<img src='https://github.com/chenyk1990/gallery/blob/main/eqcct/test_eqcct_relo_manualvsAI.png' alt='Slicing' width=960/>

DEMO7
The following figure shows a test on station density's effect on earthquake detectability. Generated by [demos/test_eqcct_stationeffect.py](https://github.com/chenyk1990/eqcct/blob/main/demos/test_eqcct_stationeffect.py)

<img src='https://github.com/chenyk1990/gallery/blob/main/eqcct/test_eqcct_stationeffect.png' alt='Slicing' width=960/>


DEMO8
The following figure shows plots of TexNet catalog of different periods (1 month, 3 months, 6 months, 1 year) on top of event density plot of the one-month EQCCT catalog. Generated by [demos/test_eqcct_eventdistribution.py](https://github.com/chenyk1990/eqcct/blob/main/demos/test_eqcct_eventdistribution.py)

<img src='https://github.com/chenyk1990/gallery/blob/main/eqcct/test_eqcct_eventdistribution1.png' alt='Slicing' width=960/>


DEMO9
The following figure shows a one-month continuous data test including station distribution, magnitude-frequency plot, depth distribution, daily detection distribution between the one-month TexNet, reported (detected via an optimized STA/LTA algorithm without manual quality control), EQCCT catalogs. Generated by [demos/test_eqcct_eventdistribution.py](https://github.com/chenyk1990/eqcct/blob/main/demos/test_eqcct_eventdistribution.py)

<img src='https://github.com/chenyk1990/gallery/blob/main/eqcct/test_eqcct_eventdistribution2.png' alt='Slicing' width=960/>

DEMO10
The following GIF animation  shows a  one-month daily accumulative seismicity using the highly complete EQCCT catalog with a NLL locator. Generated by [demos/test_eqcct_nll_gif.py](https://github.com/chenyk1990/eqcct/blob/main/demos/test_eqcct_nll_gif.py)

<img src='https://github.com/chenyk1990/gallery/blob/main/eqcct/test_eqcct_nll_gif.gif' alt='Slicing' width=960/>


DEMO11
The following example shows how to benchmark EQCCT with other models based on the TXED dataset (Earthquake AI dataset from Texas). Generated by [demos/test_eqcct_benchmark_with_others.py](https://github.com/chenyk1990/eqcct/blob/main/demos/test_eqcct_benchmark_with_others.py)

<img src='https://github.com/chenyk1990/gallery/blob/main/eqcct/EQCCT_vs_HANet.png' alt='Slicing' width=960/>

