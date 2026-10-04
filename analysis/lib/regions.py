from datetime import datetime, timedelta
import numpy as np
AR_START = datetime(1979,1,1,0,0); AR_N_STEPS = 56700
REGIONS = {
 "W_N_America": dict(lat=(30,50),   lon=[(230,245)],        winter=[11,12,1,2,3]),
 "W_Europe":    dict(lat=(37,57),   lon=[(352,360),(0,7)],  winter=[10,11,12,1,2,3]),
 "W_S_America": dict(lat=(-50,-30), lon=[(283,298)],        winter=[5,6,7,8,9]),
 "E_Australia": dict(lat=(-42,-22), lon=[(147,162)],        winter=[5,6,7,8,9]),
}
def index_to_datetime(idx): return AR_START + timedelta(hours=6*(int(idx)-1))
def cos_lat_weights(lats): return np.cos(np.deg2rad(np.asarray(lats,dtype=float)))
