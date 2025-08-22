import numpy as np
import numpy.ma as ma
import pandas as pd

def rx1day(precip_data):
    return ma.masked_array(ma.max(precip_data, axis=0), mask = precip_data.mask[0])

def rx5day(precip_data):
    # Create a sliding window view of the data with window size of 5 days
    rolling_sums = np.lib.stride_tricks.sliding_window_view(precip_data, window_shape=5, axis=0)
    # Sum along the window axis to get 5-day sums
    rolling_sums = ma.sum(rolling_sums, axis=3)
    max_5day = ma.max(rolling_sums, axis=0)
    return ma.masked_array(max_5day, mask = precip_data.mask[0])

def r10mm(precip_data):
    return ma.masked_array(ma.sum(precip_data >= 10, axis=0), mask = precip_data.mask[0])

def r20mm(precip_data):
    return ma.masked_array(ma.sum(precip_data >= 20, axis=0), mask = precip_data.mask[0])

def cdd(precip_data):
    dry_days = (precip_data < 1).astype(int)

    # Compute the lengths of consecutive dry days
    run_length = ma.zeros(dry_days.shape)
    run_length[0] = dry_days[0]
    for i in range(1, dry_days.shape[0]):
        run_length[i] = (run_length[i-1] + 1) * dry_days[i]
    
    max_cdd = ma.max(run_length, axis=0)
    return ma.masked_array(max_cdd,mask = precip_data.mask[0])
 

def cwd(precip_data):
    wet_days = (precip_data >= 1).astype(int)

    # Compute the lengths of consecutive wet days
    run_length = ma.zeros(wet_days.shape)
    run_length[0] = wet_days[0]
    for i in range(1, wet_days.shape[0]):
        run_length[i] = (run_length[i-1] + 1) * wet_days[i]
    
    max_cwd = ma.max(run_length, axis=0)
    return ma.masked_array(max_cwd,mask = precip_data.mask[0])
