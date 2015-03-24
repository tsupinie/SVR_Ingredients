import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil import tz
import matplotlib.dates as dts
import urllib
import random
import sys
from StringIO import StringIO

'''
    get_mesohistorical.py
    Author: Greg Blumberg (OU/CIMMS/SoM)
    Email: wblumberg@ou.edu

    This script will grab UA historical data from the AGWEATHER archive
    and parse it into the CSV file needed to plot it via the plot_SPCUA.py
    script.

    Archive path: http://agweather.mesonet.org/data/public/noaa/upperair/mdf/

    Inputs:
    1.) yyyymmdd (i.e. 19740403)
    2.) hr (i.e. 12 or 00)
'''

yyyymmdd = sys.argv[1]
hr = sys.argv[2]

def get_url(dt):
    
    #Load in the data for these times and append to the heights
    #Times needs to be a list of datetime objects
    RMISSD = '--'
    #Code for incrementing across each 
    url = 'http://agweather.mesonet.org/data/public/noaa/upperair/mdf/%Y/%m/%d/%Y%m%d%H%M.mdf' 
    url = datetime.strftime(dt, url) 
    return url

def readData(url):
    print "READING URL" 
    try:
        snfile = urllib.urlopen(url).read()
    except:
        print "CANNOT READ:", url
        sys.exit()
    sound_data = StringIO( snfile )
    data = np.genfromtxt( sound_data, dtype=str, skiprows=2)
    return data

try:
    sndata = readData(get_url(datetime.strptime(yyyymmdd+hr, '%Y%m%d%H')))
except Exception,e:
    print e
    print "Program failed to read in data."
    sys.exit()

def win(wordlist, word_fragment):
    boolean = []
    for w in wordlist:
        boolean.append(w.startswith(word_fragment))
    return boolean

# Open file for the data to be written to.
n = open(yyyymmdd+hr+'.csv', 'w')
pres = []
for i in sndata[0][np.where(win(sndata[0], 'TAIR_'))[0]]:
    pres.append(i.split('_')[1])

tdry_idxs = np.where(win(sndata[0], 'TAIR_'))[0]
tdew_idxs = np.where(win(sndata[0], 'TDEW_'))[0]
hght_idxs = np.where(win(sndata[0], 'HGHT_'))[0]
wspd_idxs = np.where(win(sndata[0], 'WDIR_'))[0]
wdir_idxs = np.where(win(sndata[0], 'WSPD_'))[0]
for i in range(1,len(sndata)):
    station = sndata[i,1]
    wmoid = sndata[i,0]
    lat = sndata[i,3]
    lon = sndata[i,4]
    elev = sndata[i,5]

    temp = np.where(np.asarray(sndata[i,tdry_idxs], dtype=float) <= -991.0, -999.0, sndata[i,tdry_idxs])
    dwpc = np.where(np.asarray(sndata[i,tdew_idxs], dtype=float) <= -991.0, -999.0, sndata[i,tdew_idxs])
    wdir = np.where(np.asarray(sndata[i,wdir_idxs], dtype=float) <= -991.0, -999.0, sndata[i,wdir_idxs])
    wspd = np.where(np.asarray(sndata[i,wspd_idxs], dtype=float) <= -991.0, -999.0, sndata[i,wspd_idxs])
    hght = np.where(np.asarray(sndata[i,hght_idxs], dtype=float) <= -991.0, -999.0, sndata[i,hght_idxs])
    if len(hght) < len(wspd):
        hght = np.concatenate(([elev], hght))
    for j in range(len(hght)):
        if pres[j] == 'SFC':
            continue
        string = ','.join([wmoid, station, lat, lon, elev, pres[j], temp[j], dwpc[j], hght[j], wspd[j], wdir[j]])
#       print string
        n.write(string + '\n')
n.close()


