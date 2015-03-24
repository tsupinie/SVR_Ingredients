
import numpy as np

from scipy.interpolate import interp1d

import matplotlib as mpl
mpl.use('agg')
import pylab
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap

from buf_decoder import BufkitFile

import copy
from datetime import datetime, timedelta
import cPickle
from math import floor
import os

moisture_stns_obs   = ['MMMD', '76692', 'KBRO', 'KCRP', 'KLCH', 'KLIX', 'KTLH'] 
lapse_rate_stns_obs = ['KABQ', 'KEPZ', 'KAMA', 'KDNR', 'KMAF', 'KRIW', 'KFGZ', 'KTUS', 'KGJT']
shear_stns_obs      = ['KOUN', 'KDDC', 'KFWD', 'KTOP', 'KSGF', 'KLZK', 'KSHV', 'KBNA', 'KJAN', 'KBMX']

stns_obs_fcst = {
    'MMMD':'MMMD', '76692':'MMVR', 'KBRO':'KBRO', 'KCRP':'KCRP', 'KLCH':'KLCH', 'KLIX':'KNEW', 'KTLH':'KTLH',
    'KABQ':'KABQ', 'KEPZ':'KEPZ', 'KAMA':'KAMA', 'KDNR':'KDEN', 'KMAF':'KMAF', 'KRIW':'KRIW', 'KFGZ':'KFLG', 'KTUS':'KTUS', 'KGJT':'KGJT',
    'KOUN':'KOKC', 'KDDC':'KDDC', 'KFWD':'KDFW', 'KTOP':'KTOP', 'KSGF':'KSGF', 'KLZK':'KLIT', 'KSHV':'KSHV', 'KBNA':'KBNA', 'KJAN':'KJAN', 'KBMX':'KBHM',
}

def loadCSV(dt):
    file_name = dt.strftime("%Y%m%d%H.csv")
    names = [ 'id', 'num', 'lat', 'lon', 'elv', 'pres', 'tmpc', 'dwpc', 'hght', 'wdir', 'wspd' ]
    dtypes = [ (i, c) for i, c in enumerate([ str, int, float, float, float, float, float, float, float, float, float ]) ]
    csv = np.loadtxt(file_name, dtype=[ (n, type(t)) for n, (i, t) in zip(names, dtypes) ], converters=dict(dtypes), delimiter=',')
    return csv

def loadBufkit(stid, dt):
#   url = "ftp://ftp.meteo.psu.edu/pub/bufkit/GFS/%02d/gfs3_%s.buf" % (run_hour, stid.lower())
    url ="http://mtarchive.geol.iastate.edu/%s/gfs/gfs3_%s.buf" % (dt.strftime("%Y/%m/%d/bufkit/%H"), stid.lower())
    buf = BufkitFile(url)
    return buf

def chiclet(buf_files, obs_profs, moisture_thresh=10., lapse_thresh=8., shear_thresh=35.):
    def computeChart(stns_obs, compute, thresh):
        chiclet = np.nan * np.zeros((len(fcst_times), len(run_times)))
        stn_coords = {}
        for idx, run in enumerate(run_times):
            jdy = fcst_times.index(run)

            try:
                for stn in stns_obs:
                    buf = buf_files[(stn, run)]
                    prof = dict( (k, v[0]) for k, v in buf.__dict__.iteritems() if type(v) == np.ndarray )
                    param = compute(prof)

                    if np.isnan(chiclet[jdy, idx]):
                        chiclet[(jdy):(jdy + param.shape[0]), idx] = 0.

                    chiclet[(jdy):(jdy + param.shape[0]), idx] += np.where(param >= thresh, 1, 0)
                    stn_coords[stn] = (buf.slat, buf.slon)
            except KeyError:
                continue

        chiclet /= len(stns_obs)
        return chiclet, stn_coords

    def chicletObs(chiclet, stns_obs, compute, thresh):
        _epoch = datetime(1970, 1, 1, 0, 0, 0)

        for jdy, fcst in enumerate(fcst_times):
            fcst_stamp = (fcst - _epoch).total_seconds()
            fcst_rnd = _epoch + timedelta(seconds=(floor(fcst_stamp / (6. * 3600)) * (6. * 3600)))

            try:
                idx = run_times.index(fcst_rnd)
            except ValueError:
                continue

            fcst_rnd = _epoch + timedelta(seconds=(round(fcst_stamp / (12. * 3600)) * (12. * 3600)))
            if fcst_rnd > datetime.now() - timedelta(hours=2):
                fcst_rnd -= timedelta(hours=12)
            n_profs = 0

            for stn in stns_obs:
                prof = obs_profs[(stn, fcst_rnd)]

                if len(prof['pres']) == 0:
                    continue

                prof = prof[np.newaxis, :]
                param = compute(prof)

                if np.isnan(chiclet[jdy, -1]):
                    chiclet[jdy, (idx + 1):] = 0

                chiclet[jdy, (idx + 1):] += (1 if param >= thresh else 0)
                n_profs += 1

            chiclet[jdy, (idx + 1):] /= float(n_profs)
        return chiclet

    def computeDewp(buf, level=850.):
        dwpc = buf['dwpc'].astype(np.float32)
        pres = buf['pres'].astype(np.float32)
        moist = np.empty(buf['dwpc'].shape[-2:-1])
        for ifcst in xrange(moist.shape[0]):
            pres_fcst = pres[ifcst, :]
            dwpc_fcst = dwpc[ifcst, :]

            good_idxs = np.where(dwpc_fcst >= -900.)

            moist[ifcst] = interp1d(pres_fcst[good_idxs], dwpc_fcst[good_idxs])(level)
        return moist

    def computeLapse(buf, llevel=700., ulevel=500.):
        tmpc = buf['tmpc'].astype(np.float32)
        hght = buf['hght'].astype(np.float32)
        pres = buf['pres'].astype(np.float32)

        tlower = np.empty(tmpc.shape[-2:-1])
        tupper = np.empty(tmpc.shape[-2:-1])
        zlower = np.empty(hght.shape[-2:-1])
        zupper = np.empty(hght.shape[-2:-1])
        for ifcst in xrange(tlower.shape[0]):
            pres_fcst = pres[ifcst, :]
            tmpc_fcst = tmpc[ifcst, :]
            hght_fcst = hght[ifcst, :]

            good_idxs = np.where(tmpc_fcst > -900.)
            tlower[ifcst], tupper[ifcst] = interp1d(pres_fcst[good_idxs], tmpc_fcst[good_idxs])([llevel, ulevel])
            zlower[ifcst], zupper[ifcst] = interp1d(pres_fcst[good_idxs], hght_fcst[good_idxs])([llevel, ulevel])

        lapse = -1000 * (tlower - tupper) / (zlower - zupper)
        return lapse

    def computeShear(buf, ulevel=6000.):
        wspd = buf['wspd'].astype(np.float32)
        wdir = buf['wdir'].astype(np.float32)
        hght = buf['hght'].astype(np.float32)

        u0 = np.empty(wspd.shape[-2:-1])
        v0 = np.empty(wspd.shape[-2:-1])
        u6 = np.empty(wspd.shape[-2:-1])
        v6 = np.empty(wspd.shape[-2:-1])

        for ifcst in xrange(u0.shape[0]):
            wspd_fcst = wspd[ifcst, :]
            wdir_fcst = wdir[ifcst, :]
            hght_fcst = hght[ifcst, :]

            good_idxs = np.where(wspd_fcst > -900.)
            wspd0 = wspd_fcst[good_idxs][0]
            wdir0 = wdir_fcst[good_idxs][0]

            this_u6 = -wspd_fcst[good_idxs] * np.sin(np.radians(wdir_fcst[good_idxs]))
            this_v6 = -wspd_fcst[good_idxs] * np.cos(np.radians(wdir_fcst[good_idxs]))

            u0[ifcst] = -wspd0 * np.sin(np.radians(wdir0))
            v0[ifcst] = -wspd0 * np.cos(np.radians(wdir0))
            u6[ifcst] = interp1d(hght_fcst[good_idxs], this_u6)(ulevel)
            v6[ifcst] = interp1d(hght_fcst[good_idxs], this_v6)(ulevel)

        shear = np.hypot(u6 - u0, v6 - v0)
        return shear

    stations, run_times = zip(*buf_files.keys())
    stations = sorted(list(set(stations)))
    run_times = sorted(list(set(run_times)))
    fcst_times = sorted(list(set( d for b in buf_files.values() for d in b.dates )))
    x_start, x_end = min(fcst_times), max(fcst_times)
    y_start, y_end = min(run_times), max(run_times)

    chiclet_moisture = np.nan * np.zeros((len(fcst_times), len(run_times)))
    chiclet_lapse = np.nan * np.zeros((len(fcst_times), len(run_times)))
    chiclet_shear = np.nan * np.zeros((len(fcst_times), len(run_times)))

    stn_coords = {}

    # Compute chiclet moisture chart
    chiclet_moisture, stn_coords_moisture = computeChart(moisture_stns_obs,   computeDewp,  moisture_thresh)
    chiclet_lapse, stn_coords_lapse       = computeChart(lapse_rate_stns_obs, computeLapse, lapse_thresh)
    chiclet_shear, stn_coords_shear       = computeChart(shear_stns_obs,      computeShear, shear_thresh)

    stn_coords.update(stn_coords_moisture)
    stn_coords.update(stn_coords_lapse)
    stn_coords.update(stn_coords_shear)

    # Add the observations
    chiclet_moisture = chicletObs(chiclet_moisture, moisture_stns_obs,   computeDewp,  moisture_thresh)
    chiclet_lapse    = chicletObs(chiclet_lapse,    lapse_rate_stns_obs, computeLapse, lapse_thresh)
    chiclet_shear    = chicletObs(chiclet_shear,    shear_stns_obs,      computeShear, shear_thresh)

    return (x_start, x_end), (y_start, y_end), chiclet_moisture, chiclet_lapse, chiclet_shear, stn_coords

def plotchiclet(f_bounds, r_bounds, chic_moist, chic_lapse, chic_shear, stn_coords, chic_good, moisture_thresh=(8., 14.), lapse_thresh=(7., 8.), shear_thresh=(35., 50.)):
    f_start, f_end = f_bounds
    r_start, r_end = r_bounds
    
    bare_moist_thresh, good_moist_thresh = moisture_thresh
    bare_lapse_thresh, good_lapse_thresh = lapse_thresh
    bare_shear_thresh, good_shear_thresh = shear_thresh
    chic_good_moist, chic_good_lapse, chic_good_shear = chic_good

    xs = np.arange(chic_moist.shape[0] + 1)
    ys = np.arange(chic_moist.shape[1] + 1)

    delta_f = (f_end - f_start) / (chic_moist.shape[0] - 1)
    delta_r = (r_end - r_start) / (chic_moist.shape[1] - 1)

    ticks_f = [ (f_start + x * delta_f).strftime("%d %b %HZ") for x in xs[:-1] ]
    ticks_r = [ (r_start + y * delta_r).strftime("%d/%HZ") for y in ys[:-1] ]
    ticks_x = (xs[1:] + xs[:-1]) / 2.
    ticks_y = (ys[1:] + ys[:-1]) / 2.
    x_offset = ticks_f.index([ t for t in ticks_f if t[-3:] == "00Z"  ][0])
    y_offset = ticks_r.index([ t for t in ticks_r if t[-3:] == "00Z"  ][0])
    x_thin = 8
    y_thin = 4

    xs_line = xs[:ys.shape[0]] * 2
    ys_line = ys

    xs_line = np.vstack((xs_line, xs_line)).flatten(order='F')[:-1]
    ys_line = np.vstack((ys_line, ys_line)).flatten(order='F')[1:]

    def createSubplot(chic, chic_good, bar_text, inset_map, stns, cmap, xticks=False):
        bg_rect = Rectangle((0., 0.), 1., 1., ec='k', fc='#cccccc', zorder=-10, transform=pylab.gca().transAxes)
        pylab.gca().add_patch(bg_rect)
        pylab.pcolormesh(xs, ys, np.ma.masked_where(np.isnan(chic), chic).T, vmin=0, vmax=1, cmap=cmap)
        pylab.plot(xs_line, ys_line, 'k-', linewidth=1.5)
        bar = pylab.colorbar(aspect=10, pad=0.01, shrink=0.88)
        bar.ax.text(4.5, 0.5, bar_text, rotation=90., ha='center', va='center', transform=bar.ax.transAxes)
        if xticks:
            pylab.xticks(ticks_x[x_offset::x_thin], ticks_f[x_offset::x_thin], rotation=90)
            pylab.xlabel("Time")
        else:
            pylab.xticks(ticks_x[x_offset::x_thin], [ '' for t in ticks_f[x_offset::x_thin] ])
        pylab.yticks(ticks_y[y_offset::y_thin], ticks_r[y_offset::y_thin])
        pylab.ylabel("GFS Run")
        for (idx, jdy) in np.ndindex(chic_moist.shape):
            if chic_good[idx, jdy] > 0 and chic_good[idx, jdy] >= chic[idx, jdy] / 2.:
                rect = Rectangle((xs[idx], ys[jdy]), xs[idx + 1] - xs[idx], ys[jdy + 1] - ys[jdy], ec='k', color='none', zorder=10, hatch='xxx', linewidth=0)
                pylab.gca().add_patch(rect)

        pylab.xlim(xs.min(), xs.max())
        pylab.ylim(ys.min(), ys.max())
        pylab.grid()

        par_ax = pylab.gca()
        ax_width_inches, ax_height_inches = [ f * a for f, a in zip(pylab.gcf().get_size_inches(), par_ax.get_position().size) ]
        ins_ax = inset_axes(pylab.gca(), 0.4 * ax_height_inches, 0.4 * ax_height_inches, loc=4)
        inset_map.drawstates()
        inset_map.drawcountries()
        inset_map.drawcoastlines()
        marker_color = mpl.colors.rgb2hex(cmap(0.6))
        stn_xs, stn_ys = inset_map(*zip(*[ stn_coords[stn] for stn in stns ])[::-1])
        pylab.plot(stn_xs, stn_ys, 'ko', markersize=4, mfc=marker_color)
        return

    moist_map = Basemap(projection='lcc', resolution='l',
        lat_0=25., lon_0=-91.5, width=1600000.,height=1600000.,
        lat_1=23., lat_2=29.
    )
    lapse_map = Basemap(projection='lcc', resolution='l',
        lat_0=37.5, lon_0=-106., width=1500000., height=1500000.,
        lat_1=34.5, lat_2=40.5
    )
    shear_map = Basemap(projection='lcc', resolution='l',
        lat_0=35., lon_0=-93., width=1500000., height=1500000.,
        lat_1=32., lat_2=38.
    )

    pylab.figure(figsize=(8., 8.), dpi=150)
    pylab.subplots_adjust(top=0.89, bottom=0.17, left=0.12, right=1.0, hspace=0.0)
    pylab.subplot(311)
    createSubplot(chic_moist, chic_good_moist, "Fraction 850 mb $T_d$\n$\geq$ %.1f$^{\circ}$C" % bare_moist_thresh, moist_map, moisture_stns_obs, mpl.cm.get_cmap('Greens'))
    pylab.subplot(312)
    createSubplot(chic_lapse, chic_good_lapse, "Fraction 700-500 mb $\gamma$\n$\geq$ %.1f$^{\circ}$C km$^{-1}$" % bare_lapse_thresh, lapse_map, lapse_rate_stns_obs, mpl.cm.get_cmap('Reds'))
    pylab.subplot(313)
    createSubplot(chic_shear, chic_good_shear, "Fraction 0-6 km |" + r'$\Delta\vec{V}$' + "|\n$\geq$ %.1f kts" % bare_shear_thresh, shear_map, shear_stns_obs, mpl.cm.get_cmap('Purples'), xticks=True)
    pylab.suptitle("\"Chiclet\" chart for Southern US severe convective storm ingredients")
    pylab.text(0.5, 0.92, "Observed to the top/left of the stair-stepped black line, forecast to the bottom/right.\n" +
        "Hatched boxes indicate at least half the stations meeting the \"bare\" criteria (colors) also meet\n" + 
        "\"good\" criteria (850 mb $T_d$ $\geq$ %.1f$^{\circ}$C, 700-500 mb $\gamma$ $\geq$ %.1f$^{\circ}$C km$^{-1}$, 0-6 km |" % (good_moist_thresh, good_lapse_thresh) + r'$\Delta\vec{V}$' + "| $\geq$ %.1f kts)." % good_shear_thresh, 
        fontsize=10, linespacing=0.9, ha='center', va='center', transform=pylab.gcf().transFigure)
    pylab.savefig('chic.png', dpi=pylab.gcf().dpi)
    pylab.close()

def main():
    _epoch = datetime(1970, 1, 1, 0, 0, 0)
    dt = datetime.utcnow()
    dt_stamp = (dt - _epoch).total_seconds()
    dt -= timedelta(seconds=(dt_stamp % (6 * 3600)))

    delta = timedelta(hours=12)

    bare_moist_thresh = 8.
    good_moist_thresh = 14.
    bare_lapse_thresh = 7.
    good_lapse_thresh = 8.
    bare_shear_thresh = 40.
    good_shear_thresh = 55.

    buf_file_name = 'buf_files.pkl'
    if os.path.exists(buf_file_name):
        buf_files = cPickle.load(open(buf_file_name, 'r'))
    else:
        buf_files = {}
    obs_profs = {}

    dt_start = copy.copy(dt)

    while dt > dt_start - timedelta(days=7):
        for stn in moisture_stns_obs + lapse_rate_stns_obs + shear_stns_obs:
            gfs_stn = stns_obs_fcst[stn]
            try:
                csv = loadCSV(dt)
                obs_profs[(stn, dt)] = csv[np.where(csv['id'] == stn)]
            except IOError:
                pass

            if (stn, dt) not in buf_files and dt + timedelta(hours=4) < datetime.utcnow():
                print "Working on %s at %s ..." % (gfs_stn, dt.strftime("%Y-%m-%d %H UTC")),
                try:
                    buf = loadBufkit(gfs_stn, dt)
                    buf_files[(stn, dt)] = buf
                    print
                except IOError:
                    print "[ No Data ]"
        dt -= timedelta(hours=6)

    for (stn, dt) in buf_files.keys():
        if dt <= dt_start - timedelta(days=7):
            del buf_files[(stn, dt)]

    cPickle.dump(buf_files, open(buf_file_name, 'w'), -1)

    chic_bare = chiclet(buf_files, obs_profs, moisture_thresh=bare_moist_thresh, lapse_thresh=bare_lapse_thresh, shear_thresh=bare_shear_thresh)
    chic_good = chiclet(buf_files, obs_profs, moisture_thresh=good_moist_thresh, lapse_thresh=good_lapse_thresh, shear_thresh=good_shear_thresh)
    plotchiclet(*chic_bare, chic_good=chic_good[-4:-1], 
        moisture_thresh=(bare_moist_thresh, good_moist_thresh), lapse_thresh=(bare_lapse_thresh, good_lapse_thresh),
        shear_thresh=(bare_shear_thresh, good_shear_thresh)
    )
    return

if __name__ == "__main__":
    main()
