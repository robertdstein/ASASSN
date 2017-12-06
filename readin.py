"""Script to read in ASASSN data"""


import numpy as np
import csv
import os
from astropy.time import Time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tabulate import tabulate
import os
from tqdm import tqdm
from scipy import optimize
from astropy.coordinates import SkyCoord
import astropy.units as u
from ned import check_name, check_coordinate
from sklearn import cluster, mixture

import lightcurve_model as lc

user_dir = "/afs/ifh.de/user/s/steinrob/Desktop/python/ASASSN"

output_dir = "/afs/ifh.de/user/s/steinrob/scratch/ASASSN_output/"

threshold_n_points = len(lc.default())

def wrap_around_180(ra_deg):
    ra = np.deg2rad(ra_deg)
    ra[ra > np.pi] -= 2 * np.pi
    return ra

class Observation:

    def __init__(self, info):
        self.date = Time(float("245" + info[0]) - 2400000.5, format="mjd")
        self.date.out_subfmt = "date"

        self.date_mjd = self.date.mjd

        self.field_image = info[1].split("/")
        self.xpixel = float(info[2])
        self.ypixel = float(info[3])

        self.coords = SkyCoord(info[4] + " " + info[5], unit=(u.deg, u.deg))
        self.ra = self.coords.ra
        self.dec = self.coords.dec

        self.ra_deg  = float(self.ra.deg)
        self.dec_deg = float(self.dec.deg)

        self.galactic_l = float(info[6])
        self.galactic_b = float(info[7])
        self.n_up = int(info[8])
        self.n_down = int(info[9])
        self.rfc_score = float(info[10])
        self.sub_img_counts = int(info[11])
        self.ref_img_counts = int(info[12])
        self.offset_from_galaxy = float(info[13])

        self.redshift = np.nan

        g_name = info[14].strip(":")

        extra_info = info[15].split("(")

        if extra_info[0] == 'APMUKS':
            rest = extra_info[2:]
            stripped_info = ["(".join(extra_info[:2])]
            stripped_info.extend(rest)
        else:
            stripped_info = extra_info

        if "AGN" in g_name:
            name = g_name[:-3]
            self.AGN = True
        else:
            name = g_name
            self.AGN = False

        if "AGN" in stripped_info[0]:
            stripped_info[0] = stripped_info[0][:-3]
            self.AGN = True

        if stripped_info[0] != name:

            if name.split("(")[0] == stripped_info[0]:
                self.alias = [stripped_info[0]]
            else:
                self.alias = [ x for x in [stripped_info[0], name]
                               if x != 'sdssgal']
        else:
            self.alias = [name]

        self.galaxy_name = self.alias[0]

        for i, x in enumerate(extra_info):
            if i == 0:
                pass
            elif x[-1] == ")":
                extra_info[i] = x[:-1]

        # print self.galaxy_name, info[15], extra_info

        if extra_info[1][0] == "z":
            try:
                self.redshift = float(extra_info[1][2:])
            except ValueError:
                pass

        self.additional_info = info[15]
        self.notes = info[16]
        self.image_name = info[17]

        self.NED_ra = np.nan
        self.NED_dec = np.nan
        self.NED_redshift = np.nan
        self.NED_name = np.nan
        self.NED_coords = np.nan
        self.NED_offset = np.nan

        self.add_ned()

    def add_ned(self):
        if not isinstance(self.NED_name, str):
            best = None
        #     for name in self.alias:
        #         entry = check_name(name)
        #         if entry is not None:
        #             best = entry["data_table"]
        #             if best is None:
        #
        #             else:
        #                 if best["Object Name"] == \
        #                         entry["data_table"]["Object Name"]:
        #                     pass
        #                 else:
        #                     pass
        #                     # print self.alias
        #                     # print best
        #                     # print entry["data_table"]
        #                     # raise Exception("Conflict with aliases. Each matches "
        #                     #                 "to a different object!")
        #
        #             if len(best) > 1:
        #                 raise Exception("Too many entries")

            entry = check_coordinate(self.ra.deg, self.dec.deg)
            if entry is not None:
                try:
                    best = entry["data_table"][entry["mask"]][0]
                    # if best is not None:
                    #     if best["Object Name"] != new["Object Name"]:
                    #         name_mask = entry["data_table"]["Object Name"] == \
                    #                     best["Object Name"]
                    #
                    #         alt = entry["data_table"][name_mask]
                    #
                    #         if len(alt) == 1:
                    #             best = alt
                    #         else:
                    #             check =[
                    #             x for x in entry["data_table"]["Object Name"]
                    #             if any(y.replace(" ", "") in x.replace(" ", "")
                    #             for y in self.alias)]
                    #
                    #             if len(check) > 0:
                    #
                    #                 best = entry["data_table"][
                    #                     entry["data_table"]["Object Name"] ==
                    #                     check[0]]
                    #             else:
                    #                 best = new
                    #     else:
                    #         best = new
                    # else:
                    #     best = new
                except IndexError:
                    pass


            if best is not None:
                self.alias.append(best["Object Name"])
                self.NED_name = str(best["Object Name"])
                self.NED_ra = float(best["RA(deg)"])
                self.NED_dec = float(best["DEC(deg)"])
                self.NED_redshift = float(best["Redshift"])
                self.NED_coords = SkyCoord(
                    str(self.NED_ra) + " " + str(self.NED_dec),
                    unit=(u.deg, u.deg))
                self.NED_offset = self.NED_coords.separation(self.coords).arcsec

class Transient:

    def __init__(self, points, entries, name, path=None):
        self.points = points
        self.entries = entries
        self.name = name

        obs = self.entries[0]
        self.NED_ra = obs.NED_ra
        self.NED_dec = obs.NED_dec
        self.NED_redshift = obs.NED_redshift

        dt = np.dtype([
            ('Galaxy Name', "S50"),
            ('Alias', "S50"),
            ("Date", np.float),
            ("Field Image", "S50"),
            ("X Pixel", np.float),
            ("Y Pixel", np.float),
            ("RA", np.float),
            ('Dec', np.float),
            ("Galactic L", np.float),
            ("Galactic B", np.float),
            ("Number Up", np.int),
            ("Number Down", np.int),
            ("Random Forest Classifier Score", np.float),
            ('Subtracted Image Counts', np.int),
            ("Reference Image Counts", np.int),
            ("Offset From Galaxy", np.float),
            ("Redshift", np.float),
            ("Additional Info", "S50"),
            ("Notes", "S50"),
            ("Image Name", "S50"),
            ("NED RA", np.float),
            ("NED Dec", np.float),
            ("NED Redshift", np.float),
            ("NED Offset", np.float)
        ])

        self.data_table = np.zeros_like(self.entries, dtype=dt)

        for i, obs in enumerate(self.entries):
            self.data_table[i] = np.array([
                (obs.galaxy_name, str(obs.alias),
                 float(str(obs.date)), str(obs.field_image),
                 obs.xpixel, obs.ypixel, obs.ra.deg, obs.dec.deg,
                 obs.galactic_l, obs.galactic_b, obs.n_up, obs.n_down,
                 obs.rfc_score,
                 obs.sub_img_counts, obs.ref_img_counts, obs.offset_from_galaxy,
                 obs.redshift, obs.additional_info, obs.notes, obs.image_name,
                 obs.NED_ra, obs.NED_dec, obs.NED_redshift, obs.NED_offset)],
                dtype=dt
            )

        self.AGN = False
        for det in self.entries:
            if det.AGN:
                self.AGN = True

        self.ra = self.data_table["RA"]
        self.dec = self.data_table["Dec"]
        self.redshift = self.data_table["Redshift"][
            ~np.isnan(self.data_table["Redshift"])]

        # print self.redshift, np.mean(self.redshift)

        self.n_entries = len(points)

        self.model = np.nan
        self.ll = np.nan
        self.ll_per_dof = np.nan
        self.true_ll = np.nan
        self.true_ll_per_dof = np.nan

        if self.n_entries > threshold_n_points:
            self.sufficient_points = True
        else:
            self.sufficient_points = False

        if self.sufficient_points:
            self.interesting = True
        else:
            self.interesting = False

        if path is None:
            if self.AGN:
                sub_dir = "AGN/"
            else:
                sub_dir = "transients/"

            self.save_path = output_dir + "plots/" + sub_dir + \
                             str(self.n_entries) + "/" + self.name + \
                             "/lightcurve.pdf"
        else:
            self.save_path = path

        self.make_lightcurve()

    def print_output(self):

        to_print = ["Alias", "Date", "Redshift", "RA", "Dec",
                    "Offset From Galaxy", "Galaxy Name"]

        print "\n", self.name, "\n"
        print tabulate(self.data_table[to_print], to_print)

    def make_lightcurve(self):
        fig = plt.figure()
        x = self.data_table["Date"]
        y = self.data_table["Subtracted Image Counts"]
        err = 0.1 * y
        weighted_err = 0.1 * (np.max(y) - np.min(y))
        plt.errorbar(x, y, yerr=err, fmt="o", ecolor="r")
        plt.xlabel("Date (MJD)")
        plt.ylabel("Subtracted Image Counts")

        max_count = max(y)
        max_index = list(y).index(max_count)
        max_time = self.data_table["Date"][max_index]

        pinit = lc.default(max_count)
        if len(x) >=threshold_n_points:
            def llh_weighted(p):
                time = x - max_time
                model = lc.fitfunc(time, p)
                ll = np.sum(((y - model) / weighted_err) ** 2)
                return ll

            def llh(p):
                time = x - max_time
                model = lc.fitfunc(time, p)
                ll = np.sum(((y - model) / err) ** 2)
                return ll

            out = optimize.minimize(
                llh, pinit, method='L-BFGS-B',
                bounds=lc.return_loose_bounds())

            self.ll = llh_weighted(out.x)

            self.true_ll = llh(out.x)

            self.ll_per_dof = self.ll / (
                float(self.n_entries) - len(pinit))

            self.true_ll_per_dof = self.true_ll / (
                float(self.n_entries) - len(pinit))

            self.model = out.x

            plot_x = np.linspace(min(x), max(x), 100)
            plot_y = lc.fitfunc(plot_x - max_time, self.model)
            plt.plot(plot_x, plot_y)
            plt.annotate(
                "ll per dof = " + "{0:.4f}".format(self.ll_per_dof) + "\n" + \
                "true ll per dof = " + "{0:.4f}".format(self.true_ll_per_dof),
                xy=(0.7, 0.8), xycoords="axes fraction")

        plt.title(self.name)

        dir = os.path.dirname(self.save_path)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        plt.savefig(self.save_path)

        if not np.isnan(self.ll_per_dof) and not self.AGN:

            ranked_path = output_dir + "plots/ranked_lightcurves/" + \
                          "{0:.4f}".format(self.ll_per_dof) + "_" + \
                          str(self.n_entries) + "_" + self.name + ".pdf"

            if not os.path.isdir(os.path.dirname(ranked_path)):
                os.makedirs(os.path.dirname(ranked_path))
            plt.savefig(ranked_path)

        plt.close()

class Galaxy(Transient):

    def __init__(self, points, entries, name):
        Transient.__init__(self, points, entries, name)
        self.transients = []
        self.clf = dict()
        self.find_transients()

    def make_map(self):
        fig = plt.figure()

        ax1 = plt.subplot(311)

        ra = self.data_table["RA"]
        dec = self.data_table["Dec"]
        date = np.array(self.data_table["Date"])
        time = date - np.min(date)

        plt.scatter(self.NED_ra, self.NED_dec, label="GALAXY", marker="x")

        cm = plt.cm.get_cmap('RdYlGn_r')
        sc = plt.scatter(
            ra, dec, c=time, s=15,
            cmap=cm)
        cbar = plt.colorbar(sc)
        cbar.set_label("Days since First Detection")

        gap = 0.0003
        ax1.set_xlim(min(min(ra), self.NED_ra) - gap,
                     max(max(ra), self.NED_ra) + gap)
        ax1.set_ylim(min(min(dec), self.NED_dec) - gap,
                     max(max(dec), self.NED_dec) + gap)

        plt.xlabel("Right Ascension (Degrees)")
        plt.ylabel("Declination (Degrees)")

        ax2 = plt.subplot(312)
        plt.hist(self.data_table["NED Offset"][np.array(
                     [not np.isnan(x) for x in self.data_table["NED Offset"]])
                 ])

        plt.xlabel("Distance to NED galaxy (arcsec)")
        plt.ylabel("Count")

        try:
            labels = self.clf["Spatial"]

            ax3 = plt.subplot(313)

            cm = plt.cm.get_cmap('RdYlGn_r')
            sc = plt.scatter(
                ra, dec, c=labels, s=15,
                cmap=cm)

            ax3.set_xlim(min(min(ra), self.NED_ra) - gap,
                         max(max(ra), self.NED_ra) + gap)
            ax3.set_ylim(min(min(dec), self.NED_dec) - gap,
                         max(max(dec), self.NED_dec) + gap)

        except KeyError:
            pass

        dir = os.path.dirname(self.save_path)
        if not os.path.isdir(dir):
            os.makedirs(dir)
        path = dir + "/map.pdf"

        fig.set_size_inches(8, 7)
        fig.subplots_adjust(hspace=.5)

        plt.suptitle(self.name)

        plt.savefig(path)
        plt.close()

    def search_for_clusters(self):

        if self.n_entries > 2 * threshold_n_points:

            try:

                vars = ["ra_deg", "dec_deg", "rfc_score"]

                data = np.array([[getattr(x, var) for var in vars]
                                 for x in self.entries])

                k_means = cluster.MeanShift()
                k_means.fit(data)

                self.clf["Spatial"] = k_means.labels_

                vars.append("date_mjd")

                all_data = np.array([[getattr(x, var) for var in vars]
                                 for x in self.entries])

                all_k_means = cluster.MeanShift()
                all_k_means.fit(all_data)

                # print all_k_means, all_k_means.labels_, all_k_means.fit(all_data)
                #
                # raw_input("prompt")


                self.clf["Time"] = all_k_means.labels_

            except ValueError:
                pass

    def find_transients(self):
        self.search_for_clusters()


        obs = np.array(self.entries)
        for clf_name, labels in self.clf.iteritems():

            lls = []
            trans = []
            paths = []

            clf = np.array(labels)
            for category in set(labels):
                mask = clf == category
                entries = obs[mask]
                points = np.array(self.points)[mask]

                name = self.name + "_" + clf_name + "_" + str(category)

                path = os.path.dirname(self.save_path) + "/" + \
                    clf_name + "_" + str(category) + "_lightcurve.pdf"

                new = Transient(points, entries, name, path=path)

                lls.append(new.true_ll)
                trans.append(new)
                paths.append(path)

            n_dof = float(self.n_entries - len(lc.default()) * len(set(labels)))

            ll = np.sum(lls)

            ll_per_dof = ll/n_dof

            if ll_per_dof < self.true_ll_per_dof:
                self.transients.extend(trans)

            else:
                for path in paths:
                    os.system("rm '" + str(path) + "'")

class Database:
    """
    Class containing all the individual entries from the ASASSN data, as well
    as matched information from NED regarding redshift/name.
    """

    def __init__(self):
        os.system("rm -rf " + output_dir + "/*")

        self.data_path = "/afs/ifh.de/user/s/steinrob/scratch/ASASSN_data/"
        self.filenames = [x for x in os.listdir(self.data_path) if x[0] != "."]

        self.raw_entries = []

        self.extract()

        print "Read in", len(self.raw_entries), "entries"

        self.n_objects = 0
        self.n_not_AGN = 0
        self.n_enough_points = 0
        self.n_interesting = 0

        self.combined_entries = dict()
        self.group_entries()


    def extract(self):
        """Extracts the individual observations from the ASASSN data folders,
        with observations grouped by day of detection."""
        for f in tqdm(self.filenames):
            file_path = self.data_path + f
            with open(file_path, 'rb') as file:
                reader = csv.reader(file, delimiter=' ', quotechar='|')
                for row in reader:
                    info = [x for x in row if x != ""]

                    if len(info) != 18:
                        print info
                        raise Exception("Incorrect Row Width!")

                    self.raw_entries.append((Observation(info)))

    def group_entries(self):
        """Joins individual observations together to create
        transient/variable histories. Currently groups each observation by the
        name of its host galaxy, as determined by reference to NED."""

        for i, primary_entry in enumerate(tqdm(self.raw_entries)):
            points = []
            entries = []
            name = str(primary_entry.alias[-1])
            for j, secondary_entry in enumerate(self.raw_entries[i:]):
                if str(secondary_entry.alias[-1]) == name:
                    points.append(j+i)
                    entries.append(secondary_entry)

            # If an entry has not already been created, adds a new entry for
            # the given NED name.

            if name not in self.combined_entries.keys():
                new = Galaxy(points, entries, name)
                self.combined_entries[name] = new

                self.n_objects += 1
                if not new.AGN:
                    self.n_not_AGN += 1

                if new.sufficient_points:
                    self.n_enough_points += 1

                if new.interesting:
                    self.n_interesting += 1

        print "There are", self.n_objects, "objects."
        print "There are", self.n_not_AGN, "that are not flagged AGNs."
        print "There are", self.n_enough_points, "objects with more than",
        print threshold_n_points, "datapoints."
        print "In total, there are", self.n_interesting, "interesting objects."

    def plot_histograms(self):
        attributes = ["NED_ra", "ra", "NED_dec", "dec", "n_entries",
                      "redshift", "NED_redshift", ]
        data = [[] for i in attributes]

        for obj in self.combined_entries.values():
            for i, atr in enumerate(attributes):
                val = np.mean(getattr(obj, atr))
                if not np.isnan(val):
                    data[i].append(val)

        n_rows = len(attributes) / 2 + len(attributes) % 2

        fig = plt.figure()
        for j, dataset in enumerate(data):
            plt.subplot(2, n_rows, j+1)
            plt.hist(dataset, bins=30)
            plt.xlabel(attributes[j])
            plt.yscale("log")

        fig.set_size_inches(n_rows * 4, 7)
        fig.subplots_adjust(hspace=.5)
        plt.savefig(user_dir + "/plots/histogram.pdf")
        plt.close()

    def galaxy_plots(self):

        ranked_dir = output_dir + "/plots/ranked_lightcurves/"
        if not os.path.isdir(ranked_dir):
            os.makedirs(ranked_dir)

        # os.system("rm " + ranked_dir + "/*")
        # os.system("rm -rf " + output_dir + "/*")

        for galaxy in self.combined_entries.itervalues():
            if galaxy.interesting:
                galaxy.make_map()

    def print_interesting(self):
        for galaxy in self.combined_entries.itervalues():
            if galaxy.interesting:
                galaxy.print_output()


    def plot_skymap(self):
        """Plots a map of the distribution of ASASSN candidates on the sky.
        Uses the n_entries as a colour scale.
        """
        ra = [x.NED_ra for x in self.combined_entries.values()]
        dec = [x.NED_dec for x in self.combined_entries.values()]
        n = np.log(np.array(
            [x.n_entries for x in self.combined_entries.values()]))

        int_ra = [x.NED_ra for x in self.combined_entries.values()
                  if x.interesting]
        int_dec = [x.NED_dec for x in self.combined_entries.values()
                   if x.interesting]
        int_n = np.log(np.array(
            [x.n_entries for x in self.combined_entries.values()
             if x.interesting]))

        full = [ra, dec, n, "full"]
        interesting = [int_ra, int_dec, int_n, "interesting"]

        for data in [full, interesting]:

            plt.figure()
            plt.subplot(111, projection="aitoff")

            cm = plt.cm.get_cmap('RdYlGn_r')
            sc = plt.scatter(
                wrap_around_180(data[0]), np.deg2rad(data[1]), c=data[2], s=5,
                cmap=cm)
            cbar = plt.colorbar(sc)
            cbar.set_label("Log(Number of entries)")

            path = "plots/ASASSN_skymap_" + data[3] + ".pdf"
            print "Saving to", path
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
