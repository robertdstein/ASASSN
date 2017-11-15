import numpy as np
import csv
import os
from astropy.time import Time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tabulate import tabulate
import os

user_dir = "/afs/ifh.de/user/s/steinrob/Desktop/python/ASASSN"

threshold_n_points = 3

class Database:

    def __init__(self):
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

        self.plot_histograms()


    def extract(self):
        for f in self.filenames:
            file_path = self.data_path + f
            with open(file_path, 'rb') as file:
                reader = csv.reader(file, delimiter=' ', quotechar='|')
                for row in reader:
                    info = [x for x in row if x is not ""]

                    if len(info) != 18:
                        print info
                        raise Exception("Incorrect Row Width!")

                    self.raw_entries.append((Observation(info)))

    def group_entries(self):

        for i, primary_entry in enumerate(self.raw_entries):
            points = []
            entries = []
            name = primary_entry.galaxy_name
            for j, secondary_entry in enumerate(self.raw_entries[i:]):
                if secondary_entry.galaxy_name == name:
                    points.append(j+i)
                    entries.append(secondary_entry)

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
        attributes = ["n_entries"]
        data = [[] for i in attributes]

        for obj in self.combined_entries.values():
            for i, atr in enumerate(attributes):
                data[i].append(getattr(obj, atr))

        binwidth = 1
        plt.figure()
        for j, dataset in enumerate(data):
            plt.subplot(len(data), 1, j+1)
            plt.hist(dataset, bins=range(
                min(dataset), max(dataset) + binwidth, binwidth))
            plt.xlabel(attributes[j])
            plt.yscale("log")
        plt.savefig(user_dir + "/plots/histogram.pdf")
        plt.close()

class Galaxy:

    def __init__(self, points, entries, name):
        self.points = points
        self.entries = entries
        self.name = name

        dt = np.dtype([
            ('Galaxy Name', "S50"),
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
            ("Image Name", "S50")
        ])

        self.data_table = np.zeros_like(self.entries, dtype = dt)

        for i, obs in enumerate(self.entries):
            self.data_table[i] = np.array([
                (obs.galaxy_name, float(str(obs.date)), str(obs.field_image),
                 obs.xpixel, obs.ypixel, obs.ra, obs.dec, obs.galactic_l,
                 obs.galactic_b, obs.n_up, obs.n_down, obs.rfc_score,
                 obs.sub_img_counts, obs.ref_img_counts, obs.offset_from_galaxy,
                 obs.redshift, obs.additional_info, obs.notes, obs.image_name)],
                dtype=dt
            )

        self.AGN = self.entries[0].AGN

        self.n_entries = len(points)

        if self.n_entries > threshold_n_points:
            self.sufficient_points = True
        else:
            self.sufficient_points = False

        if self.sufficient_points and not self.AGN:
            self.interesting = True
            self.make_lightcurve()
        else:
            self.interesting = False


    def print_output(self):

        to_print = ["Date", "Redshift", "RA", "Dec", "Number Up", "Number Down",
                    "Offset From Galaxy", "Random Forest Classifier Score",
                    "Notes"]

        if self.n_entries > 100 and not self.AGN:
            print "\n", self.name, "\n"
            print tabulate(self.data_table[to_print], to_print)

    def make_lightcurve(self):
        fig = plt.figure()
        x = self.data_table["Date"]
        y = self.data_table["Subtracted Image Counts"]
        err = np.sqrt(y) / self.data_table["Random Forest Classifier Score"]
        plt.errorbar(x, y, yerr=err, fmt="o", ecolor="r")
        plt.xlabel("Date (MJD)")
        plt.ylabel("Subtracted Image Counts")

        dir = "plots/objects/" + str(self.n_entries) + "/" + self.name
        if not os.path.isdir(dir):
            os.makedirs(dir)
        path = dir + "/lightcurve.pdf"
        plt.savefig(path)
        plt.close()


class Observation:

    def __init__(self, info):
        self.date = Time(float("245" + info[0]) - 2400000.5, format="mjd")
        self.date.out_subfmt = "date"
        self.field_image = info[1].split("/")
        self.xpixel = float(info[2])
        self.ypixel = float(info[3])
        self.ra = float(info[4])
        self.dec = float(info[5])
        self.galactic_l = float(info[6])
        self.galactic_b = float(info[7])
        self.n_up = int(info[8])
        self.n_down = int(info[9])
        self.rfc_score = float(info[10])
        self.sub_img_counts = int(info[11])
        self.ref_img_counts = int(info[12])
        self.offset_from_galaxy = float(info[13])

        g_name = info[14].strip(":")
        if "AGN" in g_name:
            self.galaxy_name = g_name[:-3]
            self.AGN = True
        else:
            self.galaxy_name = g_name
            self.AGN = False

        self.redshift = np.nan

        extra_info = info[15].split("(")

        if extra_info[1][0] not in ["-", "z"] \
                and not extra_info[1][0].isdigit():
            rest = extra_info[2:]
            extra_info = ["(".join(extra_info[:2])]
            extra_info.extend(rest)

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

Database()