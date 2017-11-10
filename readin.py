import numpy as np
import csv
import os
from astropy.time import Time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

user_dir = "/afs/ifh.de/user/s/steinrob/Desktop/python/ASASSN"

class Database:

    def __init__(self):
        self.data_path = "/afs/ifh.de/user/s/steinrob/scratch/ASASSN_data/"
        self.filenames = [x for x in os.listdir(self.data_path) if x[0] != "."]

        self.raw_entries = []

        self.extract()

        print "Read in", len(self.raw_entries), "entries"

        self.combined_entries = dict()
        self.group_entries()

        print "Combined to form", len(self.combined_entries), "Objects."

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
        for primary_entry in self.raw_entries:
            points = []
            name = primary_entry.galaxy_name
            for j, secondary_entry in enumerate(self.raw_entries):
                if secondary_entry.galaxy_name == name:
                    points.append(j)

            if name not in self.combined_entries.keys():
                self.combined_entries[name] = Object(points, name)

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



class Object:

    def __init__(self, points, name):
        self.points = points
        self.name = name
        self.n_entries = len(points)

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
        self.galaxy_name = info[14]

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