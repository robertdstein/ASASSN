from readin import Database, Galaxy, Observation, user_dir
import os
from sklearn.externals import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--reset", action="store_true")
parser.add_argument("-p", "--plot", action="store_true")
parser.add_argument("-n", "--ned", action="store_true")

cfg = parser.parse_args()

pickle_path = user_dir + "/pickle/stored_database.pkl"

if os.path.isfile(pickle_path) and not cfg.reset:
    print "Loading Database!"
    data = joblib.load(pickle_path)

else:
    print "Making Database!"
    data = Database()
    joblib.dump(data, pickle_path)

if cfg.plot:
    data.plot_lightcurves()
    data.plot_histograms()

# if cfg.ned:
#     data.match_to_ned()
#     joblib.dump(data, pickle_path)