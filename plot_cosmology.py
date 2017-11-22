import numpy as np
import astropy
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

small_radius = 7 * u.kpc
large_radius = 50 * u.kpc

z_range = np.linspace(0.02, 1.50, 75)

small_angles = []
large_angles = []

for z in z_range:
    scale = cosmo.arcsec_per_kpc_proper(z)
    small_angles.append((small_radius * scale).value)
    large_angles.append((large_radius * scale).value)

plt.figure()

plt.plot(z_range, small_angles, label="Typical Small Galaxy (Radius 7 Kpc)")
plt.plot(z_range, large_angles, label="Typical Large Galaxy (Radius 50 Kpc)")
plt.axhline(y=10, color='r', linestyle='-', label="ASASSN cut")
plt.legend()

path = "plots/galaxy_radius.pdf"

print "Saving to", path

plt.savefig(path)
plt.close()

z = 0.014350
d1 = cosmo.angular_diameter_distance(z)
d2 = cosmo.luminosity_distance(z)
print z, d1, d2
print d1.to("au"), d1.to("lightyear")
# d = astropy.coordinates.Distance(z=z)