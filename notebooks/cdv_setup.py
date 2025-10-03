import numpy as np
import pandas as pd
import torch
import xarray as xr
import shapely

from dased.layout import DASLayout
from dased.helpers.srcloc import SurfaceField_Distribution

x_min, x_max = 100, 2100
y_min, y_max = 0, 2000

topo_data = xr.load_dataarray("data/topo_data.nc")
topo_data = topo_data.sel(
    x=slice(x_min, x_max), y=slice(y_min, y_max))

prior_dist = SurfaceField_Distribution(
    distribution=torch.load(
        "data/surface_displacement_prior.pt",
        weights_only=False).forward(),
    topo_data=topo_data,
    depth=300,
)


nodes = pd.read_csv("data/nodes_full.csv")[['easting', 'northing']].values[:835]

shoulder_area = shapely.convex_hull(
    shapely.geometry.MultiPoint(nodes[:378])).buffer(0)

full_area = shapely.concave_hull(
    shapely.geometry.MultiPoint(nodes), ratio = 0.1).buffer(20)

from shapely import GeometryCollection, LineString, Point, MultiPolygon, Polygon

big_crack = LineString([[1440, 1480], [1480, 1400]]).buffer(5)
rockfield_1 = LineString([[1410, 1470], [1440, 1390]]).buffer(15)
rockfield_2 = LineString([[1150, 1440], [1240, 1470], [1210, 1450], [1150, 1440]]).buffer(15)
big_rockfield = LineString(
    [[1020, 1530], [1050, 1520], [1105, 1505], [1090, 1480],
     [1060, 1450], [1030, 1420], [950, 1440], [990, 1470], [1020, 1530]]).buffer(15)
big_rockfield = Polygon(list(big_rockfield.exterior.coords))

rockfield_3 = LineString([[770, 1530], [850, 1500],]).buffer(15)
rockfield_4 = LineString([[680, 1440], [710, 1420],]).buffer(15)

obstacles = GeometryCollection(
    [big_crack, rockfield_1, rockfield_2, big_rockfield, rockfield_3, rockfield_4])

design_space_shoulder = shoulder_area.difference(obstacles)
design_space_full = full_area.difference(obstacles)


x = torch.from_numpy(topo_data['x'].values).float()
y = torch.from_numpy(topo_data['y'].values).float()

X, Y = torch.meshgrid(x, y, indexing='ij')
Z = torch.from_numpy(topo_data.values).float()

prior_grid = prior_dist.log_prob(
    torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=1)).reshape(X.shape).exp().detach().numpy()

test_geometry = np.array(
    [[1550.0, 1400.0], [ 850.0, 1400.0], [ 700.0,  700.0], ])

existing_geometry = np.array(
    [[1570, 1330], [1500, 1330], [1450, 1310], [1380, 1340],
     [1330, 1340], [1300, 1310], [1150, 1330], [1000, 1310],
     [ 800, 1370], [ 700, 1390], [ 610, 1440], [ 700, 1390],
     [ 800, 1370], [1000, 1310], [1150, 1330], [1250, 1310],
     [1250, 1530], [1100, 1450], [1070, 1400], [1280, 1400],],
)

# tomo_roi = shapely.affinity.scale(
#     shoulder_area.envelope, xfact=0.6, yfact=0.7, origin=(800, 1400)
# )
tomo_roi = shoulder_area.buffer(-10)
