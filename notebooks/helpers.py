import os
import numpy as np
from sympy import N
import xarray as xr
import pandas as pd
from pyrocko import cake
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

class PhaseLookupCalculator:
    def __init__(
        self,
        lat,
        lon,
        distance_grid,
        source_depth_grid,
        receiver_depth_grid,
        data_dir="data",
    ):
        self.lat = lat
        self.lon = lon
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        self.distance_grid = distance_grid
        self.source_depth_grid = source_depth_grid
        self.receiver_depth_grid = receiver_depth_grid

        self.model = cake.load_model(
            fn=None, crust2_profile=(self.lat, self.lon))

    def _filename(self, group):
        return os.path.join(
            self.data_dir, f"{group}_lookup_{self.lat:.2f}_{self.lon:.2f}.nc"
        )

    def __call__(self, phase_groups):
        """
        phase_groups: dict, e.g. {'p': ['p', 'P'], 's': ['s', 'S']}
        Returns: dict of group -> xr.Dataset
        """
        results = {}
        for group, phases in phase_groups.items():
            fname = self._filename(group)
            print(f"Loading {fname}")
            
            if os.path.exists(fname):
                results[group] = xr.open_dataset(fname)
            else:
                results[group] = self._calculate_lookup(phases)
                results[group].to_netcdf(fname)
        return results

    def _calculate_lookup(self, phases):
        arr_shape = (
            len(self.receiver_depth_grid),
            len(self.distance_grid),
            len(self.source_depth_grid),
        )
        ds = xr.Dataset(
            {
                "arrival_time": (
                    ("receiver_depth", "distance", "source_depth"),
                    np.full(arr_shape, np.nan)
                ),
                "incidence_angle": (
                    ("receiver_depth", "distance", "source_depth"),
                    np.full(arr_shape, np.nan)
                ),
                "takeoff_angle": (
                    ("receiver_depth", "distance", "source_depth"),
                    np.full(arr_shape, np.nan)
                ),
                "efficiency": (
                    ("receiver_depth", "distance", "source_depth"),
                    np.full(arr_shape, np.nan)
                ),
            },
            coords={
                "receiver_depth": self.receiver_depth_grid,
                "distance": self.distance_grid,
                "source_depth": self.source_depth_grid,
            },
        )
        for rec_i, rec_depth in enumerate(
            tqdm(self.receiver_depth_grid, desc="Receiver Depth", leave=False)
        ):
            for src_i, src_depth in enumerate(
            tqdm(self.source_depth_grid, desc="Source Depth", leave=False)
            ):
                arrivals = self.model.arrivals(
                    self.distance_grid * cake.m2d,
                    phases=[cake.PhaseDef(p) for p in phases],
                    zstart=src_depth,
                    zstop=rec_depth,
                )
                df_list = []
                for a in arrivals:
                    df_list.append(
                        {
                            "distance": a.x * cake.d2m,
                            "arrival_time": abs(a.t),
                            "incidence_angle": (a.incidence_angle())%180,
                            "takeoff_angle": a.takeoff_angle(),
                            "phase": a.path.phase.definition(),
                            "efficiency": a.efficiency(),
                        }
                    )
                df = pd.DataFrame(df_list)
                # remove rows with NaN values
                try:
                    df = df.loc[df.groupby('distance')['arrival_time'].idxmin()]
                    
                    for quantity in ["arrival_time", "incidence_angle", "takeoff_angle", "efficiency"]:
                        ds[quantity].values[
                            rec_i, :, src_i
                        ] = df[quantity].values
                except KeyError:
                    pass
        return ds
    
    def plot_velocity_model(self, ax=None):
        """
        Plot the velocity model (P and S velocities vs depth).
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 4))            
        else:
            fig = None

        # Get depth and velocities from the model
        depths = []
        vp = []
        vs = []
        for layer in self.model.layers():
            top = layer.ztop
            bottom = layer.zbot
            depths.extend([top, bottom])
            
            vp_i = layer.material(top).vp
            vs_i = layer.material(top).vs
            
            vp.extend([vp_i, vp_i])
            vs.extend([vs_i, vs_i])

        depths = np.array(depths)
        vp = np.array(vp)
        vs = np.array(vs)
        
        ax.plot(vp/1e3, depths/1e3, label='Vp (P-wave)', color='b')
        ax.plot(vs/1e3, depths/1e3, label='Vs (S-wave)', color='r')
        
        if fig is None:
            return ax
        
        ax.set_ylim(0, max(np.max(self.source_depth_grid), np.max(self.receiver_depth_grid)) / 1e3*2)
        
        ax.invert_yaxis()
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('Depth (km)')
        ax.legend()
        ax.set_title('Velocity Model')
        plt.show()