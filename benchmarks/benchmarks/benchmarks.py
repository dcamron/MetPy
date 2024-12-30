
import numpy as np
import xarray as xr

from metpy.calc import cape_cin, el, lcl, moist_lapse, parcel_profile, saturation_vapor_pressure
from metpy.units import units

def index_xarray_data():
    """Create data for testing that index calculations work with xarray data."""
    pressure = xr.DataArray([850., 700., 500.], dims=('isobaric',), attrs={'units': 'hPa'})
    temp = xr.DataArray([[[[296., 295., 294.], [293., 292., 291.]],
                          [[286., 285., 284.], [283., 282., 281.]],
                          [[276., 275., 274.], [273., 272., 271.]]]] * units.K,
                        dims=('time', 'isobaric', 'y', 'x'))

    profile = xr.DataArray([[[[289., 288., 287.], [286., 285., 284.]],
                             [[279., 278., 277.], [276., 275., 274.]],
                             [[269., 268., 267.], [266., 265., 264.]]]] * units.K,
                           dims=('time', 'isobaric', 'y', 'x'))

    dewp = xr.DataArray([[[[294., 293., 292.], [291., 290., 289.]],
                          [[284., 283., 282.], [281., 280., 279.]],
                          [[274., 273., 272.], [271., 270., 269.]]]] * units.K,
                        dims=('time', 'isobaric', 'y', 'x'))

    dirw = xr.DataArray([[[[180., 180., 180.], [180., 180., 180.]],
                          [[225., 225., 225.], [225., 225., 225.]],
                          [[270., 270., 270.], [270., 270., 270.]]]] * units.degree,
                        dims=('time', 'isobaric', 'y', 'x'))

    speed = xr.DataArray([[[[20., 20., 20.], [20., 20., 20.]],
                           [[25., 25., 25.], [25., 25., 25.]],
                           [[50., 50., 50.], [50., 50., 50.]]]] * units.knots,
                         dims=('time', 'isobaric', 'y', 'x'))

    return xr.Dataset({'temperature': temp, 'profile': profile, 'dewpoint': dewp,
                       'wind_direction': dirw, 'wind_speed': speed},
                      coords={'isobaric': pressure, 'time': ['2020-01-01T00:00Z']})

class TimeSuite:
    """Benchmark times for calculations."""

    def setup(self):
        self.p = np.array([959., 779.2, 751.3, 724.3, 700., 269.]) * units.mbar
        self.t = np.array([22.2, 14.6, 12., 9.4, 7., -38.]) * units.celsius
        self.td = np.array([19., -11.2, -10.8, -10.4, -10., -53.2]) * units.celsius
        self.ds = index_xarray_data()
        self.slice = self.ds.isel(isobaric=0)

    def time_cape_cin(self):
        """Benchmark cape_cin."""
        parcel_prof = parcel_profile(self.p, self.t[0], self.td[0])
        cape_cin(self.p, self.t, self.td, parcel_prof)

    def time_el(self):
        """Benchmark el.."""
        el(self.p, self.t, self.td)

    def time_lcl(self):
        """Benchmark LCL calculation."""
        lcl(self.p[0], self.t[0], self.td[0])

    def time_moist_lapse(self):
        """Benchmark moist lapse calculation."""
        moist_lapse(self.p, self.t[0])

    def time_saturation_vapor_pressure(self):
        """Benchmark saturation vapor pressure calculation."""
        saturation_vapor_pressure(self.t[0])

    def time_saturation_vapor_pressure_grid(self):
        """Benchmark saturation vapor pressure on a grid."""
        saturation_vapor_pressure(self.slice.temperature)

    def time_lcl_grid(self):
        """Benchmark LCL calculation on a grid."""
        lcl(self.slice.isobaric, self.slice.temperature, self.slice.dewpoint)

