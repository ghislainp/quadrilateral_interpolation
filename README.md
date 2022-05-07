# quadrilateral_interpolation

This code provides a fast quadrilateral interpolation to perform satellite image projection from swath geometry to any
geographical projection defined by a CRS.

The method and the code implementation are intent to be very fast but may not as accurate as other backward interpolation methods (bilinear, ...). 
It is suitable for coarse resolution imagery and was initially developed for OLCI. The calculation is based on:
[Interpolation using an arbitrary quadrilateral](https://www.particleincell.com/2012/quad-interpolation/)

The API follows the conventions of [xESMF](https://github.com/pangeo-data/xESMF) for the Resampler object and 
of [pyresample](https://github.com/pytroll/pyresample) for the grid definition. See [pyresample](https://github.com/pytroll/pyresample) for useful tools
 to create and manipulate grids.

This package was designed to work for satellite images opened with [satpy](https://github.com/pytroll/satpy) 
but can be used as a standalone projection tool as long as the basic grid definition information are provided (lat, lon for the source and CRS, pixel size and offset for the destination).


# Installation

Install with pip the master version.

```
pip install git+https://github.com/ghislainp/quadrilateral_interpolation.git
```

# Example

An example notebook is provided in the example/ directory. It requires to first download an OLCI image (~700Mb) in the native ESA format.

# Contributing

Pull requests are welcome to make this code faster, more accurate, nicer, or more useful.

