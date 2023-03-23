QuantifiedMe
============

[![Build](https://github.com/ErikBjare/quantifiedme/actions/workflows/build.yml/badge.svg)](https://github.com/ErikBjare/quantifiedme/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/ErikBjare/quantifiedme/branch/master/graph/badge.svg?token=zZ7hwaw9tR)](https://codecov.io/gh/ErikBjare/quantifiedme)

Loading and plotting of various Quantified Self data sources.

You can see an example notebook with fake data built in CI here: https://erik.bjareholt.com/quantifiedme/aw-server-rust/Dashboard.html

**Note:** The code is not in a condition to be used by others than me, but I encourage you to try to anyway, and report or send PRs for any issues you encounter.


## Features

The code in this repository generally loads data from some source into a Pandas dataframe, and may provide some basic plotting for that data. 
This makes it useful for exploratory data analysis with Jupyter notebooks, for example.

Types of data supported:

 - Fitness tracker data (from Fitbit, Oura, Whoop)
   - In particular heart rate data
 - Time tracking data (from ActivityWatch, Toggl, SmarterTime)
 - Location data (from Google Location History)
   - Includes basic plotting of time spent in a certain location.
   - Includes function for computing the colocation time of two location histories (time spent together).
 - Habit data (from HabitBull)
   - Includes a calendar plot.
 - Drug consumption (from QSlang)

Can load data from:

 - Fitbit
 - Whoop
 - Oura
 - ...and more (see `src/quantifiedme/load/`)


## Notebooks

There is currently only one example notebook.

 - Dashboard
   - Uses ActivityWatch and SmarterTime data from multiple devices (desktop, laptop, phone) to create a unified overview of time spent. 
   - Used by me as a sort of personal-productivity dashboard.
   - Plots things like:
     - hours worked per day (and on what)
     - which categories are consuming most of my time on a 30-day and 365-day basis
     - how much I make in "fictional salary" over time (by assigning an hourly wage to each category)

I also have a collection of private notebooks for exploratory analysis, some of which I hope to share later.


## Configuration

The configuration is used to specify where data files are located, as well as a few settings.

An example configuration file is provided in `config.example.toml`.


## Related projects

 - [HPI](https://github.com/karlicoss/HPI) ("Human Programming Interface") by @karlicoss
