QuantifiedMe
============

[![Build](https://github.com/ErikBjare/quantifiedme/actions/workflows/build.yml/badge.svg)](https://github.com/ErikBjare/quantifiedme/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/ErikBjare/quantifiedme/branch/master/graph/badge.svg?token=zZ7hwaw9tR)](https://codecov.io/gh/ErikBjare/quantifiedme)

Loading and plotting of various Quantified Self data sources.

You can see an example notebook with fake data built in CI [down below](#notebooks).

**Note:** This code is only used by me, as far as I know, but I encourage you to try it out anyway, and report or send PRs for any issues you encounter. I will try to keep it tidy and somewhat usable.


## Features

The code in this repository generally loads data from some source into a Pandas dataframe, and provides tools to process, aggregate, and plot the data. 

This makes it a useful toolkit for exploratory data analysis with Jupyter notebooks, for example.

Types of data supported:

 - Time tracking data (from ActivityWatch, Toggl, SmarterTime)
 - Sleep data (from Fitbit, Oura, Whoop)
 - Heartrate data (from Fitbit, Oura, Whoop)
 - Location data (from Google Location History)
   - Includes basic plotting of time spent in a certain location.
   - Includes function for computing the colocation time of two location histories (time spent together).
 - Habit data (from HabitBull)
   - Includes a calendar plot.
   - Easy to adapt to any other habit app that supports CSV export
 - Drug consumption (from QSlang)

Can load data from:

 - ActivityWatch
 - Fitbit
 - Whoop
 - Oura
 - EEG devices (WIP)
 - ...and more (see `src/quantifiedme/load/`)

It also contains a bunch of useful tools for aggregating or otherwise deriving data from the sources, including helper tools for combining multiple sources for the same type of data (see `src/quantifiedme/derived`).

## Notebooks

There is currently only one example notebook.

 - Dashboard - Preview at: https://erik.bjareholt.com/quantifiedme/Dashboard.html
   - Uses ActivityWatch and SmarterTime data from multiple devices (desktop, laptop, phone) to create a unified overview of time spent. 
   - Used by me as a sort of personal-productivity dashboard.
   - Plots things like:
     - hours worked per day (and on what)
     - which categories are consuming most of my time on a 30-day and 365-day basis
     - how much I make in "fictional salary" over time (by assigning an hourly wage to each category)

I also have a collection of private notebooks for exploratory analysis, which I hope to share later.


## Configuration

The configuration is used to specify where data files are located, as well as a few settings.

An example configuration file is provided in `config.example.toml`.


## Related projects

 - [HPI](https://github.com/karlicoss/HPI) ("Human Programming Interface") by @karlicoss
