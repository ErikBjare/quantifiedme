QuantifiedMe
============

Loading and plotting of various Quantified Self data sources.

**Note:** The code is not in a condition to be used by others than me, but I encourage you to try to anyway, and report or send PRs for any issues you encounter.

## Features

The code in this repository generally loads data from some source into a Pandas dataframe, and may provide some basic plotting for that data. 
This makes it useful for exploratory data analysis with Jupyter notebooks, for example.

Types of data supported:

 - Time tracking data (from ActivityWatch, Toggl, SmarterTime)
 - Location data (from Google Location History)
   - Includes basic plotting of time spent in a certain location.
   - Includes function for computing the colocation time of two location histories (time spent together).
 - Habit data (from HabitBull)
   - Includes a calendar plot.
 - Oura data
 - QSlang data

## Notebooks

There is currently only one example notebook. It is used by me as a sort of personal-productivity dashboard and plots things like hours worked per day (and on what), which categories are consuming most of my time on a 30-day and 365-day basis, and how much I make in "fictional salary" over time (by assigning an hourly wage to each category).

## Configuration

The configuration is used to specify where data files are located, as well as a few settings.

An example configuration file is provided in `config.toml.example`.
