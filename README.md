# Using Ant Colony Optimization for the Travelling Salesman Problem

-   by Pineda, Lara Patricia B.
-   algorithm logic referenced from ant colony optimization series of videos from https://www.youtube.com/@hklam2368
-   created March 2024

## How to run:

1. Install the latest version of python3 and the numpy library in your system.

2. Download the sample datasets and main.py.

3. Navigate to the directory where the downloaded files are located.

4. Enter the command `python3 main.py` in the terminal to run the program.

5. Program:

-   You will be asked to enter values for the following variables used by the program:
    a. dataset file name
    b. number of ants
    c. number of iterations
    d. evaporation rate
    e. alpha value
    f. veta value
    g. Q value

-   You may just press ENTER to use the default values of the variables.
-   The default dataset file used is `dataset.txt`
    -   datasets are expected to have the total number of cities in the fist line
    -   each ith line after the first line contains the distance of the ith city to all other cities, delimited by whitespace
