import appkit, visual
from settings import TIME_STEP
import csv

# Get the results from the .csv files.
with open('Results//Depth.csv', 'r') as file:
    reader = csv.reader(file)
    data_y = [[float(i) for i in row] for row in reader]

with open('Results//Velocity.csv', 'r') as file:
    reader = csv.reader(file)
    data_Q = [[float(i) for i in row] for row in reader]


# Declare a 'App' object.
app = appkit.App(1000, 700, 'River Simulation')

# Declare a 'Visual' object, specifying the data to be visualized.
visual = visual.Visual(data_y, data_Q, TIME_STEP)

# Add the 'Visual' object to the 'App' object.
app.activities['home'].addItem(visual)

# Run the application.
app.run()
