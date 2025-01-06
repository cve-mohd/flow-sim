import appkit, graph, hydrograph_curve, background, sim_time
from settings import TIME_STEP, SPATIAL_STEP
import csv

with open('Results//Depth.csv', 'r') as file:
    reader = csv.reader(file)
    data_y = [[float(i) for i in row] for row in reader]

with open('Results//Velocity.csv', 'r') as file:
    reader = csv.reader(file)
    data_V = [[float(i) for i in row] for row in reader]


app = appkit.App(1200, 700, 'River Simulation')

bg = background.Background(rect=(0, 50, 810, 660), interval=(1000, 2.5),
                           grid_lines=(16, 8), padding=[50, 10, 10, 50])

graph_ = graph.Graph(data_y, data_V, TIME_STEP, SPATIAL_STEP, t_upscaling=2, background=bg)

hydrograph_curve_ = hydrograph_curve.HydrographCurve(rect=(900, 50, 300, 200))

timer = sim_time.Timer(graph_=graph_)

app.activities['home'].addItem(bg)
app.activities['home'].addItem(graph_)
app.activities['home'].addItem(timer)
#app.activities['home'].addItem(hydrograph_curve_)

app.run()
