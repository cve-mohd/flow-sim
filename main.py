import river
import preissmann_model


river = river.River(0.00061, 0.023, 120, 29000)
model = preissmann_model.Model(river, 1, 3600, 1000, 20*3600)
model.run()

model.save_results()