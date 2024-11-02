import river
import preissmann_model
import lax_model


river = river.River(0.00061, 0.023, 120., 29000.)

"""
p_model = preissmann_model.PreissmannModel(river, 1, 3600, 1000, 20*3600)
p_model.optimized_solve()
p_model.save_results()
"""

laxmodel = lax_model.LaxModel(river, 175, 1000, 20*3600)
laxmodel.solve()
laxmodel.save_results(time_steps_to_save=21)
