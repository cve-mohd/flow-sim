from .model import run

print("Start.")
run(verbose=0, file='gerd.xlsx', inflow_hyd_func=None)
print("Finished regulated scenario.")
run(verbose=0, file='no_gerd.xlsx', inflow_hyd_func=None, with_gerd=False)
print("Finished natural scenario.")

"""
for r in [487, 490]:
    folder = f'cases\\gerd_roseires\\results\\{r}\\'
    run(verbose=1, initial_roseires_level=r, folder=folder, jammed_sluice_gates=0, jammed_spillways=0, file='baseline.xlsx')
    run(verbose=1, initial_roseires_level=r, folder=folder, jammed_sluice_gates=0, jammed_spillways=1, file='spillway_jam.xlsx')
    run(verbose=1, initial_roseires_level=r, folder=folder, jammed_sluice_gates=1, jammed_spillways=1, file='double_jam.xlsx')

# py -m cases.gerd_roseires.main
"""