from .model import run

run(verbose=1, initial_roseires_level=490, jammed_sluice_gates=1, jammed_spillways=1, file='baseline.xlsx', sim_duration=120*3600)

raise
for r in [487, 490]:
    folder = f'cases\\gerd_roseires\\results\\{r}\\'
    run(verbose=1, initial_roseires_level=r, folder=folder, jammed_sluice_gates=0, jammed_spillways=0, file='baseline.xlsx')
    run(verbose=1, initial_roseires_level=r, folder=folder, jammed_sluice_gates=0, jammed_spillways=1, file='spillway_jam.xlsx')
    run(verbose=1, initial_roseires_level=r, folder=folder, jammed_sluice_gates=1, jammed_spillways=1, file='double_jam.xlsx')

# py -m cases.gerd_roseires.main