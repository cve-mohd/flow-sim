import case_study_preissmann

for peak_inflow in range(10000, 26000, 2000):
    for roseires_initial_stage in range(487, 492, 1):
        case_study_preissmann.run(peak_inflow=peak_inflow, initial_roseires_stage=roseires_initial_stage)
        