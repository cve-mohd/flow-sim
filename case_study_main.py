import case_study_preissmann_2pass, case_study_preissmann

for roseires_level_ in [485]:
    case_study_preissmann_2pass.run(roseires_level_)
    #case_study_preissmann.run(roseires_level_)
    print(f'Finished {roseires_level_}')
    
import results_processor
