import case_study_preissmann

for roseires_level_ in [485, 487, 490, 491]:
    case_study_preissmann.run(roseires_level_)
    print(f'Finished {roseires_level_}')
    
import results_processor
