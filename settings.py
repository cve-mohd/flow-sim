def read_settings(file_name):
    settings = {}
    valid_keys = {
        'BED_SLOPE': float,
        'MANNING_COEFF': float,
        'WIDTH': (int, float),
        'LENGTH': (int, float),
        'SCHEME': str,
        'LAX_APPROX': str,
        'PREISSMANN_BETA': float,
        'TIME_STEP': (int, float),
        'SPATIAL_STEP': (int, float),
        'DURATION': (int, float),
        'TOLERANCE': float,
        'RESULTS_SIZE': tuple
    }

    with open(file_name, 'r') as file:
        for line in file:
            line = line.split('#')[0].strip()  # Remove comments and trim spaces
            if line:  # Ignore empty lines
                try:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    if key not in valid_keys:
                        print(f"Error: Invalid variable name '{key}'")
                        continue

                    expected_type = valid_keys[key]
                    evaluated_value = eval(value)

                    # Check if value type matches expected type
                    if isinstance(expected_type, tuple):
                        if not isinstance(evaluated_value, expected_type):
                            raise TypeError
                    elif not isinstance(evaluated_value, expected_type):
                        raise TypeError

                    settings[key] = evaluated_value

                except (ValueError, SyntaxError):
                    print(f"Error: Syntax error or invalid format in line '{line}'")
                except TypeError:
                    print(f"Error: Invalid data type for variable '{key}'")
    return settings

# Load settings from the text file
settings = read_settings('settings.txt')

# Assign values to variables and raise an error if any required setting is missing
try:
    BED_SLOPE = settings['BED_SLOPE']
    MANNING_COEFF = settings['MANNING_COEFF']
    WIDTH = settings['WIDTH']
    LENGTH = settings['LENGTH']

    SCHEME = settings['SCHEME']
    LAX_APPROX = settings['LAX_APPROX']
    PREISSMANN_BETA = settings['PREISSMANN_BETA']
    TIME_STEP = settings['TIME_STEP']
    SPATIAL_STEP = settings['SPATIAL_STEP']
    DURATION = settings['DURATION'] * 3600  # Convert duration from hours to seconds
    TOLERANCE = settings['TOLERANCE']
    RESULTS_SIZE = settings['RESULTS_SIZE']

except KeyError as e:
    print(f"Error: Missing required setting '{e.args[0]}'")

############                Upstream Boundary                   ############

US_INIT_DEPTH = 7.5
US_INIT_DISCHARGE = 1562.5
US_INIT_STAGE = 502.5

PEAK_DISCHARGE = 10000
PEAK_HOUR = 6

US_RATING_CURVE = {"base": 500, "coefficients": [327.23, 318.44, 70.26]}

############                Downstream Boundary                 ############

DS_INIT_DEPTH = 7.5
DS_INIT_DISCHARGE = 1562.5
DS_INIT_STAGE = 490

DS_RATING_CURVE = {"base": 466.7, "coefficients": [8266.62, 469.31, -2.64]}
