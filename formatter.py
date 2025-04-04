def convert_time_to_seconds(timestring):
    """Converts a time string format 'H:M:S.f' to seconds."""
    h, m, s = map(float, timestring.split(':'))
    return h * 3600 + m * 60 + s


def format_lines(input_lines):
    """Formats each line of the input data to the specified output."""
    output_lines = []
    previous_end_time = 0.0

    for i, line in enumerate(input_lines):
        parts = line.split(',')
        time_str = parts[-1].strip()
        start_time = convert_time_to_seconds(time_str)

        # For all but the last line, calculate the end time of the current line
        # as the start time of the next line minus 0.01 seconds
        if i < len(input_lines) - 1:
            next_time_str = input_lines[i + 1].split(',')[-1].strip()
            end_time = convert_time_to_seconds(next_time_str) - 0.01
            output_line = f"{start_time:.2f} {end_time:.2f} [CHORD]"
        else:
            # Special mark for the last line
            output_line = f"{start_time:.2f} [TO_BE_FILLED] [CHORD]"

        # Add the formatted line to the output
        output_lines.append(output_line)

    return output_lines


def format_time_data(input_file_name):
    """
    Reads an input file, processes each line to convert time to seconds,
    formats it, and writes the result to a new file with '_formatted'
    appended to the original filename.
    """
    # Read the input file
    with open(input_file_name, 'r') as file:
        input_lines = file.readlines()

    # Process and format the input lines
    output_lines = format_lines(input_lines)

    # Create the formatted filename
    formatted_filename = f"{input_file_name.split('.')[0]}_formatted.txt"

    # Write the formatted lines to the file
    with open(formatted_filename, 'w') as file:
        for line in output_lines:
            file.write(f"{line}\n")

    return formatted_filename


