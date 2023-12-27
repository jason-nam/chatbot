import more_itertools as mit

data_path = "pb.txt"

# Defining lines as a list of each line
with open(data_path, 'r', encoding='utf-8') as f:
  raw_lines = f.read().split('\n')

raw_lines.reverse()
lines = []

for line in raw_lines:
    # split line into parts
    line_split = line.split(' +++$+++ ')
    # append tuple of character and line
    line_num = int(line_split[0][1:])

    current_line = line_split[4].strip()
    # append tuple of line num, character and line
    lines.append((line_num, current_line))
# make sure the lines are in order
lines = sorted(lines, key=lambda x: x[0])

# group lines by scene
by_scene = [list(group) for group in mit.consecutive_groups(lines, lambda x: x[0])]

dialog_only = [[dialog_line[1] for dialog_line in dialog_group] 
                for dialog_group in by_scene]

dialog_combos_nested = [list(map(list, zip(dialog_group, dialog_group[1:]))) for dialog_group in dialog_only]

dialog_combos = [combo for combos in dialog_combos_nested for combo in combos]

# print dialog combos:

print(dialog_combos)

