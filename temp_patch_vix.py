with open("prometheus/data/engine.py") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "def get_india_vix(" in line:
        new_lines.append(line)
        new_lines.append("        return 15.0\n")
        skip = True
        continue
    if skip:
        if "def get_fii_dii_data(" in line:
            skip = False
        else:
            continue
    new_lines.append(line)

with open("prometheus/data/engine.py", "w") as f:
    f.writelines(new_lines)

with open("prometheus/data/engine.py") as f:
    lines = f.readlines()
new_lines = []
skip = False
for line in lines:
    if "def get_vix(" in line:
        new_lines.append(line)
        new_lines.append("        return 15.0\n")
        skip = True
        continue
    if skip:
        if "def " in line and not line.startswith("        ") and "def get_vix" not in line:
            skip = False
        else:
            continue
    
    new_lines.append(line)

with open("prometheus/data/engine.py", "w") as f:
    f.writelines(new_lines)
