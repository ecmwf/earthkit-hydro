import os
import re

AUTODOCS_DIR = "source/autodocs"

for root, dirs, files in os.walk(AUTODOCS_DIR):
    for fname in files:
        if not fname.endswith(".rst"):
            continue
        path = os.path.join(root, fname)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        changed = False
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if line == "earthkit.hydro package" and i + 1 < len(lines):
                new_title = "API Reference"
                underline = "=" * len(new_title)
                new_lines.append(f"{new_title}\n")
                new_lines.append(f"{underline}\n")
                i += 2  # skip underline
                changed = True
                continue

            # === Match lines like "earthkit.hydro.data\_structures package" ===
            match = re.match(r"^([\w\.\\]+)\s+(package|module)$", line)
            if match and i + 1 < len(lines):
                title = match.group(1)  # Keep escaped underscores as-is
                underline = "=" * len(title)
                new_lines.append(f"{title}\n")
                new_lines.append(f"{underline}\n")
                i += 2  # skip underline
                changed = True
                continue

            # === Add :titlesonly: under .. toctree:: if missing ===
            if line == ".. toctree::":
                new_lines.append(lines[i])  # original line
                i += 1
                # Look ahead for options
                has_titlesonly = False
                temp_lines = []
                while i < len(lines) and lines[i].lstrip().startswith(":"):
                    if ":titlesonly:" in lines[i]:
                        has_titlesonly = True
                    temp_lines.append(lines[i])
                    i += 1
                if not has_titlesonly:
                    new_lines.append("   :titlesonly:\n")
                    changed = True
                new_lines.extend(temp_lines)
                continue

            # default case
            new_lines.append(lines[i])
            i += 1

        if changed:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
