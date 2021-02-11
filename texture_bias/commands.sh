#!/bin/bash

# normal texture bias
python3 main.py 

# background interpolation
python3 main.py --bg_interp 1
python3 main.py --bg_interp 0.75
python3 main.py --bg_interp 0.5
python3 main.py --bg_interp 0.25
python3 main.py --bg_interp 0

# resizing
python3 main.py --bg_interp 0 --size 1
python3 main.py --bg_interp 0 --size 0.75
python3 main.py --bg_interp 0 --size 0.5
python3 main.py --bg_interp 0 --size 0.25

# landscape background
python3 main.py --landscape

# only complete shapes
python3 main.py --only_complete
python3 main.py --only_complete --bg_interp 0