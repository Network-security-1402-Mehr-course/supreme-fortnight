import numpy as np
from icecream import ic

import s_box
from s_box_analysis import differential, linear

ic(differential(np.array(s_box.baby_rijndael)))
ic(linear(np.array(s_box.baby_rijndael)))

ic(differential(np.array(s_box.aes)))
ic(linear(np.array(s_box.aes, dtype=np.uint8)))
