import numpy as np
from icecream import ic

import s_box
from s_box_analysis import algebraic, differential, linear

ic(differential(np.array(s_box.baby_rijndael)))
ic(linear(np.array(s_box.baby_rijndael)))
ic(algebraic(np.array(s_box.baby_rijndael)))

ic(differential(np.array(s_box.aes)))
ic(linear(np.array(s_box.aes, dtype=np.uint8)))
# this line tries to calculate null space of a 256x137 matrix
# ic(algebraic(np.array(s_box.aes, dtype=np.uint8)))
