#Harder condition
 - Need more precise timing control

##Change 1: Harder condition
 - RED_TOTAL_FORCE = 500
 - BLUE_TOTAL_FORCE = 490

 - RED_EFFICIENCY = 0.9
 _ BLUE_EFFICIENCY = 0.9

##Change 2: Stable learning
  get_reward_8A (line 174): cut-off 2-> 0.5 (Maybe more stable)