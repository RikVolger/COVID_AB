Version history of COVID Agent Based model

v0: model skeleton, set-up of required classes and functions. Works, but uses crude parameters and relations.
v1: model with updated parameters:
	- Period of non-infectiveness: 4 days from infection to being infective

v1.1: Improvements & optimization
	- Updated infectivity, no longer instant infection. 
		Infectivity counted with infectivity factor, increased upon update. 
	- Increase only own infection factor, reading infectivity from friends. This way, each loop only writes to a single person entity. 
	- Friend IDs saved smartly so they're only looked up once. Saves ~50% runtime
	- Population sized up to 100 000

v1.1_parallel_try: side branch looking into parallel processing
	- attempt at implementing parallel processing. Turns out as slow or slower than v1.1

v1.2: Improvements on infectivity
	- Infectivity time-dependent, following supporting GSheet from Hammer & the Dance (https://medium.com/@tomaspueyo/coronavirus-the-hammer-and-the-dance-be9337092b56, datacollection here: https://docs.google.com/spreadsheets/d/1uJHvBubps9Z2Iw_-a_xeEbr3-gci6c475t1_bBVkarc/htmlview?usp=sharing)
	