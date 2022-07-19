# Rules for the tile map
# Put each rule in an array like so: [[None, 0   , None],
#                                     [0   , "R" , 1   ],
#                                     [None, 1   , None]]

# S means dont rotate or flip R means rotate X means flip on X-axis and y means flip on Y-axis
# You can combine letters expect S because that will just overide everything

# 1 means something should be there None means don't care and 0 means something shouldn't be there

# Split each rule for each tilemap into different arrays like done down bellow

# If there are not enough rules for all the tilemaps, that tilemap will just be randomized

rules = [
         # RULE MAP 1
        [[[None, 0   , None],
          [0   , "R" , 1   ],
          [None, 1   , None]],
         
         [[None, 0   , None],
          [0   , "S" , 0   ],
          [None, 0   , None]],
         
         [[None, 0   , None],
          [0   ,"R" ,       0],
          [None, 1   , None]],
         
         [[None, 1   , None],
          [None, "R" , None],
          [None, 1   , None]],
        ],

        # RULE MAP 2
        [[[None, 0   , None],
          [0   , "R" , 1   ],
          [None, 1   , 1   ]],
         
         [[None, 0   , None],
          [0   , "S", 0   ],
          [None, 0   , None]],
         
         [[None, 0   , None],
          [0   ,"R" , 0   ],
          [None, 1   , None]],
         
         [[None,1    , None],
          [0   ,"R"  , 0   ],
          [None,1    , None]],
         
         [[None, 1   , 1   ],
          [0   , "R" , 1   ],
          [None, 1   , 1   ]],
         
         [[1   , 1   , 1   ],
          [1   , "S" , 1   ],
          [1   , 1   , 1   ]],
         
         [[None, 0   , None],
          [0   , "R" , 1   ],
          [None, 1   , 0   ]],
         
         [[None, 1   , 0   ],
          [0   , "R" , 1   ],
          [None, 1   , 0   ]],
         
         [[None, 1   , 1   ],
          [0   , "RY", 1   ],
          [None, 1   , 0   ]],
         
         [[0   , 1   , 0   ],
          [1   , "S" , 1   ],
          [0   , 1   , 0   ]],
         
         [[0   , 1   , 0   ],
          [1   , "R" , 1   ],
          [1   , 1   , 0   ]],
         
         [[1   , 1   , 0   ],
          [1   , "R" , 1   ],
          [1   , 1   , 0   ]],
         
         [[0   , 1   , 1   ],
          [1   , "R" , 1   ],
          [1   , 1   , 0   ]],
         
         [[1   , 1   , 1   ],
          [1   , "R" , 1   ],
          [1   , 1   , 0   ]],
        ]]
