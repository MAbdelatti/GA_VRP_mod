from math import ceil, floor

class GRID(object):
    def __init__(self):
        self.threads_x     = 32 
        self.threads_y     = 12
        self.min_blocks    = 5

        # Max. no. of blocks in the grid:
        # 35 is good for 10k-core GPUs e.g., RTX 3090
        # 20 is good for 4k-core  GPUs e.g., RTX 2080 Ti and v100
        # 10 is good for 2k-core  GPUs e.g., RTX 3050
        # 5  is good for GTX      GPUs e.g., GTX 1080
        self.max_blocks    = 35

    def __str__(self):
        return 'Grid object has ({}, {}) blocks and ({}, {}) threads per block'.format(self.blocks_x, self.blocks_y, self.threads_x, self.threads_y)

    def blockAlloc(self, n, multiplier):
        tbp         = self.threads_x
        b_min       = self.min_blocks
        b_max       = self.max_blocks

        self.blocks_x = int(min(b_max, max(b_min, floor((2.0*n)/tbp))))
        self.blocks_y = min(b_max, 5*self.blocks_x)
        # self.blocks_y = int(min(30, max(b, floor((n*multiplier)/tbp)))) - self.blocks_x

        return self.blocks_x, self.blocks_y

