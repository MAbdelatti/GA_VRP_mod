from math import ceil, floor

class GRID(object):
    def __init__(self):
        self.threads_x     = 32 
        self.threads_y     = 16
        self.arrangement   = 1

        self.block_dict    = {1:(4, 4), 2:(8, 4), 3:(8, 8), 4:(16,8), 5:(16, 16), 6:(32, 16)}

    def __str__(self):
        return 'Grid object has {} blocks and ({}, {}) threads per block'.format(self.block_dict[self.arrangement], self.threads_x, self.threads_y)

    def blockAlloc(self, n, multiplier):
        self.arrangement = 5 

        return self.block_dict[self.arrangement][0], self.block_dict[self.arrangement][1]

if __name__ == '__main__':
    grid = GRID()
    print(grid.blockAlloc(4,4))

