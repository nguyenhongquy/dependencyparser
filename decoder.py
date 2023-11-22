import numpy as np
import time 

NEGINF = -float('inf')
#%%

class Decoder():
    "find maximum span tree using Eisner algorithm"
    def __init__(self, score_matrix):
        # self.sentence = sentence
        self.score = score_matrix
        self.n = score_matrix.shape[0]
        
        self.open_left = np.empty((self.n, self.n))
        self.open_right = np.empty((self.n, self.n))
        self.close_left = np.empty((self.n, self.n))
        self.close_right = np.empty((self.n, self.n))
        
        self.bp_open_left = np.empty((self.n, self.n), dtype=np.int16)
        self.bp_open_right = np.empty((self.n, self.n), dtype=np.int16)
        self.bp_close_left = np.empty((self.n, self.n), dtype=np.int16)
        self.bp_close_right = np.empty((self.n, self.n), dtype=np.int16)

        self.bp_close_left.fill(-1)
        self.bp_close_right.fill(-1)
        
        self.fill_matrices()
        self.arcs = self.build_tree(2, 0, self.n-1, set())

    def fill_matrices(self):
        self.open_left.fill(NEGINF)
        self.open_right.fill(NEGINF)
        self.close_left.fill(NEGINF)
        self.close_right.fill(NEGINF)
        
        for i in range(self.n):
            self.close_left[i][i] = 0.0
            self.close_right[i][i] = 0.0
            
        
        for m in range(1,self.n+1):
            for s in range(0,self.n-m):
                t = s+m
                max_open_left = NEGINF
                max_open_right = NEGINF
                max_close_left = NEGINF
                max_close_right = NEGINF
                
                for q in range(s,t):
                    op_l = self.close_left[s][q] + self.close_right[q+1][t] + self.score[s][t]
                    if op_l > max_open_left:
                        max_open_left = op_l
                        self.bp_open_left[s][t] = q
                self.open_left[s][t] = max_open_left
                
                for q in range(s,t):
                    op_r = self.close_left[s][q] + self.close_right[q+1][t] + self.score[t][s]
                    if op_r > max_open_right:
                        max_open_right = op_r
                        self.bp_open_right[s][t] = q
                self.open_right[s][t] = max_open_right
                
                for q in range(s+1,t+1):
                    cl_l = self.open_left[s][q] + self.close_left[q][t]
                    if cl_l > max_close_left:
                        max_close_left = cl_l 
                        self.bp_close_left[s][t] = q
                self.close_left[s][t] = max_close_left

                for q in range(s,t):
                    cl_r = self.close_right[s][q] + self.open_right[q][t]
                    if cl_r > max_close_right:
                        max_close_right = cl_r
                        self.bp_close_right[s][t] = q
                self.close_right[s][t] = max_close_right
                

        return None
    
    def build_tree(self, typ, s, t, arcs):
        "backtracking using dynamic programming"
        
        if typ == 4: #open left
            q = self.bp_open_left[s][t]
            if q == -1:
                return set()
            else:
                self.build_tree(2, s, q, arcs) 
                self.build_tree(1, q+1, t, arcs)
                arc = (s,t)
            arcs.add(arc)
            
        elif typ==3: #open right
            q = self.bp_open_right[s][t]
            if q == -1:
                return set()
            else:
                self.build_tree(2, s, q, arcs)  
                self.build_tree(1, q+1, t, arcs)
                arc = (t, s)
            arcs.add(arc)
        
        elif typ == 2: #close left table
            q = self.bp_close_left[s][t]
            if q == -1:
                return set()
            else:
                self.build_tree(4, s, q, arcs)
                self.build_tree(2, q, t, arcs)
        
        
        else: # typ == 1: #close right table
            q = self.bp_close_right[s][t]
            if q == -1:
                return set()
            else:
                self.build_tree(1, s, q, arcs) 
                self.build_tree(3, q, t, arcs)
                
        return arcs

#%% some trivial test cases
def test_case_1():
    score = np.array(
        [
            [0, 100, 50],
            [0, 0, 4],
            [0, 11, 0]
        ]

    )
    start = time.time()
    
    for i in range(2):
        decoder = Decoder(score)
        edges = decoder.arcs
    print(edges)
    print(time.time() - start)
    
def test_case_2():
    score = np.array([[-np.inf, 0, 0, 0],
                      [-np.inf, -np.inf, 0, 0],
                      [-np.inf, 0, -np.inf, 0],
                      [-np.inf, 0, 0, -np.inf]
                      ])

    start = time.time()
    
    for i in range(2):
        decoder = Decoder(score)
        edges = decoder.arcs
    print(edges)
    print(time.time() - start)
    
if __name__ == "__main__":
    test_case_1() #should be {(0, 1), (0, 2)}
    test_case_2() #should be {(0, 1), (1, 2), (2, 3)}


