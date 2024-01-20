from markovchain import MarkovChain as drwMarkovChain
import matplotlib.pyplot as plt
import numpy as np

# module from this repository
class MarkovChain():
    def __init__(self, MC, names) -> None:
        self.MC = np.array(MC)
        self.names = names

    def get_graph(self):
        mc = drwMarkovChain(self.MC, self.names)
        fig=mc.draw()


    def get_transition_matrix(self):
        return self.MC
    
    def get_transition_matrix_n_steps(self, n):
        return np.linalg.matrix_power(self.MC, n)

    def check_reducibility(self):
        # Find Pt-I and check if it is irreducible
        A = np.transpose(self.MC) - np.eye(self.MC.shape[0])
        print(A)
        print(np.linalg.matrix_rank(A))
        if np.linalg.matrix_rank(A) == self.MC.shape[0]:
            print("Irreducible")
        else:
            print("Reducible")




    def get_estimated_first_passage_times(self):
        U =[]
        for i in range(self.MC.shape[0]):
            # set column i to 0
            A= self.MC.copy()
            A[:,i] = 0
            # I-A
            A = np.eye(self.MC.shape[0]) - A
            # set b to 1
            b = np.ones(self.MC.shape[0])
            # solve the system of linear equations
            x = np.linalg.solve(A, b)
            # x = np.linalg.inv(A) * b
            U.append(x)
        U = np.transpose(np.array(U))


        return U

    def get_probability_first_time_passage_n_steps(self, n, i, j):
    #    f_{i j}^{(1)}=p_{i j}^{(1)}
    # f_{i j}^{(n)}=p_{i j}^{(n)}-\sum_{k=1}^{n-1} f_{i j}^{(k)} \cdot p_{i j}^{(n-k)}

        f=[]
        for n1 in range(1,n+1):
            if n1 == 1:
                f.append(self.MC[i,j])
            else:

                f.append(self.get_transition_matrix_n_steps(n1)[i,j] - np.sum([f[k-1]*self.get_transition_matrix_n_steps(n1-k)[i,j] for k in range(1,n1)]))
        print(f)
        return f[-1]

    def get_steady_state(self): 
        # Let P be the transition matrix of the Markov chain.

        # Lets define the matrix A = P^T - I
        A =np.transpose(self.MC) - np.eye(self.MC.shape[0])

        # Add a row of ones to A at the bottom
        A = np.vstack((A, np.ones(self.MC.shape[0])))

        # define b = [0, 0, ..., 0, 1]^T
        b = np.zeros(self.MC.shape[0])
        b = np.append(b, 1)
        # Solve the system of linear equations Ax = b, remember A is not square
        x = np.linalg.lstsq(A, b, rcond=None)[0]

        return x


    def get_probability_first_time_passage_n_simulations_through_simulation(self,  i, j, n_simulations):
        # Simulate the Markov Chain
        steps_to_j = []
        for _ in range(n_simulations):
            # Start in state i
            state = i
            # Count the number of steps
            steps = 0
            # Repeat until we reach state j
            while state != j or steps == 0:
                # Take a step
                state = np.random.choice(range(self.MC.shape[0]), p=self.MC[state])
                # Increase the number of steps
                steps += 1
            # Add the number of steps to the list
            steps_to_j.append(steps)



        
        
        return np.mean(steps_to_j),steps_to_j

    def draw_probability_distribution_first_time_n_simulation(self,i,j,n):
        mean,list = self.get_probability_first_time_passage_n_simulations_through_simulation(i,j,n)

        x= np.arange(1, len(list)+1)
        y = [np.mean(list[:i]) for i in x]

        # Find the probability of going from state i to state j in n steps

        real=self.get_estimated_first_passage_times()[i,j]


        plt.plot(x,y)
        plt.plot(x,[real]*len(x))
        plt.show()




        

    
    
    

