import numpy as np

# Ant Colony Optimization (ACO) implementation for the Traveling Salesman Problem (TSP)
# - algorithm logic referenced from ant colony optimization series of videos from https://www.youtube.com/@hklam2368 

class Ant:
    def __init__(self, num_cities, alpha, beta, distance_matrix):
        self.num_cities = num_cities
        self.visited = [False] * num_cities
        self.alpha = alpha
        self.beta = beta

        # 2D numpy array that represents the distance between each pair of cities
        self.distance_matrix = distance_matrix
        # represents the sequence of cities visited by the ant during its tour
        self.tour = []

    # method for selecting the next city to visit based on the pheromone trails
    def select_next_city(self, current_city, pheromone_matrix):
        unvisited_probabilities = []
        total_prob = 0
        # calculate probabilities for cities that are not yet visited by this ant instance
        for city in range(self.num_cities):
            if not self.visited[city]:
                # check the pheromone trail and distance from current city
                pheromone = pheromone_matrix[current_city][city]
                distance = self.distance_matrix[current_city][city]

                # calculate probability
                probability = pheromone ** self.alpha * (1.0 / distance) ** self.beta
                unvisited_probabilities.append((city, probability))
                total_prob += probability
        
        # select next city based on probability
        # generate number between 0 and total probability
        selected = np.random.rand() * total_prob
        
        # randomly select next city through a probabilistic selection process
        current_prob = 0
        for city, probability in unvisited_probabilities:
            current_prob += probability
            if current_prob >= selected:
                return city

    # method only for changing current city of ant and updating concerned variables
    def travel(self, start_city, pheromone_matrix):
        current_city = start_city
        self.visited[current_city] = True
        self.tour.append(current_city)
        # move to next city until all cities are visited
        for _ in range(self.num_cities - 1):
            next_city = self.select_next_city(current_city, pheromone_matrix)
            current_city = next_city
            self.visited[current_city] = True
            self.tour.append(current_city)

class ACO:
    def __init__(self, num_ants, num_iterations, evaporation_rate, alpha, beta, Q, distance_matrix):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.distance_matrix = distance_matrix

        # higher evaporation rate means pheromone levels decrease faster, influencing the exploration-exploitation balance
        self.evaporation_rate = evaporation_rate
        #controls the influence of pheromone on the ant's decision-making process. a higher alpha value gives more weight to pheromone levels
        self.alpha = alpha
        # controls the influence of distance between cities on the ant's decision-making process
        self.beta = beta
        # pheromone deposit constant, determines the amount of pheromone deposited on paths by ants
        self.Q = Q
        # initial pheromone level, calculated based on the number of cities and average distance between them
        # only used once for creation of self.tau
        self.tau0 = 1.0 / (num_cities * np.average(distance_matrix))
        # 2D numpy array that represents the pheromone trails which stores the pheromone levels on each path between cities and updated by ants during travel and by pheromone evaporation
        self.tau = np.ones((num_cities, num_cities)) * self.tau0
        # keeps track of the best tour found so far by the algorithm
        self.best_tour = None
        # stores the length of the best tour found so far, used for comparison with new solutions
        self.best_distance = float('inf')

    def initialize_ants(self):
        # create instances of ant objects based on num_ants (total number of ants for the problem)
        self.ants = [Ant(self.distance_matrix.shape[0], self.alpha, self.beta, self.distance_matrix) for _ in range(self.num_ants)]

    def construct_solutions(self):
        # construct solutions for each ant
        for ant in self.ants:
            # starting from city #1
            start_city = 0 
            ant.travel(start_city, self.tau)

    def update_pheromones(self):
        # update pheromone trails based on ant solutions
        for ant in self.ants:
            total_distance = 0
            for i in range(ant.num_cities):
                # retrieve starting city of edge
                start = ant.tour[i]

                # in the travelling salesman problem, each city should be visited exactly once and return to the starting city
                # hence, we should have a cyclic graph and the ants should return to city #1 at the end of the tour

                # retrieve ending city of the edge by using 'i+1'
                # when i is the last city needed to be visited, i+1 will be equal to the total # of cities (0-based indexing is used here so this kind of i+1 will be invalid)
                    # '% ant.num_cities' ensures that the index stays within the range of valid indices by
                    # wrapping 'i+1' around to the beginning of the list using the remainder operator %
                    # this ensures that the next city to be visited after the final city is actually the first city in the tour
                end = ant.tour[(i + 1) % ant.num_cities]

                # accumulate the total distance traveled by an ant during its tour using the distance matrix provided
                total_distance += self.distance_matrix[start][end]
            
            # calculate the deposit amount of pheromone to be added to the paths taken by the ant
                # Q represents the total amount of pheremones to be deposited
                # dividing Q by total_distance ensures that shorter tours result in a higher density of pheromone on the edges they traverse
            deposit_amount = self.Q / total_distance

            # deposit the pheromone on the path taken by each ant
            for i in range(ant.num_cities):
                start = ant.tour[i]
                end = ant.tour[(i + 1) % ant.num_cities]

                # update the pheromone level for each edge
                # we are using a 2d array here so matrix[start][end] and matrix[end][start] needs to be updated to maintain symmetry
                
                # '(1 - self.evaporation_rate) * self.tau[start][end]' represents the evaporation of existing pheromones on the trails
                # add the deposit amount to the resulting product of the above calculation
                # this formula achieves a balance between exploration and exploitation in the search process
                self.tau[start][end] = (1 - self.evaporation_rate) * self.tau[start][end] + deposit_amount
                self.tau[end][start] = (1 - self.evaporation_rate) * self.tau[end][start] + deposit_amount

    # method used for the decay of pheromone trails over time
    # this allows for maintaining diversity and avoiding stagnation, allowing ants to explore new paths
    def evaporate_pheromones(self):
        # evaporate pheromones from all trails
        self.tau *= (1 - self.evaporation_rate)

    def optimize(self):
        # main optimization loop
        for iteration in range(self.num_iterations):
            self.initialize_ants()
            self.construct_solutions()
            self.update_pheromones()
            self.evaporate_pheromones()
            # update best solution found so far
            for ant in self.ants:
                # iterate over each city using list comprehension and add the distance between current city and next city to tour_distance accumulator
                tour_distance = sum(self.distance_matrix[ant.tour[i]][ant.tour[(i + 1) % ant.num_cities]] for i in range(ant.num_cities))
                
                # if this tour has shorter distance/beats the best tour path found so far, replace the best tour path
                if tour_distance < self.best_distance:
                    self.best_distance = tour_distance
                    self.best_tour = ant.tour
            
            # print the current best distance
            print(f"Iteration {iteration + 1}: Best Distance = {self.best_distance}")



if __name__ == "__main__":
    # default values
    num_ants = 10
    num_iterations = 100
    evaporation_rate = 0.1
    alpha = 1
    beta = 2
    Q = 1

    try:
        print("\nANT COLONY OPTIMIZATION FOR TRAVELING SALESMAN PROBLEM\n")
        print("\nUSER INPUTS [press ENTER/empty values to use default values in the program]\n")

        # read distance matrix from given .txt file
        dataset_filename = input("Enter the filename of the dataset (with file extension) [default 'dataset.txt']: ")

        # if user entered empty string for dataset file
        if not dataset_filename:
            dataset_filename = "dataset.txt"

        with open(dataset_filename, "r") as file:
            # the dataset's first line should tell the total number of cities
            num_cities = int(file.readline().strip())

            # each ith line after the first line contains the distance of the ith city to all other cities, delimited by whitespace
            distance_matrix = np.zeros((num_cities, num_cities))
            for i in range(num_cities):
                distances = list(map(float, file.readline().strip().split()))
                distance_matrix[i] = distances

        # distance_matrix will serve as heuristic information to guide ants on decision-making
        print("\nDistance matrix:\n")
        print(distance_matrix)

        # number of ants
        num_ants_input = input("\nEnter the number of ants (default 10): ")
        num_ants = num_ants if num_ants_input == "" else int(num_ants_input)

        # number of iterations            
        num_iterations_input = input("Enter the number of iterations (default 100): ")
        num_iterations = num_iterations if num_iterations_input == "" else int(num_iterations_input)

        # evaporation rate
        evaporation_rate_input = input("Enter the evaporation rate (default 0.1): ")
        evaporation_rate = evaporation_rate if evaporation_rate_input == "" else float(evaporation_rate_input)

        #alpha value
        alpha_input = input("Enter the alpha value (default 1): ")
        alpha = alpha if alpha_input == "" else float(alpha_input)
        
        # beta value
        beta_input = input("Enter the beta value (default 2): ")
        beta = beta if beta_input == "" else float(beta_input)
        
        # Q value
        # a constant value that determines the total amount of pheromone deposited by each ant during its tour
        Q_input = input("Enter the Q value (default 1): ")
        Q = Q if Q_input == "" else float(Q_input)

        # ensure valid values for parameters
        if num_ants <= 0 or num_iterations <= 0 or evaporation_rate <= 0 or alpha <= 0 or beta <= 0 or Q <= 0:
            raise ValueError("All parameters must be positive")
        print()

        # commence the algorithm
        aco = ACO(num_ants, num_iterations, evaporation_rate, alpha, beta, Q, distance_matrix)
        aco.optimize()
        print("\n\nBest Tour:", aco.best_tour)
        print("\nBest Distance:", aco.best_distance, "\n")

    except ValueError as e:
        print("Error: ", e)