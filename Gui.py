"""
Genetic Algorithm with Wisdom of Crowds (WoC) for solving an NP-complete problem.
This program implements a hybrid GA+WoC approach to optimize a specified route,
aiming to solve the Traveling Salesman Problem.
"""

import random
import networkx as nx
from collections import Counter
import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(level=logging.ERROR)

def parse_tsp_file(filename):
    """
        Parses a .tsp file to extract city coordinates.

        Args:
            filename (str): Path to the .tsp file formatted with city coordinates.

        Returns:
            list of tuple: A list of (x, y) coordinates for each city.

        Expected Format:
            The file should contain a "NODE_COORD_SECTION" section with lines
            formatted as "index x y". The function ignores any lines outside of this section.
        """
    cities = []
    with open(filename, 'r') as file:
        is_node_section = False
        for line in file:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                is_node_section = True
                continue
            elif line == "EOF":
                break
            if is_node_section:
                parts = line.split()
                if len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    cities.append((x, y))
    return cities

def select_experts(population, fitnesses, num_experts):
    """
        Selects the top-performing individuals in the population based on fitness.

        Args:
            population (list): List of routes (individuals) in the current population.
            fitnesses (list): List of fitness scores corresponding to each individual.
            num_experts (int): Number of top individuals to select.

        Returns:
            list: A list of the top 'num_experts' routes based on fitness.
        """
    sorted_individuals = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    experts = [individual for individual, _ in sorted_individuals[:num_experts]]
    return experts

def aggregate_expert_routes(experts, num_cities):
    """
       Forms a consensus route by aggregating the paths from expert solutions.

       Args:
           experts (list): List of expert routes chosen from the population.
           num_cities (int): Total number of cities.

       Returns:
           list: A consensus route that combines common paths among experts.

       Purpose:
           This function leverages the Wisdom of Crowds (WoC) approach by using
           top routes to form a single path that represents collective intelligence.
       """
    if not experts:
        return []

    consensus_route = []
    remaining_cities = set(range(num_cities))
    start_city = Counter(route[0] for route in experts).most_common(1)[0][0]
    consensus_route.append(start_city)
    remaining_cities.remove(start_city)

    while remaining_cities:
        current_city = consensus_route[-1]
        next_city_counts = Counter(
            route[(route.index(current_city) + 1) % len(route)]
            for route in experts if current_city in route
        )

        # Choose the next city based on frequency among experts
        next_city = next((city for city, _ in next_city_counts.most_common() if city in remaining_cities), None)

        if next_city is None:
            next_city = remaining_cities.pop()  # Fallback if no common city is found
        else:
            remaining_cities.remove(next_city)

        consensus_route.append(next_city)

    return consensus_route

def smart_swap_mutation(individual, mutation_rate):
    """
    Applies swap mutation to an individual's route with a given probability.

    Args:
        individual (list): The route to mutate.
        mutation_rate (float): The probability of performing a mutation.

    Returns:
        list: The mutated route.

    Mutation Process:
        Randomly selects two positions in the route and swaps them,
        introducing variation to avoid local optima.
    """
    if random.random() < mutation_rate:
        swap_indexes = random.sample(range(len(individual)), 2)
        individual[swap_indexes[0]], individual[swap_indexes[1]] = (
            individual[swap_indexes[1]], individual[swap_indexes[0]]
        )
    return individual

def precompute_distances(cities):
    """
        Precomputes pairwise distances between cities.

        Args:
            cities (list): List of (x, y) tuples representing city coordinates.

        Returns:
            dict: A dictionary with keys as city index pairs (i, j) and values as distances.
        """
    num_cities = len(cities)
    distances = {}
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            dist = np.sqrt((cities[i][0] - cities[j][0]) ** 2 + (cities[i][1] - cities[j][1]) ** 2)
            distances[(i, j)] = dist
            distances[(j, i)] = dist
    return distances


def calculate_coverage(edges, population):
    used_edges = set()
    for route in population:
        for u, v in edges:
            if u in route and v in route:
                used_edges.add((u, v))
    return len(used_edges) / len(edges)


def heuristic_initialize_population(cities, distances, population_size):
    """
        Initializes the population with routes based on a Minimum Spanning Tree (MST) heuristic.

        Args:
            cities (list): List of city coordinates.
            distances (dict): Precomputed pairwise distances.
            population_size (int): Desired population size.

        Returns:
            list: A list of initial routes (each route is a list of city indices).

        Heuristic:
            Uses MST to produce more optimal initial routes, improving convergence speed.
        """
    graph = nx.Graph()
    for (u, v), dist in distances.items():
        graph.add_edge(u, v, weight=dist)
    mst = nx.minimum_spanning_tree(graph)

    population = []
    for _ in range(population_size // 2):
        route = list(mst.nodes)
        random.shuffle(route)
        population.append(route)
    for _ in range(population_size - len(population)):
        population.append(random.sample(range(len(cities)), len(cities)))
    return population


def heuristic_edge_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    used_edges = set()
    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent1[start:end]
    used_edges.update([(parent1[i], parent1[i + 1]) for i in range(start, end - 1)])
    remaining_cities = set(range(size)) - set(child)
    for i in range(size):
        if child[i] == -1:
            for j in range(size):
                u, v = parent2[j], parent2[(j + 1) % size]
                if u in remaining_cities:
                    child[i] = u
                    remaining_cities.remove(u)
                    break
    return child

def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Selects one individual from the population using tournament selection.
    Parameters:
        population (list): The population of routes.
        fitnesses (list): The corresponding fitness scores for each individual in the population.
        tournament_size (int): The number of individuals to include in each tournament.
    Returns:
        selected_individual: The individual selected as the winner of the tournament.
    """
    # Randomly select 'tournament_size' individuals from the population
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    # Sort selected individuals by fitness in descending order and pick the best one
    selected = sorted(selected, key=lambda x: x[1], reverse=True)
    # Return the individual with the highest fitness in the tournament
    return selected[0][0]


def calculate_fitness(route, distances):
    total_distance = sum(distances[(route[i], route[(i + 1) % len(route)])] for i in range(len(route)))
    return 1 / total_distance  # Higher fitness for shorter distances


def genetic_algorithm_cpp(cities, population_size=50, generations=100,
                          mutation_rate=0.1, expert_percentage=0.1):
    """
        Runs the Genetic Algorithm (GA) with Wisdom of Crowds (WoC) for optimizing a route.

        Args:
            cities (list): List of city coordinates.
            population_size (int): Number of individuals in the population.
            generations (int): Number of generations to run the algorithm.
            mutation_rate (float): Probability of mutation occurring.
            expert_percentage (float): Percentage of population selected as experts.

        Returns:
            tuple: Best route found, its fitness, and a list of fitness values over generations.

        Process:
            1. Initializes population with a heuristic-based approach.
            2. Evaluates fitness of routes, selecting top experts.
            3. Aggregates routes from experts to guide convergence.
            4. Applies mutation adaptively based on progress to avoid stagnation.
        """

    distances = precompute_distances(cities)
    population = heuristic_initialize_population(cities, distances, population_size)
    best_solution = None
    best_fitness = -1
    fitness_progress = []

    for gen in tqdm(range(generations)):
        # Calculate fitness based on total route length
        fitnesses = [calculate_fitness(route, distances) for route in population]

        # Track the best solution in each generation
        gen_best_fitness = max(fitnesses)
        gen_best_solution = population[fitnesses.index(gen_best_fitness)]

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_solution = gen_best_solution

        fitness_progress.append(best_fitness)

        # Adjust mutation rate if no improvement (adaptive mutation)
        if gen > 1 and fitness_progress[-1] == fitness_progress[-2]:
            mutation_rate = min(mutation_rate * 1.1, 0.3)  # Gradually increase up to a limit
        else:
            mutation_rate = max(mutation_rate * 0.9, 0.05)  # Gradually decrease

        # Select experts and aggregate their routes
        experts = select_experts(population, fitnesses, int(expert_percentage * population_size))
        consensus_route = aggregate_expert_routes(experts, len(cities))

        # Create new population with consensus route as elite every 5 generations
        new_population = [consensus_route] if gen % 5 == 0 else []
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            child1 = heuristic_edge_crossover(parent1, parent2)
            child2 = heuristic_edge_crossover(parent2, parent1)

            # Apply mutation with updated mutation rate
            if random.random() < mutation_rate:
                child1 = smart_swap_mutation(child1, mutation_rate)
            if random.random() < mutation_rate:
                child2 = smart_swap_mutation(child2, mutation_rate)

            new_population.extend([child1, child2])

        population = new_population[:population_size]
    return best_solution, best_fitness, fitness_progress

def plot_route(cities, route):
    """
        Plots the given route on a 2D plane for visualization.

        Args:
            cities (list): List of (x, y) coordinates representing cities.
            route (list): List of city indices representing the route order.

        Visualization:
            Draws the route as a connected path, with each city represented as a point.
        """
    ordered_cities = [cities[i] for i in route]
    ordered_cities.append(ordered_cities[0])  # return to the first city
    x = [city[0] for city in ordered_cities]
    y = [city[1] for city in ordered_cities]
    plt.plot(x, y, 'o-')
    plt.title('Best TSP Route')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()
# ... importing and defining functions ...

if __name__ == "__main__":
    tsp_file = 'C:/Users/josef/PycharmProjects/FP-CSE545/Random100.tsp'
    cities = parse_tsp_file(tsp_file)
    edges = [(i, j) for i in range(len(cities)) for j in range(i + 1, len(cities))]

    # Get user inputs for algorithm parameters
    population_size = int(input("Enter population size (default is 50)(from 1-400): ") or 50)
    generations = int(input("Enter number of generations (default is 100)(from 1 - 1000): ") or 100)
    mutation_rate = float(input("Enter mutation rate (default is 0.1%)(from 0.1% - 25%): ") or 0.1)

    best_route, best_coverage, progress = genetic_algorithm_cpp(cities,
                                                                population_size=population_size,
                                                                generations=generations,
                                                                mutation_rate=mutation_rate)

    # Plot fitness progress over generations
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(progress) + 1), progress, marker='o', color='b')
    plt.ylabel("Distance")
    plt.xlabel("Generation")
    plt.title("Fitness Progress over Generations")
    plt.grid(True)
    plt.show()
    # Plot the route of the best solution
    plot_route(cities, best_route)