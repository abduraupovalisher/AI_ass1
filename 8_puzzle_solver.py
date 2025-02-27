import matplotlib.pyplot as plt
import numpy as np
from queue import PriorityQueue

class TreeNode:
    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state  
        self.parent = parent  
        self.action = action  
        self.cost = cost  
        self.heuristic = self.calculate_heuristic()  

    def calculate_heuristic(self):
        heuristic = 0
        goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]  
        for i in range(3):
            for j in range(3):
                if self.state[i][j] != 0:
                    x, y = divmod(self.state[i][j] - 1, 3)
                    heuristic += abs(x - i) + abs(y - j)
        return heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def get_blank_position(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return (i, j)

def get_neighbors(node):
    neighbors = []
    i, j = get_blank_position(node.state)
    actions = {
        "yuqori": (i-1, j),
        "past": (i+1, j),
        "chap": (i, j-1),
        "o'ng": (i, j+1)
    }
    for action, (x, y) in actions.items():
        if 0 <= x < 3 and 0 <= y < 3:
            new_state = [row[:] for row in node.state]
            new_state[i][j], new_state[x][y] = new_state[x][y], new_state[i][j]
            neighbors.append(TreeNode(new_state, node, action, node.cost + 1))
    return neighbors

def is_solvable(state):
    inversions = 0
    flat_state = [num for row in state for num in row if num != 0]
    for i in range(len(flat_state)):
        for j in range(i + 1, len(flat_state)):
            if flat_state[i] > flat_state[j]:
                inversions += 1
    return inversions % 2 == 0

def a_star_search(initial_state):
    if not is_solvable(initial_state):
        return "Not solvable"

    start_node = TreeNode(initial_state)
    frontier = PriorityQueue()
    frontier.put(start_node)
    explored = set()

    while not frontier.empty():
        current_node = frontier.get()
        if current_node.state == [[1, 2, 3], [4, 5, 6], [7, 8, 0]]:
            return get_solution_path(current_node)

        explored.add(tuple(map(tuple, current_node.state)))

        for neighbor in get_neighbors(current_node):
            if tuple(map(tuple, neighbor.state)) not in explored:
                frontier.put(neighbor)

    return "No solution found"

def get_solution_path(node):
    # Yechim yo'lini topish
    path = []
    while node:
        path.append((node.action, node.state))
        node = node.parent
    return list(reversed(path))

def visualize_puzzle(state, title="8-Puzzle"):
    # 8-puzzle doskasini chizish
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(0, 3, 1))
    ax.set_yticks(np.arange(0, 3, 1))
    ax.grid(which="both")
    ax.imshow(np.zeros((3, 3)), cmap="gray")  # Bo'sh doska

    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:
                ax.text(j, i, str(state[i][j]), fontsize=20, ha="center", va="center")
    ax.set_title(title)
    plt.show()

initial_state = [
    [1, 3, 0],
    [4, 2, 5],
    [7, 8, 6]
]

unsolvable_state = [
    [1, 2, 3],
    [4, 5, 6],
    [8, 7, 0]
]

solution = a_star_search(initial_state)
if solution == "Not solvable":
    print("Not solvable")
else:
    for step in solution:
        print(f"Action: {step[0]}")
        visualize_puzzle(step[1], title=f"Action: {step[0]}")
        print()

solution = a_star_search(unsolvable_state)
print(solution)  
