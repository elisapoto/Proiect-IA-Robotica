import matplotlib.pyplot as plt

def plot_path(cities, path, title="TSP Solution"):
    plt.figure()

    # traseu
    for i in range(len(path)):
        start = cities[path[i]]
        end = cities[path[(i + 1) % len(path)]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'b-')

    # puncte orașe
    for i, city in enumerate(cities):
        plt.scatter(city[0], city[1])
        plt.text(city[0], city[1], str(i))

    plt.title(title)
    plt.show()