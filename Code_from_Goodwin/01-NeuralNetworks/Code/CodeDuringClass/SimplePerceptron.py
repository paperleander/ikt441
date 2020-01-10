import matplotlib.pyplot as plt
import numpy as np

rocks = [
    [2.7810836, 2.550537003, 0],
    [1.465489372, 2.362125076, 0],
    [3.396561688, 4.400293529, 0],
    [1.38807019, 1.850220317, 0],
    [3.06407232, 3.005305973, 0]
    ]


not_rocks = [
    [7.627531214, 2.759262235, 1],
    [5.332441248, 2.088626775, 1],
    [6.922596716, 1.77106367, 1],
    [8.675418651, -0.242068655, 1],
    [7.673756466, 3.508563011, 1]
    ]


# Plotting

np_rocks = np.array(rocks)
np_not_rocks = np.array(not_rocks)

fig, ax = plt.subplots()
ax.plot(np_rocks[:, 0], np_rocks[:, 1], 'ro')
ax.plot(np_not_rocks[:, 0], np_not_rocks[:, 1], 'go')

ax.grid()
plt.show()

weights = [-0.1, 0.2, -0.2]


def predict(row, weights):
    a = weights[0]*1
    for i in range(len(row)-1):
        a += weights[i]*row[i]
    return 1.0 if a >= 0.0 else 0.0


for n in rocks + not_rocks:
    print(predict(n, weights))
    print(n[-1])
    print("\n")

