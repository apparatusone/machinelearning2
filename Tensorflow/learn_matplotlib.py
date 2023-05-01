import matplotlib.pyplot as plt
import numpy as np


# create a data set
x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]

# change the graph size
fig, ax = plt.subplots(figsize=(9, 7))

# Add grid lines
ax.grid(color='lightgrey', linestyle='--', linewidth=0.5)

# plot x axis, y axis, and 'ro' represent color=r and o=circle
ax.plot(x, y, 'b+', markersize=6)

# set axis limit for x and y
ax.axis([0, 6, 0, 20])

# show the graph
plt.show()