import random
import matplotlib.pyplot as plt
import numpy as np
import time
start_time = time.time()
rolls = []
for i in range(100):
    rolls.append(random.choice(range(7)))
plt.hist(rolls, bins = np.linspace(0.5,6.5,7));
plt.show()
end_time = time.time()
print("Time taken:", end_time - start_time, "seconds")