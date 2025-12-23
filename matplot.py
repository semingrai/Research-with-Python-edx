import numpy as np
import matplotlib.pyplot as plt
x = np.logspace(-1, 1, 40)
y1 = x**2.0
y2 = x**3.0
plt.loglog(x, y1,"rd-", label="First")
plt.loglog(x, y2,"gs-", label="Second")
plt.xlabel("$X-axis$")
plt.ylabel("$Y-axis$")
plt.title("Sample Plot")
plt.axis([0,2.5,-5,30])
plt.legend(loc="upper left")
plt.savefig("tri.pdf")
plt.show()