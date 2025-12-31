from sklearn import datasets
import scipy.stats as ss
import random
import numpy as np
import matplotlib.pyplot as plt

def distance(p1,p2):
    '''Finds the distance between points p1 and p2.'''
    return np.sqrt(np.sum(np.power(p2-p1,2)))

points = np.array([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]])
p = np.array([2.5,2])
plt.plot(points[:,0],points[:,1],"ro")
plt.plot(p[0],p[1],"bo")
plt.axis([0.5, 3.5, 0.5, 3.5])

def majority_vote(votes):
    """xxx""" 
    vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1
    winners =[]
    max_counts = max(vote_counts.values())
    for vote, count in vote_counts.items():
        if count == max_counts:
            winners.append(vote)
    return random.choice(winners)

def find_nearest_neighbors(p, points, k=5):
    '''Find the k nearest neighbors of point p and return their indices'''
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]

def knn_predict(p, points, outcomes, k = 5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote(outcomes[ind])

outcomes = np.array([0,0,0,0,1,1,1,1,1])

def make_prediction_grid(predictors, outcomes, limits, h, k):
    '''Classify each points on the prediction grid.'''
    x_min, x_max, y_min, y_max =  limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)

    prediction_grid = np.zeros(xx.shape, dtype = int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = knn_predict(p, predictors, outcomes, k)
    return(xx,yy, prediction_grid)

def generate_synth_data(n=50):
    '''Create 2 sets of points from bivariate normal distributions.'''
    points = np.concatenate((ss.norm(0,1).rvs((n,2)),ss.norm(1,1).rvs((n,2))), axis =0)
    outcomes = np.concatenate((np.repeat(0,n), np.repeat(1,n)))
    return (points, outcomes)

def plot_prediction_grid(xx, yy, prediction_grid, filename, predictors, outcomes, k):
    '''Plot the prediction grid and save to file.'''
    from matplotlib.colors import ListedColormap
    
    background_colormap = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, prediction_grid, cmap=background_colormap, alpha=0.5)
    
    # Overlay actual data points
    plt.plot(predictors[outcomes == 0][:,0], predictors[outcomes == 0][:,1], "ro", markersize=8, label='Class 0')
    plt.plot(predictors[outcomes == 1][:,0], predictors[outcomes == 1][:,1], "bo", markersize=8, label='Class 1')
    if len(np.unique(outcomes)) > 2:
        plt.plot(predictors[outcomes == 2][:,0], predictors[outcomes == 2][:,1], "go", markersize=8, label='Class 2')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'K-NN Decision Boundary (k={k})')
    plt.legend()
    plt.savefig(filename)
    plt.show()

# Iris data (THIS IS WHAT YOU WANT TO MATCH THE VIDEO)
iris = datasets.load_iris()
predictors = iris.data[:, 0:2]
outcomes = iris.target

k = 5
filename = "iris.pdf"
limits = (4, 8, 1.5, 4.5)
h = 0.1

# Generate and plot the prediction grid for iris data
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename, predictors, outcomes, k)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= 5)
knn.fit(predictors, outcomes)
sk_predictions = knn.predict(predictors)

my_predictions = np.array([knn_predict(p, predictors, outcomes, k ) for p in predictors ])