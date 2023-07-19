import numpy as np

def f_create_moon(n=1000):

    n *= 2
    x = np.random.random((n, 2)) * np.array([np.pi * 1.5, 0.10]) + np.array([np.pi * -0.25, 0.25])

    train_targets = (np.random.random(n) < 0.5)

    train_input = np.concatenate((np.sin(x[:, 0:1]) * x[:, 1:2], np.cos(x[:, 0:1]) * x[:, 1:2]), 1)
    train_input[:, 0] *= train_targets * 2 - 1
    train_input[:, 0] += 0.05 * (train_targets * 2 - 1)
    train_input[:, 1] -= 0.15 * (train_targets * 2 - 1)
    train_input *= 1.2

    data = np.column_stack((train_input * 6, train_targets * 6))

    return data


def f_create_spiral(n=1000):
    

    # Define the parameters for each spiral
    noise = 0.05
    class_1_turns = 1.5
    class_2_turns = 2.5

    # Create an array of angles for each spiral
    theta_1 = np.linspace(0, class_1_turns * 2 * np.pi, n)
    theta_2 = np.linspace(0, class_2_turns * 2 * np.pi, n)

    # Define the radius as a function of the angle for each spiral
    radius_1 = (theta_1 + 3) ** 0.5
    radius_2 = theta_2 ** 0.5

    # Add noise to the radius for each spiral
    radius_1 += np.random.randn(n) * noise
    radius_2 += np.random.randn(n) * noise

    # Calculate the x and y coordinates of each spiral
    x_1 = radius_1 * np.cos(theta_1) 
    y_1 = radius_1 * np.sin(theta_1)
    x_2 = radius_2 * np.cos(theta_2)
    y_2 = radius_2 * np.sin(theta_2)

    # Combine the coordinates and classes into a single array
    X = np.vstack((np.hstack((x_1, x_2)), np.hstack((y_1, y_2)))).T
    y = np.hstack((np.zeros(n), np.ones(n)))

    data = np.column_stack((X, y))

    return data

def f_create_circle(n=1000):
    def f_circle(n=1000, r=1):

        # Generate random angles between 0 and 2*pi
        angles = np.random.uniform(0, 2*np.pi, n)

        # Calculate x and y coordinates of the data points
        x = r * np.cos(angles) + np.random.random(n)
        y = r * np.sin(angles) + np.random.random(n)

        # Combine x and y into a 2D array
        data = np.column_stack((x, y))
        return data
    
    circle1 = f_circle(n=n,r=2)
    circle2 = f_circle(n=n,r=5)

    data = np.concatenate((
        np.concatenate((circle1,np.zeros((n,1))), axis=1),
        np.concatenate((circle2,np.ones((n,1))), axis=1)),
        axis=0)
    
    return data