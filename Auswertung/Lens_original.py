import math

import numpy as np


class AsphericalLens():

    def __init__(
            self,
            center,
            height,
            length,
            width,
            radius_of_curvature: float,  # Radius of Curvature
            slice_size: float,
            conic_constant_k: float,  # conic constant
            *,
            alpha: float,  # aspherical coefficient
            # circle_object_factory,
            hatch_size: float,
            velocity: float
    ):
        super().__init__()
        # self.circle_object_factory = circle_object_factory
        self.hatch_size = hatch_size
        self.velocity = velocity
        self.k = conic_constant_k
        self.alpha = alpha

        self.height = height
        self.length = length
        self.width = width

        self.center = center
        self.radius_of_curvature = radius_of_curvature
        self.N = int(np.ceil(radius_of_curvature / slice_size))
        self.slice_size = slice_size

    @property
    def center_point(self):
        return self.center

    def sag(self, x: float, y: float) -> float:
        """
        Function to calculate the sag of the aspherical lens
        """
        r = np.sqrt(x ** 2 + y ** 2)
        term1 = r ** 2 / (self.radius_of_curvature * (
                1 + np.sqrt(1 - (1 + self.k) * (r ** 2 / self.radius_of_curvature ** 2))))

        return term1

    def circle_radius(self, z=0) -> float:
        lens_sag = z if z <= self.height else self.height
        if lens_sag == 0.0:
            circle_radius = 0
        else:
            circle_radius = lens_sag * math.sqrt(2 * self.radius_of_curvature / lens_sag - (1 - self.k))
        return circle_radius

    def calc_profile(self, axis: str, pos: float, step_size=None) -> list:
        profile = []
        height_profile = []
        if step_size is None: step_size = self.slice_size
        n_iteration = self.height / step_size
        if not isinstance(n_iteration, int):
            n_iteration = int(np.ceil(n_iteration))

        if axis.lower() == 'x':
            max_line_length = self.length
        elif axis.lower() == 'y':
            max_line_length = self.width
        else:
            raise ValueError('Axis must be either x or y')

        for i in range(n_iteration):
            height = i * step_size
            circle_radius = self.circle_radius(height)
            val = np.sqrt(circle_radius ** 2 - pos ** 2)

            val = min(val, max_line_length / 2)
            height_profile.append(height)
            profile.append([-val, val])

        return profile, height_profile

    def slice_at_height(self, z: float, num_points: int = 100):
        """
        Calculate the x and y coordinates for a given height z.
        """
        # Generate a range of radial distances
        r = np.linspace(0, np.sqrt(self.radius_of_curvature ** 2 - z), num_points)

        # Calculate the x and y values
        x = r
        y = np.zeros_like(r)  # For simplicity, we calculate along the x-axis

        # Adjust x and y values to the correct positions based on the sag equation
        term1 = r ** 2 / (self.radius_of_curvature * (
                1 + np.sqrt(1 - (1 + self.k) * (r ** 2 / self.radius_of_curvature ** 2))))
        term2 = self.alpha * r ** 4
        calculated_z = term1 + term2

        valid_indices = np.where(np.abs(calculated_z - z) < 1e-6)[0]
        return x[valid_indices], y[valid_indices]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    lens = AsphericalLens((0, 0, 0), 1030, 0.05,
                          -2.3, alpha=0, hatch_size=0.1, velocity=40000)
    # Define the range of x and y values
    x = np.arange(-300, 300, 1)
    y = np.arange(-300, 300, 1)
    X, Y = np.meshgrid(x, y)

    # Calculate the corresponding z values
    Z = -lens.sag(X, Y)
    z_0 = -lens.sag(x, 0)

    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax1 = fig.add_subplot(121)
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax1.plot(x, z_0)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
