import numpy as np
import matplotlib.pyplot as plt
from numpy import trapezoid
from typing import Tuple


class MobiusStrip:
    def __init__(self, R: float = 1.0, w: float = 0.3, n: int = 100):
        """
        Initialize Möbius strip parameters.
        :param R: Radius from center to strip midline.
        :param w: Width of the strip.
        :param n: Number of points for mesh resolution.
        """
        self.R = R
        self.w = w
        self.n = n

        # Create 1D arrays for parameters u (along length) and v (across width)
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)

        # Create 2D parameter grid for mesh
        self.U, self.V = np.meshgrid(self.u, self.v)

        # Compute 3D coordinates of the strip on the mesh grid
        self.X, self.Y, self.Z = self._compute_coordinates()

    def _compute_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the (X, Y, Z) coordinates of the Möbius strip surface.
        """
        U, V = self.U, self.V
        X = (self.R + V * np.cos(U / 2)) * np.cos(U)
        Y = (self.R + V * np.cos(U / 2)) * np.sin(U)
        Z = V * np.sin(U / 2)
        return X, Y, Z

    def compute_surface_area(self) -> float:
        """
        Approximate surface area using numerical integration.
        """
        dx_du = np.gradient(self.X, self.u, axis=1)
        dx_dv = np.gradient(self.X, self.v, axis=0)
        dy_du = np.gradient(self.Y, self.u, axis=1)
        dy_dv = np.gradient(self.Y, self.v, axis=0)
        dz_du = np.gradient(self.Z, self.u, axis=1)
        dz_dv = np.gradient(self.Z, self.v, axis=0)

        cross_x = dy_du * dz_dv - dz_du * dy_dv
        cross_y = dz_du * dx_dv - dx_du * dz_dv
        cross_z = dx_du * dy_dv - dy_du * dx_dv

        dA = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
        area = trapezoid(trapezoid(dA, self.v, axis=0), self.u, axis=0)
        return area

    def compute_edge_length(self) -> float:
        """
        Approximate total length of the Möbius strip's edge.
        """
        u = self.u
        v_edge = self.w / 2

        def edge_coords(v: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            x = (self.R + v * np.cos(u / 2)) * np.cos(u)
            y = (self.R + v * np.cos(u / 2)) * np.sin(u)
            z = v * np.sin(u / 2)
            return x, y, z

        total_length = 0.0
        for v_val in [v_edge, -v_edge]:
            x, y, z = edge_coords(v_val)
            dx = np.gradient(x, u)
            dy = np.gradient(y, u)
            dz = np.gradient(z, u)
            ds = np.sqrt(dx**2 + dy**2 + dz**2)
            total_length += trapezoid(ds, u)

        return total_length

    def plot(self, cmap: str = 'plasma', alpha: float = 0.9, save_path: str = 'plot.png') -> None:
        """
        Render and save a 3D plot of the Möbius strip surface.
        """
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, cmap=cmap, edgecolor='k', alpha=alpha)

        ax.set_title("Möbius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()


if __name__ == "__main__":
    mobius = MobiusStrip(R=1.0, w=0.4, n=200)
    area = mobius.compute_surface_area()
    print(f"Surface Area ≈ {area:.4f} units²")

    edge_len = mobius.compute_edge_length()
    print(f"Edge Length ≈ {edge_len:.4f} units")

    mobius.plot()
