import numpy as np
import matplotlib.pyplot as plt

# Define stress components (you can replace these with your actual data)
sigma_xx = np.linspace(-150, 150, 100)
sigma_yy = np.linspace(-150, 150, 100)
sigma_zz_values = np.array([10, 20, 30])  # Choose different shear stress values

# Create a meshgrid for sigma_xx and sigma_yy
sigma_xx_grid, sigma_yy_grid = np.meshgrid(sigma_xx, sigma_yy)

# Compute equivalent von Mises stress for each sigma_zz value
for sigma_zz in sigma_zz_values:
    sigma_eq = np.sqrt(sigma_xx_grid**2 + sigma_yy_grid**2 + 3*sigma_zz**2
                              - sigma_xx_grid * sigma_yy_grid)

    # Create contour plot
    plt.contour(sigma_xx_grid, sigma_yy_grid, sigma_eq, levels=1, cmap='viridis', label=f"σ_zz = {sigma_zz}")

plt.xlabel(r'$\sigma_{xx}$')
plt.ylabel(r'$\sigma_{yy}$')
plt.title('Von Mises Yield Surface Slices')
plt.legend()
plt.grid(True)
plt.colorbar(label='Equivalent Von Mises Stress')
plt.show()
print('hey')