import numpy as np

def simulator(grid_size, n_particles):
    grid = np.zeros((grid_size, grid_size))
    grid_center = grid_size // 2
    #on place une particule au centre de la grille
    grid[grid_center, grid_center] = 1
    radii = np.zeros(n_particles, dtype=float)
    rmax = 1.0
    #on définit les bords du système
    for i in range(1, n_particles):
        #cercle d'injection et de desctruction
        r_spawn = rmax + 4.0
        r_kill = rmax + 5.0
        r_kill_sq= r_kill ** 2
        #on tire un angle aléatoire pour la particule
        theta = np.random.uniform(0, 2 * np.pi)
        #toutes les particules spawn au même rayon r_spawn
        x = int(round(grid_center + r_spawn * np.cos(theta)))
        y = int(round(grid_center + r_spawn * np.sin(theta)))
        #on vérifie que la particule est dans les limites de la grille
        if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
            continue
        #on fait bouger la particule jusqu'à ce qu'elle atteigne le centre ou qu'elle sorte de la grille
        while True:
            #on tire un angle aléatoire pour le mouvement de la particule
            theta = np.random.uniform(0, 2 * np.pi)
            # on fait bouger la particule via mouvement brownien (marche aléatoire)
            x_new = x + np.random.choice([-1, 0, 1])
            y_new = y + np.random.choice([-1, 0, 1]) #si on fait ca le problème est que 
            #on vérifie que la particule est dans les limites de la grille
            if x_new < 0 or x_new >= grid_size or y_new < 0 or y_new >= grid_size:
                break
            #on vérifie si la particule est proche du centre
            if np.sqrt((x - grid_center) ** 2 + (y - grid_center) ** 2) <= 1.0:
                grid[x, y] = 1
                radii[i] = np.sqrt((x - grid_center) ** 2 + (y - grid_center) ** 2)
                rmax = max(rmax, radii[i])
                break
            #si la particule touche le cercle de destruction, on l'élimine
            if (x - grid_center) ** 2 + (y - grid_center) ** 2 >= r_kill_sq:
                break
            #contact avec le voisin(déjà 1 dans la grille) alors on ajoute 1 à la position d'avant
            if grid[x_new, y_new] == 1:
                grid[x, y] = 1
                radii[i] = np.sqrt((x - grid_center) ** 2 + (y - grid_center) ** 2)
                rmax = max(rmax, radii[i])
                break
            #on met à jour les coordonnées de la particule
            x, y = x_new, y_new
            #on retourne la grille et les rayons des particules
    return grid, radii

def calcul_dimension_fractale(radii):
    #on calcule la dimension fractale via la méthode de box-counting
    rmax = np.max(radii)
    rmin = np.min(radii[radii > 0])
    n_boxes = []
    box_sizes = []
    for box_size in np.logspace(np.log10(rmin), np.log10(rmax), num=20):
        n_box = np.sum(radii <= box_size)
        n_boxes.append(n_box)
        box_sizes.append(box_size)
    #on fait une régression linéaire sur les points (log(box_size), log(n_boxes))
    log_box_sizes = np.log(box_sizes)
    log_n_boxes = np.log(n_boxes)
    coeffs = np.polyfit(log_box_sizes, log_n_boxes, 1)
    dimension_fractale = coeffs[0]
    return dimension_fractale

def plot(grid, radii):
    import matplotlib.pyplot as plt
    plt.imshow(grid)
    plt.title('Simulation de DLA')
    plt.show()
    plt.hist(radii[radii > 0], bins=30)
    plt.title('Distribution des rayons des particules')
    plt.xlabel('Rayon')
    plt.ylabel('Nombre de particules')
    plt.show()

if __name__ == "__main__":
    grid_size = 101
    n_particles = 10000
    grid, radii = simulator(grid_size, n_particles)
    dimension_fractale = calcul_dimension_fractale(radii)
    print(f"Dimension fractale estimée: {dimension_fractale}")
    plot(grid, radii)


"""après plot des figures on constate qu'on est tombé dans le cas des "growing animals" 
    donc la proba de croissance de chaque périmètre est uniforme, 
    et souvent l'agrégat a tendance à se développer dans une direction particulière, ``
    bug avec la dimension fractale  qui est estimée à ~0.01"""
    
