import numpy as np
import random
import math
import matplotlib.pyplot as plt

#même modèle que premier_code.py (DLA de Witten et Sander) mais les particules sont entraînées par un COURANT
#(un écoulement uniforme, ou un champ qui pousse des particules chargées, ou la gravité pour de la sédimentation)
#
#comment on met le courant dans la marche aléatoire :
#chaque pas reste un pas de 1 maille vers haut / bas / gauche / droite, mais les 4 directions ne sont plus
#équiprobables. si u est le vecteur unitaire du courant et epsilon = "courant" (entre 0 et 1) :
#     p(direction d) = (1 + epsilon * (d . u)) / 4
#- la somme des 4 probabilités vaut 1 car les (d . u) se compensent deux à deux (d et -d)
#- toutes les probabilités restent >= 0 tant que epsilon <= 1 car |d . u| <= 1
#- epsilon = 0 : on retrouve exactement la DLA normale
#- vitesse moyenne de dérive : v = somme des p(d) * d = epsilon / 2 * u   (en mailles par pas)
#- coefficient de diffusion sans courant : D_diff = 1/4 (variance 1/2 par pas sur chaque axe = 2 D_diff)
#
#échelle de longueur importante : l_c = D_diff / v = 1 / (2 epsilon)
#- les détails de l'agrégat plus petits que l_c sont dominés par la diffusion (branches type DLA)
#- au dessus de l_c c'est le courant qui domine : les particules arrivent presque en ligne droite par l'amont,
#  l'agrégat pousse CONTRE le courant et devient plus dense (on s'attend à ce que D tende vers 2 pour
#  des agrégats beaucoup plus grands que l_c, c'est ce que Meakin a étudié en 1983 si je me souviens bien,
#  à vérifier dans l'article : Phys. Rev. B 28, 5221)
#
#problème potentiel (réseau) : avec le courant la diffusion n'est plus tout à fait isotrope :
#variance par pas le long du courant = 1/2 - epsilon²/4 et perpendiculairement = 1/2
#pour epsilon <= 0.5 la différence est de moins de 12%, pour epsilon proche de 1 elle est grande

#déplacements autorisés : haut, bas, gauche, droite (pas de diagonale)
DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


#############################################################################
#  PETITES FONCTIONS DE BASE
#############################################################################

def distance_au_centre(x, y, grid_center):
    #distance euclidienne entre le site (x, y) et la graine au centre
    dx = x - grid_center
    dy = y - grid_center
    return math.sqrt(dx * dx + dy * dy)


def est_dans_la_grille(x, y, grid_size):
    #on vérifie que la particule est dans les limites de la grille
    if x < 0 or x >= grid_size:
        return False
    if y < 0 or y >= grid_size:
        return False
    return True


def touche_agregat(grid, x, y):
    #règle de Witten et Sander : la particule s'arrête dès qu'un de ses 4 voisins est occupé
    for dx, dy in DIRECTIONS:
        if grid[x + dx, y + dy] == 1:
            return True
    return False


def probabilites_des_pas(courant, angle_courant):
    #on calcule UNE FOIS pour toute la simulation les probabilités des 4 directions
    #angle_courant en degrés : 0 = le courant va vers les x positifs (vers la droite sur la figure),
    #90 = vers les y positifs (vers le haut sur la figure)
    if courant < 0 or courant > 1:
        raise ValueError("le courant doit être entre 0 et 1 sinon certaines probabilités deviennent négatives")
    angle = math.radians(angle_courant)
    ux = math.cos(angle)
    uy = math.sin(angle)
    probas = []
    for dx, dy in DIRECTIONS:
        produit_scalaire = dx * ux + dy * uy
        p = (1 + courant * produit_scalaire) / 4
        probas.append(p)
    #probabilités cumulées pour le tirage (ex : [0.3, 0.5, 0.75, 1.0])
    probas_cumulees = []
    total = 0.0
    for p in probas:
        total = total + p
        probas_cumulees.append(total)
    #à cause des arrondis le dernier peut valoir 0.9999999999, on force 1
    probas_cumulees[-1] = 1.0
    return probas, probas_cumulees


def position_injection(grid_center, r_spawn):
    #on tire un angle aléatoire pour la particule
    #problème potentiel : avec un courant, l'injection UNIFORME sur le cercle n'est plus exactement la bonne
    #(sans courant c'est exact : partir d'un point uniforme sur un cercle qui entoure l'agrégat revient au même
    #que venir de l'infini). avec courant, les vraies particules venues de l'infini arrivent surtout par l'amont.
    #les particules injectées en aval sont emportées par le courant et détruites, ça coûte du calcul mais
    #ce n'est pas très grave ; ce qui est approximatif c'est le poids relatif des particules qui arrivent
    #par les côtés. qualitativement ça ne change pas le résultat
    theta = random.uniform(0, 2 * math.pi)
    #toutes les particules spawn au même rayon r_spawn
    x = int(round(grid_center + r_spawn * math.cos(theta)))
    y = int(round(grid_center + r_spawn * math.sin(theta)))
    return x, y


def pas_aleatoire_avec_courant(x, y, probas_cumulees):
    #on tire un nombre entre 0 et 1 et on regarde dans quel intervalle il tombe
    #ex : probas_cumulees = [0.3, 0.5, 0.75, 1.0] et tirage = 0.6 => direction numéro 2
    tirage = random.random()
    numero = 0
    while tirage > probas_cumulees[numero]:
        numero = numero + 1
    dx, dy = DIRECTIONS[numero]
    return x + dx, y + dy


def saut_rapide_avec_courant(x, y, longueur, courant, angle_courant):
    #même astuce que dans premier_code.py : quand la particule est loin de l'agrégat on la place directement
    #sur le cercle de rayon "longueur" autour d'elle
    #MAIS avec un courant le point de sortie du cercle n'est plus uniforme : la particule sort plus souvent en aval
    #résultat exact pour un mouvement brownien avec dérive v et diffusion D_diff (théorème de Girsanov) :
    #   densité de l'angle de sortie  ~  exp( v * longueur * cos(theta - angle_courant) / (2 D_diff) )
    #c'est une loi de von Mises de paramètre kappa = v * longueur / (2 D_diff)
    #avec v = courant / 2 et D_diff = 1/4 on trouve kappa = courant * longueur
    #(pour courant = 0, kappa = 0 et on retrouve la loi uniforme)
    #d'où ça vient en 2 lignes : la vraisemblance d'un chemin avec dérive divisée par celle sans dérive vaut
    #exp(v.(X_fin - X_debut)/(2 D_diff) - v² t /(4 D_diff)) ; pour un brownien parti du centre d'un disque
    #le temps de sortie t et l'angle de sortie sont indépendants donc seul le premier facteur dépend de l'angle
    #problème potentiel : c'est exact pour le brownien continu, sur le réseau c'est une approximation qui devient
    #moins bonne quand le courant est proche de 1 (diffusion anisotrope, voir en haut du fichier)
    #random.vonmisesvariate existe directement dans le module random
    kappa = courant * longueur
    theta = random.vonmisesvariate(math.radians(angle_courant), kappa)
    x_new = int(round(x + longueur * math.cos(theta)))
    y_new = int(round(y + longueur * math.sin(theta)))
    return x_new, y_new


#############################################################################
#  UNE PARTICULE
#############################################################################

def lancer_une_particule(grid, grid_center, rmax, r_spawn, r_kill, probas_cumulees, courant, angle_courant, utiliser_sauts):
    #on fait marcher UNE particule jusqu'à ce qu'elle colle (on renvoie sa position)
    #ou qu'elle soit détruite (on renvoie None)
    grid_size = grid.shape[0]
    x, y = position_injection(grid_center, r_spawn)

    while True:
        d = distance_au_centre(x, y, grid_center)

        #si la particule touche le cercle de destruction, on l'élimine
        #avec un courant les particules emportées en aval ne reviennent quasiment jamais (la marche n'est
        #plus récurrente) donc les détruire ne crée presque pas de biais
        if d > r_kill:
            return None

        #si la particule est loin de l'agrégat on fait un grand saut au lieu d'un petit pas
        longueur = d - rmax - 2
        if utiliser_sauts and longueur >= 2:
            x, y = saut_rapide_avec_courant(x, y, longueur, courant, angle_courant)
            continue

        #on vérifie que la particule est dans les limites de la grille (avec 1 maille de marge pour les voisins)
        if not est_dans_la_grille(x - 1, y - 1, grid_size) or not est_dans_la_grille(x + 1, y + 1, grid_size):
            return None

        #contact avec un voisin (déjà 1 dans la grille) : la particule colle là où elle est
        if touche_agregat(grid, x, y):
            return x, y

        #on met à jour les coordonnées de la particule (pas biaisé par le courant)
        x, y = pas_aleatoire_avec_courant(x, y, probas_cumulees)


#############################################################################
#  SIMULATION COMPLÈTE
#############################################################################

def simulator(grid_size, n_particles, courant=0.0, angle_courant=0.0, utiliser_sauts=True, graine=None):
    #courant : entre 0 (pas de courant, DLA normale) et 1 (courant maximal)
    #angle_courant : direction vers laquelle va le courant, en degrés (0 = droite, 90 = haut sur la figure)
    #n_particles = nombre de particules COLLÉES qu'on veut
    if graine is not None:
        random.seed(graine)
        np.random.seed(graine)

    probas, probas_cumulees = probabilites_des_pas(courant, angle_courant)
    print(f"probabilités des pas (x+, x-, y+, y-) : {[round(p, 3) for p in probas]}")
    if courant > 0:
        print(f"longueur de croisement diffusion/courant l_c = 1/(2 courant) = {1 / (2 * courant):.1f} mailles")

    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    grid_center = grid_size // 2
    #on place une particule au centre de la grille
    grid[grid_center, grid_center] = 1

    positions = [(grid_center, grid_center)]
    radii = []
    rmax = 1.0
    n_lancees = 0

    while len(radii) < n_particles:
        #cercle d'injection et de destruction
        r_spawn = rmax + 5.0
        if utiliser_sauts:
            r_kill = 20.0 * r_spawn
        else:
            r_kill = min(3.0 * r_spawn, grid_center - 2)

        #on arrête si l'agrégat arrive près du bord de la grille
        #(avec un courant fort l'agrégat pousse vite vers l'amont, rmax grandit plus vite qu'en DLA)
        if r_spawn + 3 >= grid_center:
            print(f"arrêt : l'agrégat touche presque le bord de la grille ({len(radii)} particules collées)")
            break

        n_lancees += 1
        resultat = lancer_une_particule(grid, grid_center, rmax, r_spawn, r_kill, probas_cumulees,
                                        courant, angle_courant, utiliser_sauts)

        #particule détruite : on en relance une autre
        if resultat is None:
            continue

        x, y = resultat
        grid[x, y] = 1
        positions.append((x, y))
        r = distance_au_centre(x, y, grid_center)
        radii.append(r)
        rmax = max(rmax, r)

    #avec un courant fort, beaucoup plus de particules sont lancées que collées (celles injectées en aval)
    print(f"{len(radii)} particules collées pour {n_lancees} lancées")
    return grid, np.array(radii), np.array(positions)


#############################################################################
#  DIMENSION FRACTALE
#############################################################################

def dimension_rayon_giration(positions, n_min=50):
    #pendant la croissance, le rayon de giration Rg de l'agrégat de n particules vérifie n ~ Rg^D
    #avec un courant c'est la méthode la plus fiable des deux : l'agrégat n'est plus centré sur la graine
    #(il pousse vers l'amont) et Rg est calculé autour du centre de masse, pas autour de la graine
    x = positions[:, 0].astype(float)
    y = positions[:, 1].astype(float)
    n = np.arange(1, len(x) + 1)
    moy_x = np.cumsum(x) / n
    moy_y = np.cumsum(y) / n
    moy_x2 = np.cumsum(x * x) / n
    moy_y2 = np.cumsum(y * y) / n
    #Rg² = <x²> - <x>² + <y²> - <y>²
    rg = np.sqrt(moy_x2 - moy_x ** 2 + moy_y2 - moy_y ** 2)
    garder = n >= n_min
    coeffs = np.polyfit(np.log(rg[garder]), np.log(n[garder]), 1)
    #problème potentiel : si l'agrégat passe d'un régime DLA (petite taille) à un régime dense (grande taille)
    #log(n) en fonction de log(Rg) n'est pas une droite et la pente est une moyenne des deux régimes
    return coeffs[0]


def dimension_box_counting(grid):
    #on recouvre l'agrégat avec des boîtes de côté s et on compte les boîtes qui contiennent au moins 1 particule
    #N_boites(s) ~ s^(-D)
    occupes_x, occupes_y = np.nonzero(grid)
    x0 = occupes_x.min()
    y0 = occupes_y.min()
    cote = max(occupes_x.max() - x0, occupes_y.max() - y0) + 1
    tailles = []
    n_boites = []
    s = 2
    while s <= cote / 4:
        bx = (occupes_x - x0) // s
        by = (occupes_y - y0) // s
        boites = set(zip(bx.tolist(), by.tolist()))
        tailles.append(s)
        n_boites.append(len(boites))
        s = s * 2
    coeffs = np.polyfit(np.log(tailles), np.log(n_boites), 1)
    return -coeffs[0]


#############################################################################
#  FIGURES
#############################################################################

def plot(grid, positions, courant, angle_courant, nom_fichier="figure_courant.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    #1) l'agrégat coloré par ordre d'arrivée, avec une flèche pour le courant
    #on transpose et on met origin='lower' pour que x soit horizontal et y vertical
    ordre = np.full(grid.shape, np.nan)
    for i, (x, y) in enumerate(positions):
        ordre[x, y] = i
    ax = axes[0]
    ax.imshow(ordre.T, origin='lower', cmap='viridis', interpolation='nearest')
    if courant > 0:
        taille = grid.shape[0]
        angle = math.radians(angle_courant)
        longueur_fleche = 0.15 * taille
        x_debut = 0.15 * taille
        y_debut = 0.85 * taille
        ax.arrow(x_debut, y_debut, longueur_fleche * math.cos(angle), longueur_fleche * math.sin(angle),
                 width=0.01 * taille, color='red')
        ax.text(x_debut, y_debut - 0.05 * taille, 'courant', color='red')
    ax.set_title(f'DLA avec courant = {courant} (couleur = ordre d\'arrivée)')

    #2) rayon de giration en fonction du nombre de particules (log-log)
    #pour une DLA normale c'est une droite de pente 1/D ~ 0.58
    #avec courant on s'attend à une pente qui change quand Rg dépasse l_c
    x = positions[:, 0].astype(float)
    y = positions[:, 1].astype(float)
    n = np.arange(1, len(x) + 1)
    rg = np.sqrt(np.cumsum(x * x) / n - (np.cumsum(x) / n) ** 2 + np.cumsum(y * y) / n - (np.cumsum(y) / n) ** 2)
    ax = axes[1]
    ax.loglog(n[1:], rg[1:], label='simulation')
    ax.loglog(n[1:], rg[-1] * (n[1:] / n[-1]) ** (1 / 1.71), 'k--', label='pente 1/1.71 (DLA)')
    ax.loglog(n[1:], rg[-1] * (n[1:] / n[-1]) ** (1 / 2), 'r:', label='pente 1/2 (objet compact)')
    if courant > 0:
        ax.axhline(1 / (2 * courant), color='gray', ls='--', label='l_c = 1/(2 courant)')
    ax.set_xlabel('nombre de particules n')
    ax.set_ylabel('rayon de giration Rg')
    ax.legend()

    plt.tight_layout()
    plt.savefig(nom_fichier, dpi=120)
    plt.show()


if __name__ == "__main__":
    grid_size = 401
    n_particles = 5000
    #PARAMÈTRE DE COURANT : 0 = DLA normale, 1 = courant maximal
    courant = 0.2
    #direction du courant en degrés : 0 = vers la droite, 90 = vers le haut, 270 = vers le bas
    angle_courant = 270
    grid, radii, positions = simulator(grid_size, n_particles, courant=courant, angle_courant=angle_courant)
    print(f"Dimension fractale estimée (rayon de giration): {dimension_rayon_giration(positions):.3f}")
    print(f"Dimension fractale estimée (box-counting): {dimension_box_counting(grid):.3f}")
    #on ne refait pas la méthode masse-rayon de premier_code.py : elle compte les particules dans des disques
    #centrés sur la graine, or avec un courant l'agrégat n'est plus centré sur la graine
    plot(grid, positions, courant, angle_courant)
