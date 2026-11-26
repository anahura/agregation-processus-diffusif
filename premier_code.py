import numpy as np
import random
import math
import matplotlib.pyplot as plt

#on utilise le module random de python (et pas np.random) pour tirer UN SEUL nombre à la fois
#np.random.randint(4) coûte ~1 microseconde par appel alors que random.randrange(4) est ~10 fois plus rapide
#comme on fait des millions de pas c'est ce qui domine le temps de calcul

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
    #ATTENTION c'était un des problèmes de l'ancien code : on ne regardait QUE le site
    #vers lequel la particule essayait d'aller (grid[x_new, y_new]) donc une particule collée
    #à l'agrégat avait seulement 1 chance sur 4 de s'arrêter à chaque pas
    #(ça revient à une probabilité de collage < 1, ce n'est pas le modèle de l'article)
    for dx, dy in DIRECTIONS:
        if grid[x + dx, y + dy] == 1:
            return True
    return False


def position_injection(grid_center, r_spawn):
    #on tire un angle aléatoire pour la particule
    theta = random.uniform(0, 2 * math.pi)
    #toutes les particules spawn au même rayon r_spawn
    x = int(round(grid_center + r_spawn * math.cos(theta)))
    y = int(round(grid_center + r_spawn * math.sin(theta)))
    return x, y


def pas_aleatoire(x, y):
    #on fait bouger la particule via mouvement brownien (marche aléatoire)
    #pas de déplacement en diagonale : un seul des deux axes bouge de ±1
    dx, dy = DIRECTIONS[random.randrange(4)]
    return x + dx, y + dy


def saut_rapide(x, y, longueur):
    #ACCÉLÉRATION (astuce classique de Meakin) :
    #quand la particule est loin de l'agrégat, faire des pas de 1 est une perte de temps énorme
    #si aucun site de l'agrégat n'est à moins de "longueur" de la particule, alors la marche aléatoire
    #va forcément sortir du cercle de rayon "longueur" centré sur la particule avant de toucher quoi que ce soit
    #et pour un mouvement brownien partant du centre d'un cercle, le point de sortie est UNIFORME sur le cercle
    #(par symétrie de rotation) donc on peut directement placer la particule sur ce cercle
    #problème potentiel : c'est exact pour le mouvement brownien continu, pour la marche sur réseau c'est
    #une approximation (très bonne dès que la longueur fait plus de quelques mailles)
    #on perd aussi l'information sur le TEMPS mis par la particule mais ici on n'en a pas besoin
    theta = random.uniform(0, 2 * math.pi)
    x_new = int(round(x + longueur * math.cos(theta)))
    y_new = int(round(y + longueur * math.sin(theta)))
    return x_new, y_new


#############################################################################
#  UNE PARTICULE
#############################################################################

def lancer_une_particule(grid, grid_center, rmax, r_spawn, r_kill, utiliser_sauts):
    #on fait marcher UNE particule jusqu'à ce qu'elle colle (on renvoie sa position)
    #ou qu'elle soit détruite (on renvoie None)
    grid_size = grid.shape[0]
    x, y = position_injection(grid_center, r_spawn)

    while True:
        d = distance_au_centre(x, y, grid_center)

        #si la particule touche le cercle de destruction, on l'élimine
        #ATTENTION c'était le GROS problème de l'ancien code : r_kill = rmax + 5 et r_spawn = rmax + 4
        #la particule naissait à 1 maille du cercle de destruction donc presque toutes mouraient au bout
        #de quelques pas (d'où les ~7% de particules collées seulement)
        #et surtout les seules qui survivaient étaient celles qui naissaient juste en face de la pointe
        #la plus avancée (la seule partie de l'agrégat à moins de ~4 mailles du cercle d'injection)
        #donc la pointe la plus longue grandissait encore plus => l'agrégat part d'un seul côté
        #ici r_kill est beaucoup plus grand que r_spawn (voir simulator)
        if d > r_kill:
            return None

        #si la particule est loin de l'agrégat on fait un grand saut au lieu d'un petit pas
        #tous les sites occupés sont à une distance <= rmax du centre, donc à une distance >= d - rmax
        #de la particule ; on garde une marge de 2 mailles pour l'arrondi sur le réseau
        longueur = d - rmax - 2
        if utiliser_sauts and longueur >= 2:
            x, y = saut_rapide(x, y, longueur)
            continue

        #sinon on est près de l'agrégat : marche aléatoire normale sur le réseau
        #on vérifie que la particule est dans les limites de la grille (avec 1 maille de marge pour les voisins)
        if not est_dans_la_grille(x - 1, y - 1, grid_size) or not est_dans_la_grille(x + 1, y + 1, grid_size):
            #ça ne devrait arriver que si utiliser_sauts = False et que r_kill est trop grand pour la grille
            return None

        #contact avec un voisin (déjà 1 dans la grille) : la particule colle là où elle est
        if touche_agregat(grid, x, y):
            return x, y

        #on met à jour les coordonnées de la particule
        x, y = pas_aleatoire(x, y)


#############################################################################
#  SIMULATION COMPLÈTE
#############################################################################

def simulator(grid_size, n_particles, utiliser_sauts=True, graine=None):
    #n_particles = nombre de particules COLLÉES qu'on veut (et pas le nombre de particules lancées)
    #dans l'ancien code chaque particule détruite était perdue et laissait un 0 dans radii,
    #c'est ce qui faisait planter la dimension fractale (voir calcul_dimension_fractale)
    if graine is not None:
        random.seed(graine)
        np.random.seed(graine)

    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    grid_center = grid_size // 2
    #on place une particule au centre de la grille
    grid[grid_center, grid_center] = 1

    #on garde la position et le rayon de chaque particule collée (dans l'ordre d'arrivée)
    positions = [(grid_center, grid_center)]
    radii = []
    rmax = 1.0
    n_lancees = 0

    while len(radii) < n_particles:
        #cercle d'injection et de destruction
        r_spawn = rmax + 5.0
        if utiliser_sauts:
            #avec les sauts, s'éloigner coûte très peu de calcul (la distance est multipliée à chaque saut)
            #donc on peut mettre le cercle de destruction très loin
            #problème potentiel : tuer une particule et en relancer une nouvelle uniformément sur le cercle
            #d'injection n'est pas exactement équivalent à la laisser revenir (en 2D elle reviendrait toujours,
            #la marche aléatoire est récurrente) mais si r_kill >> rmax elle reviendrait de manière quasi uniforme
            #donc l'erreur est négligeable
            r_kill = 20.0 * r_spawn
        else:
            #sans les sauts, le cercle de destruction doit tenir dans la grille
            #problème potentiel : si r_kill n'est pas beaucoup plus grand que r_spawn on retombe sur
            #le problème de l'ancien code (biais vers les pointes)
            r_kill = min(3.0 * r_spawn, grid_center - 2)

        #on arrête si l'agrégat arrive près du bord de la grille
        if r_spawn + 3 >= grid_center:
            print(f"arrêt : l'agrégat touche presque le bord de la grille ({len(radii)} particules collées)")
            break

        n_lancees += 1
        resultat = lancer_une_particule(grid, grid_center, rmax, r_spawn, r_kill, utiliser_sauts)

        #particule détruite : on en relance une autre
        if resultat is None:
            continue

        x, y = resultat
        grid[x, y] = 1
        positions.append((x, y))
        r = distance_au_centre(x, y, grid_center)
        radii.append(r)
        rmax = max(rmax, r)

    print(f"{len(radii)} particules collées pour {n_lancees} lancées")
    #on retourne la grille, les rayons des particules et leurs positions
    return grid, np.array(radii), np.array(positions)


#############################################################################
#  DIMENSION FRACTALE
#############################################################################

def calcul_dimension_fractale(radii, r_min_fit=3.0, fraction_max_fit=0.5):
    #méthode masse-rayon (et pas box-counting comme écrit avant : on compte des particules dans des disques,
    #pas des boîtes) : N(r) = nombre de particules à une distance <= r de la graine ~ r^D
    #BUG de l'ancien code : radii contenait un 0 pour chaque particule détruite (~90% des particules)
    #donc N(r) était quasiment constant (~9000 + quelques centaines) et la pente valait ~0.01
    #ici radii ne contient que les particules collées
    rmax = np.max(radii)
    #on ne fitte pas sur tout l'intervalle :
    #- en dessous de quelques mailles le réseau carré domine (pas de comportement en loi de puissance)
    #- au dessus de ~rmax/2 l'agrégat n'est pas "fini" : c'est la zone active qui est encore en train de
    #  pousser, il y manque des particules et N(r) s'aplatit (ça ferait baisser D)
    #problème potentiel : le choix de ces bornes change un peu le résultat (de l'ordre de ±0.05)
    r_max_fit = fraction_max_fit * rmax
    rayons = np.logspace(np.log10(r_min_fit), np.log10(r_max_fit), num=20)
    n_dans_disque = []
    for r in rayons:
        #+1 pour la graine au centre
        n = np.sum(radii <= r) + 1
        n_dans_disque.append(n)
    #on fait une régression linéaire sur les points (log(r), log(N))
    log_r = np.log(rayons)
    log_n = np.log(n_dans_disque)
    coeffs = np.polyfit(log_r, log_n, 1)
    dimension_fractale = coeffs[0]
    #on renvoie aussi l'ordonnée à l'origine : N(r) = exp(coeffs[1]) * r^D (sert pour le tracé de l'histogramme)
    return dimension_fractale, coeffs[1]


def dimension_rayon_giration(positions, n_min=50):
    #deuxième méthode indépendante : pendant la croissance, le rayon de giration Rg de l'agrégat
    #de n particules vérifie Rg ~ n^(1/D)  <=>  n ~ Rg^D
    #avantage : ne dépend pas du choix du centre, et utilise toute l'histoire de la croissance
    x = positions[:, 0].astype(float)
    y = positions[:, 1].astype(float)
    n = np.arange(1, len(x) + 1)
    #moyennes cumulées : <x>, <y>, <x²>, <y²> pour les n premières particules
    moy_x = np.cumsum(x) / n
    moy_y = np.cumsum(y) / n
    moy_x2 = np.cumsum(x * x) / n
    moy_y2 = np.cumsum(y * y) / n
    #Rg² = <x²> - <x>² + <y²> - <y>²
    rg = np.sqrt(moy_x2 - moy_x ** 2 + moy_y2 - moy_y ** 2)
    #on enlève les petits agrégats (effets de réseau)
    garder = n >= n_min
    coeffs = np.polyfit(np.log(rg[garder]), np.log(n[garder]), 1)
    return coeffs[0]


def dimension_box_counting(grid):
    #troisième méthode : le vrai box-counting
    #on recouvre l'agrégat avec des boîtes de côté s et on compte les boîtes qui contiennent au moins 1 particule
    #N_boites(s) ~ s^(-D)
    occupes_x, occupes_y = np.nonzero(grid)
    #on découpe un carré qui contient juste l'agrégat
    x0 = occupes_x.min()
    y0 = occupes_y.min()
    cote = max(occupes_x.max() - x0, occupes_y.max() - y0) + 1
    tailles = []
    n_boites = []
    s = 2
    #problème potentiel : quand s devient comparable à la taille de l'agrégat il n'y a plus que
    #quelques boîtes et la statistique est mauvaise, on s'arrête à cote / 4
    while s <= cote / 4:
        #numéro de la boîte de chaque particule
        bx = (occupes_x - x0) // s
        by = (occupes_y - y0) // s
        #nombre de boîtes différentes occupées
        boites = set(zip(bx.tolist(), by.tolist()))
        tailles.append(s)
        n_boites.append(len(boites))
        s = s * 2
    coeffs = np.polyfit(np.log(tailles), np.log(n_boites), 1)
    return -coeffs[0]


#############################################################################
#  FIGURES
#############################################################################

def plot(grid, radii, positions, dimension_fractale, log_prefacteur, nom_fichier="figure_dla.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    #1) l'agrégat, coloré par ordre d'arrivée
    #on transpose et on met origin='lower' pour que x soit horizontal et y vertical
    #(avec plt.imshow(grid) directement, x est la ligne donc l'axe vertical vers le BAS, c'est trompeur)
    ordre = np.full(grid.shape, np.nan)
    for i, (x, y) in enumerate(positions):
        ordre[x, y] = i
    ax = axes[0]
    ax.imshow(ordre.T, origin='lower', cmap='viridis', interpolation='nearest')
    ax.set_title('Simulation de DLA (couleur = ordre d\'arrivée)')

    #2) distribution des rayons
    #comme les particules ne bougent plus une fois collées, le nombre de particules entre r et r+dr est
    #dN = dN/dr * dr avec N(r) = A r^D donc dN/dr = D A r^(D-1) : l'histogramme doit CROÎTRE comme r^(D-1) ~ r^0.7
    #(c'est la "forme particulière" attendue) puis chuter vers rmax (zone active pas encore remplie)
    #dans l'ancien code l'agrégat poussait comme une ligne (D~1) donc l'histogramme était à peu près plat
    ax = axes[1]
    comptes, bords, _ = ax.hist(radii, bins=40, alpha=0.6, label='simulation')
    largeur = bords[1] - bords[0]
    r = np.linspace(1, radii.max(), 200)
    #courbe attendue SANS paramètre libre : D et A viennent du fit masse-rayon
    attendu = dimension_fractale * np.exp(log_prefacteur) * r ** (dimension_fractale - 1) * largeur
    ax.plot(r, attendu, 'r-', label=f'D A r^(D-1), D = {dimension_fractale:.2f}')
    ax.set_title('Distribution des rayons des particules')
    ax.set_xlabel('Rayon')
    ax.set_ylabel('Nombre de particules')
    ax.legend()

    #3) masse-rayon en log-log
    ax = axes[2]
    rayons = np.logspace(0, np.log10(radii.max()), 40)
    n_dans_disque = [np.sum(radii <= rr) + 1 for rr in rayons]
    ax.loglog(rayons, n_dans_disque, 'o', label='N(r) simulation')
    ax.loglog(rayons, np.exp(log_prefacteur) * rayons ** dimension_fractale, 'r-', label=f'pente D = {dimension_fractale:.2f}')
    ax.axvline(radii.max() / 2, color='gray', ls='--', label='rmax/2 (fin du fit)')
    ax.set_xlabel('r')
    ax.set_ylabel('N(r)')
    ax.set_title('Méthode masse-rayon')
    ax.legend()

    plt.tight_layout()
    plt.savefig(nom_fichier, dpi=120)
    plt.show()


if __name__ == "__main__":
    #la grille doit être assez grande pour contenir l'agrégat : rmax ~ n_particles^(1/1.71)
    #pour 5000 particules on attend rmax ~ 100-150 mailles
    grid_size = 401
    n_particles = 5000
    grid, radii, positions = simulator(grid_size, n_particles)
    dimension_fractale, log_prefacteur = calcul_dimension_fractale(radii)
    print(f"Dimension fractale estimée (masse-rayon): {dimension_fractale:.3f}")
    print(f"Dimension fractale estimée (rayon de giration): {dimension_rayon_giration(positions):.3f}")
    print(f"Dimension fractale estimée (box-counting): {dimension_box_counting(grid):.3f}")
    print("valeur attendue pour la DLA en 2D : D ~ 1.71")
    plot(grid, radii, positions, dimension_fractale, log_prefacteur)


"""ce qui n'allait pas dans la version d'avant :
    1) r_kill = rmax + 5 juste à côté de r_spawn = rmax + 4 : ~93% des particules étaient détruites tout de suite,
       et seules celles qui naissaient en face de la pointe la plus avancée survivaient => rétroaction positive,
       la pointe la plus longue grandit toujours plus vite et l'agrégat part d'un seul côté
       (ce n'était pas les "growing animals" : les animaux/Eden donnent un agrégat compact D = 2, pas une ligne)
    2) on ne testait que le voisin vers lequel la particule allait (proba de collage 1/4 par voisin occupé
       au lieu de 1 comme chez Witten-Sander)
    3) la dimension ~0.01 venait des zéros dans radii pour les particules perdues
    4) le test "distance au centre <= 1" était inutile (déjà couvert par le test des voisins) et l'angle tiré
       dans la boucle while ne servait à rien
    5) grille 101 trop petite : dès que rmax + 4 > 50 toutes les particules spawnaient dehors (continue)
"""
