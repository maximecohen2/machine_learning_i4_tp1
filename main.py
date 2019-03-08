#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Multi-layer perceptron
# -----------------------------------------------------------------------------

import samples.samples_triangle as st
import samples.samples_carre as sc
import samples.samples_hexagone as sh
import samples.samples_octogone as so
from pylab import *
from MLP import MLP


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    global ListeSamples
    ListeSamples = []

    global titreGraphique
    titreGraphique = ''
    
    def learn(network, samples, epochs=2500, lrate=.001, momentum=0.1):
        # Train
        input = [s[0] for s in samples]
        output = [s[1] for s in samples]
        for e in range(epochs):
            for i in input:
                network.propagate_forward(i)
        print('    Train')

    def test(network, listeSamples, titreGraphique):
        # Test
        print('    Test')


    def learnCarre(nb_samples_triangle):
        # Exemple 6: Learning square
        # ---------------------------------------------------------------------
        print("Learning the carre")

        samples_c = np.zeros(2*nb_samples_carre, dtype=[('input',  float, 2), ('output', float, 1)])
        carre = sc.samples(nb_samples_carre)
        carre.create_samples()
        for i in range(carre.nb_samples):
            samples_c[i] = carre.samples[i]

        #displayForm(carre.samples_in, carre.samples_out)
        listeSamples = sc.exercice.samples_list

        titreGraphique = ' Apprentissage du carre'

        # Etape 1 : Déclarer un réseau MLP avec une stucture répondant à la tache demandé
        mlp = MLP(2, 10, 1)

        # Etape 2 : réalisé l'apprentissage
        learn(mlp, carre.samples, 1)

        # Etape 3 : Réalisé la phase de test

    def learnTriangle(nb_samples_triangle):
        # Exemple 5: Learning triangulus
        # -------------------------------------------------------------------------
        print("Learning the triangle")

        samples_t = np.zeros(2 * nb_samples_triangle, dtype=[('input', float, 2), ('output', float, 1)])
        triangle = st.samples(nb_samples_triangle)
        triangle.create_samples()
        for i in range(triangle.nb_samples):
            samples_t[i] = triangle.samples[i]

        listeSamples = st.exercice.samples_list
        displayForm(triangle.samples_in, triangle.samples_out)
        titreGraphique = 'Apprentissage du triangle'
        # Etape 1 : Déclarer un réseau MLP avec une stucture répondant à la tache demandé

        # Etape 2 : réalisé l'apprentissage

        # Etape 3 : Réalisé la phase de test


    def displayForm(liste_sample_in, liste_sample_out):
        print('    Display')
        x, y = [], []
        for i in range(len(liste_sample_in)):
            x.append(liste_sample_in[i][0][0])
            y.append(liste_sample_in[i][0][1])
            plot(x, y, 'bx')

        x, y = [], []

        for i in range(len(liste_sample_out)):
            x.append(liste_sample_out[i][0][0])
            y.append(liste_sample_out[i][0][1])
            plot(x, y, 'rx')

        axis([-6, 6, -6, 6])
        grid()          
        show()

    ##############################################################################    
    nb_samples_carre = 1000                                                      #
    nb_samples_triangle = 1000                                                   #
    ##############################################################################

    learnCarre(nb_samples_carre)
    #learnTriangle(nb_samples_triangle)
