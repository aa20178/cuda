#ifndef MATRICE_H_ /* Include guard */
#define MATRICE_H_
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>

void user_input(bool& affiche, int& n, int argc, char **argv);
void genmat(float *A, int n);
float verify(const float *A, const float *B, int n);
void afficher_matrice(float *A, int nombre_lignes, int nombre_colonnes);
void afficher_matrice(float *A, int n);
void multiplier_matrice_naive(float *A, float *B, float *C, int n);
void multiplier_matrice(float *A, float *B, float *C, int n);
void multiplier_matrice(float *A, float *B, float *C, int nb_lignesA, int nb_colonnesA, int nb_lignesB, int nb_colonnesB);
void soustraction_2_matrices(float *A, float *B, float *C, int n);
void transpose_matrice_naive(float *A, float *B, int n);
void transpose_matrice(float *A, float *B, int n);
void transpose_matrice(float *A, float *B, int n, int m);
int compter_inegalites(float *h_A, float *h_B, int n) ;
bool egalite_2_matrices(float *h_A, float *h_B, int n) ;
void free_cpu(float *h_A);
void free_gpu(float *h_A);
void affichage_performance(float t_ms, float octets_echanges);
void affichage_resultats_du_kernel(float *matrice1, float *matrice2, int taille_des_matrices_carree, float temps_en_milisecondes, float octets_echanges, bool affiche);
void affichage_resultats_du_kernel(float *matrice1, float *matrice2, float *matrice_res, int taille_des_matrices_carree, float temps_en_milisecondes, float octets_echanges, bool affiche);
void affichage_resultats_du_kernel(float *matrice1, float *matrice2, float *matrice_res,float *matrice_res2, int taille_des_matrices_carree, float temps_en_milisecondes, float octets_echanges, bool affiche);



#endif