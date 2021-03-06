#include "matrice.h"

void user_input(bool& affiche, int& n, int argc, char **argv)
{
	if (argc < 2)
	{
		std::cout << argc << " il faut entrer un argument (taille matrice) " << std::endl;
		exit(-1);
	}
	if (argv[1] != NULL && atoi(argv[1]) > 1)
	{
		n = atoi(argv[1]);
	}
	if (argv[2] != NULL)
	{
		affiche = true;
	}
}

void genmat(float *A, int n)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			A[i * n + j] = rand() / (float)RAND_MAX;
}
float verify(const float *A, const float *B, int n)
{
	float error = 0;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			error = std::max(error, abs(A[i * n + j] - B[i * n + j]));

	return error;
}

void afficher_matrice(float *A, int nombre_lignes, int nombre_colonnes)
{
	for (int i = 0; i < nombre_lignes; i++)
	{
		for (int j = 0; j < nombre_colonnes; j++)
		{
			std::cout << A[i * nombre_colonnes + j] << " ";
		}
		std::cout << std::endl;
	}
}

void afficher_matrice(float *A, int n)
{
	afficher_matrice(A, n, n);
}

void multiplier_matrice_naive(float *A, float *B, float *C, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			C[j + (i * n)] = 0;
			for (int k = 0; k < n; k++)
			{
				C[j + (i * n)] = C[j + (i * n)] + A[(i * n) + k] * B[(k * n) + j];
			}
		}
	}
}
void multiplier_matrice(float *A, float *B, float *C, int n)
{
	multiplier_matrice(A, B, C, n, n, n, n);
}

void multiplier_matrice(float *A, float *B, float *C, int nb_lignesA, int nb_colonnesA, int nb_lignesB, int nb_colonnesB)
{
	// manque la verif : A[n,m] * B[m,p] donne C[n,p] et on considère une mat comme un tableau 1d d'ou n*n
	if (nb_lignesB == nb_colonnesA)
	{
		for (int i = 0; i < nb_lignesA; i++)
		{
			for (int j = 0; j < nb_colonnesB; j++)
			{
				C[j + i * nb_colonnesA] = 0;
				for (int k = 0; k < nb_lignesB; k++)
				{
					C[j + (i * nb_colonnesA)] += (A[i * nb_colonnesA + k] * B[(k * nb_colonnesB) + j]);
					//std::cout << " je multiplie " << A[i * nb_colonnesA + k] << " * " << B[(k * nb_colonnesB) + j] << std::endl;
				}
				//std::cout << std::endl;
			}
		}
	}
}

void soustraction_2_matrices(float *A, float *B, float *C, int n)
{
	for (int i = 0; i < n * n; i++)
	{
		for (int j = 0; j < n * n; j++)
		{
			C[i + j * n] = -1;
			for (int k = 0; k < n * n; k++)
			{
				C[i + j * n] = A[i + k * n] - B[k + (j * n)];
			}
		}
	}
}

void transpose_matrice_naive(float *A, float *B, int n)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			B[(n * j) + i] = A[(n * i) + j];
}

void transpose_matrice(float *A, float *B, int n)
{
	transpose_matrice(A, B, n, n);
}

void transpose_matrice(float *A, float *B, int n, int m)
{

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			B[(n * i) + j] = A[(m * j) + i];
		}
	}
}

int compter_inegalites(float *h_A, float *h_B, int n) // n c'est le côté de la mat
{
	int compteur = 0;

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (h_A[i * n + j] != h_B[i * n + j])
			{
				compteur++;
			}
		}
	}
	return compteur;
}

bool egalite_2_matrices(float *h_A, float *h_B, int n) // n c'est le côté de la mat
{
	return (compter_inegalites(h_A, h_B, n) == 0);
}

void free_cpu(float *h_A)
{
	if (h_A)
		delete[] h_A;
}

void free_gpu(float *h_A)
{
	if (h_A)
		cudaFree(h_A);
}

void affichage_performance(float t_ms, float octets_echanges)
{
	printf("Temps d'exécution du Kernel : %e (ms)\n", t_ms);
	printf("Bande passante GPU: %e GO/s\n", octets_echanges / t_ms);
	
}

void affichage_resultats_du_kernel(float *matrice1, float *matrice2, int taille_des_matrices_carree, float temps_en_milisecondes, float octets_echanges, bool affiche)
{
	affichage_performance(temps_en_milisecondes, octets_echanges);

	if (affiche == true)
	{
		std::cout << " entree : " << std::endl;
		afficher_matrice(matrice1, taille_des_matrices_carree);

		std::cout << " resultat : " << std::endl;
		afficher_matrice(matrice2, taille_des_matrices_carree);
	}
}

void affichage_resultats_du_kernel(float *matrice1, float *matrice2, float *matrice_res, int taille_des_matrices_carree, float temps_en_milisecondes, float octets_echanges, bool affiche)
{
	affichage_performance(temps_en_milisecondes, octets_echanges);

	if (affiche == true)
	{
		std::cout << " entree 1 : " << std::endl;
		afficher_matrice(matrice1, taille_des_matrices_carree);

		std::cout << " entree 2 : " << std::endl;
		afficher_matrice(matrice2, taille_des_matrices_carree);
		std::cout << " resultat : " << std::endl;
		afficher_matrice(matrice_res, taille_des_matrices_carree);
	}
}