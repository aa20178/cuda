#include "matrice.h"

// Code GPU

int main(int argc, char **argv)
{
	int n(0);

	float t1[] = {
		2,
		1,
		0,
	};
	float t2[] = {1, 6, 0, 2, 1, 3};
	float t3[] = {0, 0, 0, 0, 0, 0};

	float res[] = {0, 0};

	//multiplier_matrice(t1, t2, res, 1, 3, 3, 2);
	afficher_matrice(t2, 2, 3);
	transpose_matrice(t2, t3,2, 3);
	transpose_matrice(t3, t2,3, 2);
	afficher_matrice(t2, 2, 3);

	//afficher_matrice(produit3, 3);

	return 0;
}
