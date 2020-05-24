#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define DIM_PORTION 32
#define LIGNES_BLOC 8
// Code GPU

__global__ void copymat_device(const float *input, float *output, int n)
{
	int x_matrice = blockIdx.x * blockDim.x + threadIdx.x;
	int y_matrice = blockIdx.y * blockDim.y + threadIdx.y;
	int largeur_matrice = blockDim.x * gridDim.x;
	int indice_lin =0;

	//__shared__ float s_data[DIM_PORTION];
	for (int j = 0; j < DIM_PORTION; j += LIGNES_BLOC)
	{
			int indice_lin = (largeur_matrice * (y_matrice+j)) + x_matrice; // addresse

		if (x_matrice < n && (y_matrice+j) < n) //j ou pas?
		{
			output[indice_lin] = input[indice_lin];
		}
		else
		{
			return;
		}
	}
}
// Code CPU
void afficher_matrice(float *A, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			std::cout << A[i * n + j] << "  ";
		}
		std::cout << std::endl;
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

int compter_occurences_de_difference(float *h_A, float *h_B, int n) // n c'est le côté de la mat
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

int main(int argc, char **argv)
{

	if ((argc != 2) || (atoi(argv[1]) < 1))
	{
		std::cout << " il faut un seul argument ! " << std::endl;
		exit(-1);
	}

	int n = atoi(argv[1]);
	size_t size = n * n * sizeof(float);

	// Matrices CPU
	float *h_A = nullptr, *h_B = nullptr;
	// Matrices GPU
	float *d_A = nullptr, *d_B = nullptr;

	// Allocatation des vecteurs dans la mémoire CPU
	h_A = new float[n * n];
	h_B = new float[n * n];

	// Allocation des vecteurs dans la mémoire GPU
	checkCudaErrors(cudaMalloc((void **)&d_A, size));
	checkCudaErrors(cudaMalloc((void **)&d_B, size));

	// Initialisation de la matrice A
	srand(time(NULL));
	genmat(h_A, n);

	// Copie de la matrice A dans la mémoire GPU
	checkCudaErrors(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

	// Definition de la taille des blocs et de la grille
	dim3 threadsPerBlock(DIM_PORTION, DIM_PORTION);
	dim3 numBlocks(ceil(n / (float)threadsPerBlock.x), ceil(n / (float)threadsPerBlock.x));

	copymat_device<<<numBlocks, threadsPerBlock>>>(d_A, d_B, n);
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Copie du résultat
	checkCudaErrors(cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost));
	printf("Erreur max: %e\n", verify(h_A, h_B, n));

	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	const int nb = 10;
	checkCudaErrors(cudaEventRecord(start, 0));
	for (int i = 0; i < nb; i++)
		copymat_device<<<numBlocks, threadsPerBlock>>>(d_A, d_B, n);
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	float t_ms;
	checkCudaErrors(cudaEventElapsedTime(&t_ms, start, stop));
	t_ms /= nb;
	t_ms /= 1000;
	float octets_echanges(2 * size / pow(10, 9));

	printf("Temps d'exécution du Kernel : %e (ms)\n", t_ms);
	printf("Bande passante GPU: %e GO/s\n", octets_echanges / t_ms);
	return 0;
}
