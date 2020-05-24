/* Include standard  C/C++ */
#include <iostream>

/* support pour le format PAM */
#include "pamalign.h"

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>



#define RADIUS 1


/* Définition de l'opérateur += pour le type ushort4 en ignorant 
la dernière composante */
inline __device__ void operator+=(ushort4 &a, const ushort4 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}


/* Définition de l'opérateur / entre les type ushort4 et ushort en ignorant 
la dernière composante */
inline __device__ ushort4 operator/(const ushort4 &a, const ushort b)
{
    return make_ushort4(a.x / b, a.y / b, a.z / b, 0);
}


// Kernel de traitement d'image
__global__ void filtre_device(ushort4  *dst, const ushort4 *src,
	const uint width, const uint height)
{
	// A COMPLETER
}


int main(int argc, char **argv) {
	const char imgname[] = "image.pam";
	const char imgsave[] = "saved.pam";
	const char refname[] = "reference.pam";

	int pam_status;
	imgInfo img, ref;


	/* Chargement de l'image source */
	if (pam_status = load_pam(imgname, &img)) {
		return pam_status;
	}

	printf("Image %s chargée (%u-canaux), %ux%ux%u\n",
		imgname, img.channels, img.width, img.height, img.depth);

	ushort4 *d_Src, *d_Dst;

	size_t memSize = img.data_size;

	// Allocation de la mémoire sur la GPU
	checkCudaErrors(cudaMalloc(&d_Src, memSize));
	checkCudaErrors(cudaMalloc(&d_Dst, memSize));

	// Copie de l'image dans la mémoire GPU
	checkCudaErrors(cudaMemcpy(d_Src, img.data, memSize, cudaMemcpyHostToDevice));

	dim3 blockSize;
	blockSize.x = 32;
	blockSize.y = 32;
	dim3 gridSize;
	gridSize.x = ceil((float)img.width/blockSize.x);
	gridSize.y = ceil((float)img.height/blockSize.y);

	// Lancement du kernel
	filtre_device<<<gridSize, blockSize>>>(d_Dst, d_Src, img.width, img.height);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

	// Copie de l'image traitée 
	checkCudaErrors(cudaMemcpy(img.data, d_Dst, memSize, cudaMemcpyDeviceToHost));

	
	// Verification du résultat
	if (RADIUS == 1) {
		// Chargement de l'image de référence
		if (pam_status = load_pam(refname, &ref)) {
			return pam_status;
		}
		// Comparaison
		if (memcmp(img.data, ref.data, img.data_size)) {
			printf("Erreur de traitement\n");
			save_pam(imgsave, &img);
		}
		else
			printf("Traitement correct\n");
	}
	
	// Timing	
	float runtime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	const int nb = 5;
	for (int i=0;i <nb; i++)
		filtre_device<<<gridSize, blockSize>>>(d_Dst, d_Src, img.width, img.height);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&runtime, start, stop);
	runtime /= (float) nb;
	
	printf("Temps de traitement: %g (ms)\n", runtime);

	cudaEventDestroy(stop);
	cudaEventDestroy(start);
	cudaFree(d_Dst);
	cudaFree(d_Src);

	/* Sauvegarde de l'image traitée */
	if (pam_status = save_pam(imgsave, &img)) {
		return pam_status;
	}
	printf("Image %s sauvée (%u-canaux), %ux%ux%u\n",
		imgsave, img.channels, img.width, img.height, img.depth);

}
