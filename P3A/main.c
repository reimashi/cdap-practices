#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi/mpi.h>

#define NFil 4
#define NCol 4
#define ColdTemp 20.0
#define HotTemp 80.0
#define NumIter 100
#define Coeff 0.5

MPI_Comm commCart;

// Obtiene el número de vecinos de un nodo
int getNumVec(MPI_Comm comCartesiano) {
    int nV = 0, src, dst;

    //Vecinos arriba y abajo
    MPI_Cart_shift(comCartesiano, 0, 1, &src, &dst);
    if (src >= 0) {
        nV = nV + 1;
    }
    if (dst >= 0) {
        nV = nV + 1;
    }

    //Vecinos hacia los lados
    MPI_Cart_shift(comCartesiano, 1, 1, &src, &dst);
    if (src >= 0) {
        nV = nV + 1;
    }

    if (dst >= 0) {
        nV = nV + 1;
    }

    return nV;
}

// Obtiene la lista de los vecinos de un nodo
void getVec(int *vec, MPI_Comm comCartesiano) {
    int src, dst;

    MPI_Cart_shift(comCartesiano, 0, 1, &src, &dst);
    vec[0] = src;
    vec[1] = dst;

    MPI_Cart_shift(comCartesiano, 1, 1, &src, &dst);
    vec[2] = src;
    vec[3] = dst;
}

void main(int argc, char* argv[]) {
    int nDims = 2, dims[2] = {NFil, NCol};
    int periods[2] = {0, 0};
    MPI_Status status;


    MPI_Init(&argc, &argv);

    int myRank;
    int numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    // Comprobación de número de procesos correcto
    if (numProcs != NFil * NCol) {
        if (myRank == 0) printf("El numero de procesos deberia ser %d\n", NFil * NCol);
    } else {
        // Creación del comunicador cartesiano
        MPI_Comm comCartesiano;
        MPI_Cart_create(MPI_COMM_WORLD, nDims, dims, periods, 0, &comCartesiano);

        // Obtenemos los vecinos
        int numVec, vec[4];
        numVec = getNumVec(comCartesiano);
        getVec(vec, comCartesiano);

        // Información de la topología cartesiana
        printf("Soy el nodo %d y tengo %d vecinos: ", myRank, numVec);
        for (int i = 0; i < 4; i++) {
            if (vec[i] >= 0) {
                printf("%d ", vec[i]);
            }
        }
        printf("\n");

        // Valores iniciales de temperatura
        float myTemp = (myRank == 0) ? HotTemp : ColdTemp;

        float recvTemp, newTemp = 0, aux;
        for (int contIter = 1; contIter <= NumIter; ++contIter) {
            aux = 0;

            for (int i = 0; i < 4; i++) {
                // Si el ID se corresponde con un vecino
                if (vec[i] >= 0) {
                    // TODO: envío de información a los vecinos
                    MPI_Send(&myTemp, 1, MPI_FLOAT, vec[i], 0, comCartesiano);

                    // TODO: recepción de información de los vecinos y cálculo de la nueva temperatura en newTemp
                    MPI_Recv(&recvTemp, 1, MPI_FLOAT, vec[i], 0, comCartesiano, &status);
                    aux = aux + recvTemp;
                }
            }

            newTemp = (1 - Coeff) * myTemp + Coeff * (aux / numVec);

            // Actualización temperaturas
            if (myRank != 0) {
                myTemp = newTemp;
            }

            // Muestra información 
            if (myRank == numProcs - 1)
                if (contIter % 10 == 0)
                    printf("Iter %d - soy el nodo %d y mi temp. es %f\n", contIter, myRank, myTemp);
        }
    }

    MPI_Finalize();
    exit(0);
}