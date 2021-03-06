#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <mpi/mpi.h>

/* Numero de procesos */
#define NP 3

/* Numero de filas */
#define L 3

/* Rellena un vector con numeros aleatorios entre 0 y 9 */
void fillVector(int* vector, int elements) {
    static int randInit = 0;
    
    if (!randInit) {
        time_t tim;
        time(&tim);
        srand((unsigned) tim);
        randInit = 1;
    }

    for (int x = 0; x < elements; ++x) {
        vector[x] = rand() % 10;
    }
}

/* Imprime un vector por pantalla */
void printVector(int* vector, int elements) {
    for (int x = 0; x < elements; ++x) {
        printf(" %d", vector[x]);
    }
    printf("\n");
}

/* Imprime un vector por pantalla en vertical */
void printVectorVertical(int* vector, int elements) {
    for (int x = 0; x < elements; ++x) {
        printf(" %d\n", vector[x]);
    }
}

/* Punto de entrada del programa */
int main(int argc, char* argv[]) {
    const int root = 0;

    // Inicializamos la comunicación MPI
    int processCount, selfRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &selfRank);

    if (processCount != NP) {
        if (selfRank == 0) {
            printf("El programa requiere exactamente %d  procesos.\n", NP);
        }
    } else {
        MPI_Status status;

        // Tipo de dato derivado, columna
        MPI_Datatype tColumn;
        MPI_Type_vector(L, L, L, MPI_INT, &tColumn);
        MPI_Type_commit(&tColumn);

        // Si somos el proceso 0
        if (selfRank == 0) {
            // Creamos la matriz de origen
            int columnCount = L * (processCount - 1);
            int matrix[L][columnCount];
            
            for (int i = 0; i < L; ++i) fillVector(matrix[i], columnCount);

            // Cremaos la matriz de resultados de cada hilo
            int resultColumnCount = processCount - 1;
            int resultMatrix[resultColumnCount][L];
            
            // Creamos el vector de resultados global
            int resultVector[L];
            for (int i = 0; i < L; ++i) resultVector[i] = 0;

            // Mostramos la matriz por pantalla
            printf("El proceso %d está enviando la matriz a los %d procesos\n", selfRank, processCount - 1);
            for (int i = 0; i < L; ++i) printVector(matrix[i], columnCount);
            
            // Por cada proceso de calculo
            for (int rank = 1; rank < processCount; ++rank) {
                // Enviamos la matriz correspondiente
                for (int i = 0; i < L; ++i) {
                    int *toSend = matrix[i];
                    toSend += L * (rank - 1);
                    MPI_Send(toSend, 1, tColumn, rank, 0, MPI_COMM_WORLD);
                }

                // Esperamos la recepcion del calculo
                MPI_Recv(&resultMatrix[rank - 1], 1, tColumn, rank, 0, MPI_COMM_WORLD, &status);
                
                // Mostramos el resultado recibido de cada hilo por pantalla
                printf("\nEl proceso %d recibió el resultado de las operaciones de %d: \n", selfRank, rank);
                printVectorVertical(resultMatrix[rank - 1], L);
            }
            
            // Por cada proceso que genera un resultado
            for (int i = 0; i < L; i++) {
                for (int rank = 1; rank < processCount; ++rank) {
                    resultVector[i] += resultMatrix[rank - 1][i];
                }
            }

            // Mostamos la matriz con el resultado por pantalla
            printf("\nEl proceso %d calculó el resultado final: \n", selfRank);
            printVectorVertical(resultVector, L);
        }            
        // Si somos un proceso de calculo
        else {
            // Creamos la matriz de recepción
            int matrix[L][L]; // Revisar tamaño
            
            // Creamos la matriz de resultados y la inicializamos a 0
            int resultMatrix[L];
            for (int i = 0; i < L; i++) { resultMatrix[i] = 0; }
            
            // Por cada columna
            for (int i = 0; i < L; i++) {
                // Recibimos la fila
                MPI_Recv(&matrix[i], 1, tColumn, 0, 0, MPI_COMM_WORLD, &status);
                
                // Sumamos todos sus elementos
                for (int j = 0; j < L; j++) {
                    resultMatrix[i] = matrix[i][j] + resultMatrix[i];
                }
            }
            
            // Enviamos el resultado del calculo
            MPI_Send(&resultMatrix, 1, tColumn, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    exit(0);
}