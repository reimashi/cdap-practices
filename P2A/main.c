#include <stdio.h>
#include <string.h>
#include <mpi/mpi.h>
#include <stdlib.h>

/* Numero de procesos */
#define NP 3

/* Numero de filas */
#define L 3

/* Rellena una matriz con numeros aleatorios entre 0 y 9 */
void fillMatrix(int* matrix, int columns, int rows) {
    time_t t;
    srand((unsigned) time(&t));

    for (int x = 0; x < columns; ++x) {
        for (int y = 0; y < rows; ++y) {
            matrix[x][y] = rand() % 10;
        }
    }
}

/* Imprime una matriz por pantalla */
void printMatrix(int* matrix, int columns, int rows) {
    for (int x = 0; x < columns; ++x) {
        for (int y = 0; y < rows; ++y) {
            printf(" %d", matrix[x][y]);
        }
        printf("\n");
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
            int matrix[columnCount][L];
            fillMatrix(matrix, columnCount, L);

            // Cremaos la matriz de resultados
            int resultColumnCount = processCount - 1;
            int resultMatrix[resultColumnCount][L];

            // Mostramos la matriz por pantalla
            printf("El proceso %d está enviando la matriz a los %d procesos\n", selfRank, processCount - 1);
            printMatrix(matrix, columnCount, L);

            // Por cada proceso de calculo
            for (int rank = 1; rank < processCount; ++rank) {
                // Enviamos la matriz correspondiente
                for (int i = 0; i < L; ++i) {
                    MPI_Send(matrix[L * (rank - 1)][i], 1, tColumn, rank, 0, MPI_COMM_WORLD);
                }

                // Esperamos la recepcion del calculo
                MPI_Recv(&resultMatrix[rank - 1], 1, tColumn, rank, 0, MPI_COMM_WORLD, &status);
            }

            // Mostamos la matriz con el resultado por pantalla
            printf("\nEl proceso %d recibió el resultado de las operaciones: \n", selfRank);
            printMatrix(resultMatrix, resultColumnCount, L);
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
                MPI_Recv(&matrix[0][i], 1, tColumn, 0, 0, MPI_COMM_WORLD, &status);
                
                // Sumamos todos sus elementos
                for (int j = 0; j < L; j++) {
                    resultMatrix[i] = matrix[j][i] + resultMatrix[i];
                }
            }
            
            // Enviamos el resultado del calculo
            MPI_Send(&resultMatrix, 1, tColumn, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
}