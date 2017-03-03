#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#define TAG_MATRIZ 0
#define TAG_VECTOR 0
#define TAG_RESULT 0

void main(int argc, char* argv[]) {
	if(argc >= 2){
		//proceso padre
		int numProcs;
		int hijos= atoi(argv[1]);
		int matriz[hijos][hijos];
		int matriz2[hijos][hijos];
		int vector[hijos];

		MPI_Comm intercom,intracom;

		MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD,&numProcs);
	  
	   MPI_Win res_ventana;

		if(numProcs!=1){
			printf("numero de procesos incorrecto\n");
			MPI_Finalize;
			exit(0);
		}

		printf("numero de hijos: %d\n", hijos);

		for (int i=0; i<hijos; i++)
			for (int j=0; j<hijos; j++) {
				matriz[i][j]=i+j;
				matriz2[i][j]=+i+j;
		}

		printf("Matriz:\n");
		 for (i=0; i<hijos; i++) {
			printf(" ");
			for (j=0; j<hijos; j++){
				printf("%3d",matriz[i][j]);
				printf("\n");
			}
		}

		MPI_Comm_spawn("p4",MPI_ARGV_NULL,hijos,MPI_INFO_NULL,0,MPI_COMM_WORLD,&intercom,MPI_ERRCODES_IGNORE);
		MPI_Intercomm_merge(intercom, 0, &intracom);

		MPI_Win_create(&vector[0], hijos*sizeof(int), 1, MPI_INFO_NULL, intracom, &res_ventana);
		
		for (i=0; i<hijos; i++) {
		  MPI_Send(&matriz[i][0],hijos,MPI_INT,i,TAG_MATRIZ,intercom);
	   }
	   
	   MPI_Win_fence(0, res_ventana);
	   printf("Resultado:\n");
	   MPI_Win_fence(0, res_ventana);
	   for (i=0; i<hijos; i++){
		  printf("  %3d\n",vector[i]);
		  }
	   
	   MPI_Win_free(&res_ventana);
	   
	   MPI_Finalize();
   }
   else{
		//procesos hijos
		int result;
		int i,j,hijos;
		 MPI_Win res_win;
		MPI_Comm intercom;

	   MPI_Init(&argc, &argv);

	   MPI_Comm_get_parent(&intercom);

	    MPI_Intercomm_merge(intercom, 0, &intracom);

		MPI_Comm_size(intercom, &hijos);
		int fila[hijos], vector[hijos];

  
	   MPI_Win_create(MPI_BOTTOM, 0, 1, MPI_INFO_NULL, intracom, &res_win);

	   MPI_Recv(fila,hijos,MPI_INT,MPI_ANY_SOURCE,TAG_MATRIZ,intercom,MPI_STATUS_IGNORE);
	   MPI_Recv(vector,hijos,MPI_INT,MPI_ANY_SOURCE,TAG_VECTOR,intercom,MPI_STATUS_IGNORE);
		

	   //C疝culo
	   result=0;
	   for (i=0;i<6;i++){
		  result += vector[i]*fila[i];
		}
	   
	   
	   //MPI_Send(&result,1,MPI_INT,0,TAG_RESULT,intercom);

	   MPI_Win_fence(0, res_win);
	
		//Se envian los resultados
	   MPI_Win_fence(0, res_win);
   
   MPI_Win_free(&res_win);
		
		
		MPI_Finalize();
   }

}