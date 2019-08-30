#include <cmath>
#include <mpi.h>


// Standard All-Reduce Recursive Doubling Algorithm
// At each step s of log(p) steps, exchange with following:
// if (rank/(2^s) % 2) == 0)
//     rank + (2^s)
// else
//     rank - (2^s)

int MPIX_Allreduce_RD(const int* sendbuf, int* recvbuf, int count, 
        MPI_Datatype datatype, MPI_Comm comm);
int MPIX_Allreduce_RD(const double* sendbuf, double* recvbuf, int count, 
        MPI_Datatype datatype, MPI_Comm comm);
int MPIX_Allreduce_SMP(const int* sendbuf, int* recvbuf, int count,
        MPI_Datatype datatype, MPI_Comm local_comm, MPI_Comm master_comm);
int MPIX_Allreduce_SMP(const double* sendbuf, double* recvbuf, int count,
        MPI_Datatype datatype, MPI_Comm local_comm, MPI_Comm master_comm);
int MPIX_Allreduce_NAP(const int* sendbuf, int* recvbuf, int count, 
        MPI_Datatype datatype, MPI_Comm local_comm, 
        int ppn, int* step_sizes, int n_steps);
int MPIX_Allreduce_NAP(const double* sendbuf, double* recvbuf, int count, 
        MPI_Datatype datatype, MPI_Comm local_comm, 
        int ppn, int* step_sizes, int n_steps);


