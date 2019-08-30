#include "nap_allreduce.hpp"
#include <cstring>

// Standard All-Reduce Recursive Doubling Algorithm
// At each step s of log(p) steps, exchange with following:
// if (rank/(2^s) % 2) == 0)
//     rank + (2^s)
// else
//     rank - (2^s)

template <typename T> 
int MPIX_Reduce_helper(T* recvbuf, T* tmp_buf, 
        int count, MPI_Datatype datatype, MPI_Comm comm)
{
    if (count <= 0) return 0;

    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    int n_steps = log2(num_procs);
    int tag = 9382;
    MPI_Status recv_status;
    int proc;
   
    int split = 1;
    for (int i = 0; i < n_steps; i++)
    {
        if (rank % split == 0)
        {
            if (((rank / split) % 2) == 0)
            {
                proc = rank + split;
                MPI_Recv(tmp_buf, count, datatype, proc, tag, comm, &recv_status);
                for (int i = 0; i < count; i++)
                    recvbuf[i] += tmp_buf[i];
            }
            else 
            {
                proc = rank - split;
                MPI_Send(recvbuf, count, datatype, proc, tag, comm);
            }
        }
        split *= 2;
        tag++;
    }

    return 0;
}

template <typename T>
int MPIX_Bcast_helper(T* buffer, int count, MPI_Datatype datatype, MPI_Comm comm)
{
    if (count <= 0) return 0;
    
    int rank, num_procs; 
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    int n_steps = log2(num_procs);
    int tag = 12945;
    MPI_Status recv_status;
    int proc;

    int split = num_procs / 2;
    for (int i = 0; i < n_steps; i++)
    {
        if (rank % split == 0)
        {
            if (((rank / split) % 2) == 0)
            {
                proc = rank + split;
                MPI_Send(buffer, count, datatype, proc, tag, comm);
            }
            else
            {
                proc = rank - split;
                MPI_Recv(buffer, count, datatype, proc, tag, comm, &recv_status);
            }
        }
        split /= 2;
        tag++;
    }          
    
    return 0;
}

template <typename T>
int MPIX_Allreduce_RD_helper(T* recvbuf, T* tmp_buf, 
        int count, MPI_Datatype datatype, MPI_Comm comm)
{
    if (count <= 0) return 0;

    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    int n_steps = log2(num_procs);
    int tag = 1235;
    MPI_Status recv_status;
    int proc;

    int split = 1;
    for (int i = 0; i < n_steps; i++)
    {
        if (((rank / split) % 2) == 0)
        {
            proc = rank + split;
        }
        else
        {
            proc = rank - split;
        }

        if (rank < proc)
        {
            MPI_Send(recvbuf, count, datatype, proc, tag, comm);
            MPI_Recv(tmp_buf, count, datatype, proc, tag, comm, &recv_status);
        }
        else
        {
            MPI_Recv(tmp_buf, count, datatype, proc, tag, comm, &recv_status);
            MPI_Send(recvbuf, count, datatype, proc, tag, comm);
        }

        for (int i = 0; i < count; i++)
            recvbuf[i] += tmp_buf[i];

        split *= 2;
        tag++;
    }

    return 0;
}

template <typename T>
int MPIX_Allreduce_SMP_helper(T* recvbuf, T* tmp_buf, int count,
        MPI_Datatype datatype, MPI_Comm local_comm, MPI_Comm master_comm,
        MPI_Comm comm = MPI_COMM_WORLD)
{
    if (count <= 0) return 0;

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);


    // First Reduce on node to local master rank (localrank 0)
    MPIX_Reduce_helper(recvbuf, tmp_buf, count, datatype, local_comm);

    // Then Allreduce among master ranks
    if (local_rank == 0) 
        MPIX_Allreduce_RD_helper(recvbuf, tmp_buf, count, datatype, master_comm);
        //MPI_Allreduce(MPI_IN_PLACE, recvbuf, count, datatype, MPI_SUM, master_comm);

    // Finally Broadcast result on node
    MPIX_Bcast_helper(recvbuf, count, datatype, local_comm);

    return 0;
}

template <typename T>
int MPIX_Allreduce_NAP_helper(T* recvbuf, T* tmp_buf, 
        int count, MPI_Datatype datatype, MPI_Comm local_comm, 
        int ppn, int* step_sizes, int n_steps,
        MPI_Comm comm = MPI_COMM_WORLD)
{
    if (count <= 0) return 0;

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_nodes = num_procs / ppn;

    MPI_Request req[2];
    int tag = 6543;
    MPI_Status recv_status;

    T* local_tmp_buf = new T[count];

    // Allreduce initial data local to node
    MPIX_Allreduce_RD_helper(recvbuf, tmp_buf, count, datatype, local_comm);
    if (n_steps == 0) return 0;

    int split = ppn;
    int next_split = split*step_sizes[0];
    int local_sum = 0;
    int local_rank = rank % ppn; // dest_group_idx (in my outer group)
    int proc;
    for (int i = 0; i < n_steps; i++)
    {
        int group_start = (rank / (next_split)) * next_split; 
        int group = (rank-group_start) / split; // local group == dest_local_rank
        proc = group_start + local_sum + (local_rank * split) + group; 

        if (rank == proc || local_rank >= step_sizes[i])
        {
            for (int i = 0; i < count; i++) tmp_buf[i] = 0;
        }
        else
        {
            if (rank < proc)
            {
                MPI_Send(recvbuf, count, datatype, proc, tag, comm);
                MPI_Recv(tmp_buf, count, datatype, proc, tag, comm, &recv_status);
            }
            else
            {
                MPI_Recv(tmp_buf, count, datatype, proc, tag, comm, &recv_status);
                MPI_Send(recvbuf, count, datatype, proc, tag, comm);
            }
        }

        MPIX_Allreduce_RD_helper(tmp_buf, local_tmp_buf, count, datatype, local_comm);

        for (int i = 0; i < count; i++)
            recvbuf[i] += tmp_buf[i];

        local_sum += (group*split);
        split = next_split;
        if (i < n_steps-1)
            next_split *= step_sizes[i+1];
        tag++;
    }

    delete[] local_tmp_buf;

    return 0;
}



int MPIX_Allreduce_RD(const int* sendbuf, int* recvbuf, int count, 
        MPI_Datatype datatype, MPI_Comm comm)
{
    int* tmp_buf = new int[count];
    std::memcpy(recvbuf, sendbuf, count*sizeof(int));
    MPIX_Allreduce_RD_helper(recvbuf, tmp_buf, count, datatype, comm);
    delete[] tmp_buf;
}
int MPIX_Allreduce_RD(const double* sendbuf, double* recvbuf, int count, 
        MPI_Datatype datatype, MPI_Comm comm)
{
    double* tmp_buf = new double[count];
    std::memcpy(recvbuf, sendbuf, count*sizeof(double));
    MPIX_Allreduce_RD_helper(recvbuf, tmp_buf, count, datatype, comm);
    delete[] tmp_buf;
}
int MPIX_Allreduce_SMP(const int* sendbuf, int* recvbuf, int count,
        MPI_Datatype datatype, MPI_Comm local_comm, MPI_Comm master_comm)
{
    int* tmp_buf = new int[count];
    std::memcpy(recvbuf, sendbuf, count*sizeof(int));
    MPIX_Allreduce_SMP_helper(recvbuf, tmp_buf, count, datatype,
        local_comm, master_comm);
    delete[] tmp_buf;
}
int MPIX_Allreduce_SMP(const double* sendbuf, double* recvbuf, int count,
        MPI_Datatype datatype, MPI_Comm local_comm, MPI_Comm master_comm)
{
    double* tmp_buf = new double[count];
    std::memcpy(recvbuf, sendbuf, count*sizeof(double));
    MPIX_Allreduce_SMP_helper(recvbuf, tmp_buf, count, datatype,
        local_comm, master_comm);
    delete[] tmp_buf;
}
int MPIX_Allreduce_NAP(const int* sendbuf, int* recvbuf, int count, 
        MPI_Datatype datatype, MPI_Comm local_comm, 
        int ppn, int* step_sizes, int n_steps)
{
    int* tmp_buf = new int[count];
    std::memcpy(recvbuf, sendbuf, count*sizeof(int));
    MPIX_Allreduce_NAP_helper(recvbuf, tmp_buf, count, datatype, 
        local_comm, ppn, step_sizes, n_steps);
    delete[] tmp_buf;
}
int MPIX_Allreduce_NAP(const double* sendbuf, double* recvbuf, int count, 
        MPI_Datatype datatype, MPI_Comm local_comm, 
        int ppn, int* step_sizes, int n_steps)
{
    double* tmp_buf = new double[count];
    std::memcpy(recvbuf, sendbuf, count*sizeof(double));
    MPIX_Allreduce_NAP_helper(recvbuf, tmp_buf, count, datatype, 
        local_comm, ppn, step_sizes, n_steps);
    delete[] tmp_buf;
}





