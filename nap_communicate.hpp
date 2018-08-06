#ifndef NAPCOMM_COMMUNICATE_HPP
#define NAPCOMM_COMMUNICATE_HPP

#include <mpi.h>
#include "nap_comm_struct.hpp"

template <typename T>
struct NAPCommData{
    T* buf;
    T* global_buffer;
    MPI_Request* global_requests;
    MPI_Datatype datatype;
    int tag;

    NAPCommData()
    {
        buf = NULL;
        global_buffer = NULL;
        global_requests = NULL;
    }

    ~NAPCommData()
    {
        delete[] global_buffer;
        delete[] global_requests;
    }
};

struct NAPData{
    void* send_data;
    void* recv_data;
    int tag;

    NAPData()
    {
        send_data = NULL;
        recv_data = NULL;
    }
};

// Forward Declarations
template <typename T>
void MPI_intra_comm(comm_pkg* comm, T* send_data, T** recv_data,
        int tag, MPI_Comm local_comm, MPI_Datatype send_type, 
        MPI_Datatype recv_type);
template <typename T>
void MPI_inter_send(comm_pkg* comm, T* send_data,
        int tag, MPI_Comm mpi_comm, MPI_Datatype datatype,
        MPI_Request** send_request_ptr, T** send_buffer_ptr);
template <typename T>
void MPI_inter_recv(comm_pkg* comm,
        int tag, MPI_Comm mpi_comm, MPI_Datatype datatype,
        MPI_Request** recv_request_ptr, T** recv_buffer_ptr);
void MPI_inter_waitall(comm_pkg* comm, MPI_Request* send_requests, 
        MPI_Request* recv_requests);
template <typename T>
void MPI_intra_recv_map(comm_pkg* comm, T* intra_recv_data,
        T* inter_recv_data);


// Main Methods
template <typename T>
void MPI_INAPsend(T* buf, NAPComm* nap_comm,
        MPI_Datatype datatype, int tag,
        MPI_Comm comm, NAPData* nap_data)
{
    NAPCommData<T>* nap_send_data = new NAPCommData<T>();
    nap_send_data->buf = buf;
    nap_send_data->datatype = datatype;
    nap_send_data->tag = tag;

    int local_S_tag = nap_send_data->tag + 1;
    T* local_L_recv_data = NULL;
    T* local_S_recv_data = NULL;
    T* global_send_buffer = NULL;
    MPI_Request* global_send_requests = NULL;

    // Initial intra-node redistribution (step 1 in nap comm)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (int i = 0; i < nap_comm->local_S_comm->send_data->num_msgs; i++)   
    {
        int size = nap_comm->local_S_comm->send_data->indptr[i+1] 
            - nap_comm->local_S_comm->recv_data->indptr[i];
        if (size <= 0) printf("Rank %d: Size of msg %d is %d\n", rank, i, size);
    }
    MPI_intra_comm(nap_comm->local_S_comm, buf, &local_S_recv_data,
            local_S_tag, nap_comm->topo_info->local_comm, datatype, datatype);

    // Initialize Isends for inter-node step (step 2 in nap comm)
    MPI_inter_send(nap_comm->global_comm, local_S_recv_data, tag,
            comm, datatype, &global_send_requests, &global_send_buffer);

    // Store global_send_requests and global_send_buffer, as to not free data
    // before sends are finished
    nap_send_data->global_requests = global_send_requests;
    nap_send_data->global_buffer = global_send_buffer;
    nap_data->send_data = nap_send_data;

    delete[] local_S_recv_data;
}

template <typename T>
void MPI_INAPrecv(T* buf, NAPComm* nap_comm, 
        MPI_Datatype datatype, int tag, 
        MPI_Comm comm, NAPData* nap_data)
{
    NAPCommData<T>* nap_recv_data = new NAPCommData<T>();
    nap_recv_data->buf = buf;
    nap_recv_data->datatype = datatype;
    nap_recv_data->tag = tag;
    MPI_Request* global_recv_requests = NULL;
    T* global_recv_buffer = NULL;
 
    // Initialize Irecvs for inter-node step (step 2 in nap comm)
    MPI_inter_recv(nap_comm->global_comm, tag, comm, datatype,
            &global_recv_requests, &global_recv_buffer);

    // Store global_recv_requests and global_recv_buffer, as to not free data
    // before recvs are finished
    nap_recv_data->global_requests = global_recv_requests;
    nap_recv_data->global_buffer = global_recv_buffer;
    nap_data->recv_data = nap_recv_data;
}

template <typename T, typename U>
void MPI_NAPwait(NAPComm* nap_comm, NAPData* nap_data)
{
    NAPCommData<T>* nap_send_data = (NAPCommData<T>*) nap_data->send_data;
    NAPCommData<U>* nap_recv_data = (NAPCommData<U>*) nap_data->recv_data;

    U* local_R_recv_data = NULL;
    U* local_L_recv_data = NULL;
    T* global_send_buffer = nap_send_data->global_buffer;
    U* global_recv_buffer = nap_recv_data->global_buffer;
    T* send_buf = nap_send_data->buf;
    U* recv_buf = nap_recv_data->buf;
    MPI_Request* global_send_requests = nap_send_data->global_requests;
    MPI_Request* global_recv_requests = nap_recv_data->global_requests;
    MPI_Datatype send_type = nap_send_data->datatype;
    MPI_Datatype recv_type = nap_recv_data->datatype;

    int local_R_tag = nap_recv_data->tag + 2;
    int local_L_tag = nap_recv_data->tag + 3;

    MPI_inter_waitall(nap_comm->global_comm, global_send_requests,
            global_recv_requests);

    // Final intra-node redistribution (step 3 in nap comm)
    MPI_intra_comm(nap_comm->local_R_comm, global_recv_buffer, &local_R_recv_data,
            local_R_tag, nap_comm->topo_info->local_comm, recv_type, recv_type);

    // Fully intra-node communication
    MPI_intra_comm(nap_comm->local_L_comm, send_buf, &local_L_recv_data,
            local_L_tag, nap_comm->topo_info->local_comm, send_type, recv_type);


    // Map recv buffers from final intra node steps to correct locations in
    // recv_data
//    MPI_intra_recv_map(nap_comm->local_L_comm, local_L_recv_data, recv_buf);
//    MPI_intra_recv_map(nap_comm->local_R_comm, local_R_recv_data, recv_buf);

    nap_recv_data->buf = NULL;

    delete nap_send_data;
    delete nap_recv_data;
}




// Helper Methods
template <typename T>
void MPI_intra_comm(comm_pkg* comm, T* send_data, T** recv_data,
        int tag, MPI_Comm local_comm, MPI_Datatype send_type,
        MPI_Datatype recv_type)
{
    if (comm->send_data->num_msgs + comm->recv_data->num_msgs == 0) return;

    MPI_Request* send_requests;
    MPI_Request* recv_requests;
    T* send_buffer;
    T* recv_buffer;
    int idx, proc, start, end;

    send_requests = new MPI_Request[comm->send_data->num_msgs];
    recv_requests = new MPI_Request[comm->send_data->num_msgs];
    send_buffer = new T[comm->send_data->size_msgs];
    recv_buffer = new T[comm->recv_data->size_msgs];

    for (int i = 0; i < comm->send_data->num_msgs; i++)
    {
        proc = comm->send_data->procs[i];
        start = comm->send_data->indptr[i];
        end = comm->send_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = comm->send_data->indices[j];
            send_buffer[j] = send_data[idx];
        }
        MPI_Isend(&send_buffer[start], end - start, send_type, proc, tag, 
                local_comm, &send_requests[i]);
    }
    for (int i = 0; i < comm->recv_data->num_msgs; i++)
    {
        proc = comm->recv_data->procs[i];
        start = comm->recv_data->indptr[i];
        end = comm->send_data->indptr[i+1];
        MPI_Irecv(&recv_buffer[start], end - start, recv_type, proc, tag, 
                local_comm, &recv_requests[i]);
    }

    if (comm->send_data->num_msgs)
    {
        MPI_Waitall(comm->send_data->num_msgs, send_requests, MPI_STATUSES_IGNORE);
    }

    if (comm->recv_data->num_msgs)
    {
        MPI_Waitall(comm->recv_data->num_msgs, recv_requests, MPI_STATUSES_IGNORE);
    }

    *recv_data = recv_buffer;
    delete[] send_requests;
    delete[] recv_requests;
    delete[] send_buffer;
}

template <typename T>
void MPI_inter_send(comm_pkg* comm, T* send_data,
        int tag, MPI_Comm mpi_comm, MPI_Datatype mpi_type,
        MPI_Request** send_request_ptr, T** send_buffer_ptr)
{
    MPI_Request* send_requests;
    T* send_buffer;
    int idx, proc, start, end;

    send_requests = new MPI_Request[comm->send_data->num_msgs];
    send_buffer = new T[comm->send_data->size_msgs];

    for (int i = 0; i < comm->send_data->num_msgs; i++)
    {
        proc = comm->send_data->procs[i];
        start = comm->send_data->indptr[i];
        end = comm->send_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = comm->send_data->indices[j];
            send_buffer[j] = send_data[idx];
        }
        MPI_Isend(&send_buffer[start], end - start, mpi_type, proc, tag, 
                mpi_comm, &send_requests[i]);
    }

    *send_buffer_ptr = send_buffer;
    *send_request_ptr = send_requests;
}

template <typename T>
void MPI_inter_recv(comm_pkg* comm,
        int tag, MPI_Comm mpi_comm, MPI_Datatype mpi_type,
        MPI_Request** recv_request_ptr, T** recv_buffer_ptr)
{
    MPI_Request* recv_requests;
    T* recv_buffer;
    int proc, start, end;

    recv_requests = new MPI_Request[comm->recv_data->num_msgs];
    recv_buffer = new T[comm->recv_data->size_msgs];

    for (int i = 0; i < comm->recv_data->num_msgs; i++)
    {
        proc = comm->recv_data->procs[i];
        start = comm->recv_data->indptr[i];
        end = comm->recv_data->indptr[i+1];
        MPI_Isend(&recv_buffer[start], end - start, mpi_type, proc, tag, 
                mpi_comm, &recv_requests[i]);
    }

    *recv_buffer_ptr = recv_buffer;
    *recv_request_ptr = recv_requests;
}

void MPI_inter_waitall(comm_pkg* comm, MPI_Request* send_requests, 
        MPI_Request* recv_requests)
{
    if (comm->send_data->num_msgs)
    {
        MPI_Waitall(comm->send_data->num_msgs, send_requests, MPI_STATUSES_IGNORE);
    }

    if (comm->recv_data->num_msgs)
    {
        MPI_Waitall(comm->recv_data->num_msgs, recv_requests, MPI_STATUSES_IGNORE);
    }
}

template <typename T>
void MPI_intra_recv_map(comm_pkg* comm, T* intra_recv_data, T* inter_recv_data)
{
    int idx;

    for (int i = 0; i < comm->recv_data->size_msgs; i++)
    {
        idx = comm->recv_data->indices[i];
        inter_recv_data[idx] = intra_recv_data[i];
    }
}

#endif
