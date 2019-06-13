#ifndef NAPCOMM_COMMUNICATE_HPP
#define NAPCOMM_COMMUNICATE_HPP

#include <mpi.h>
#include "nap_comm_struct.hpp"

/******************************************
 ****
 **** Communication Structs
 ****
 ******************************************/
struct NAPCommData{
    void* buf;
    char* global_buffer;
    char* local_L_buffer;
    MPI_Datatype datatype;
    int tag;

    NAPCommData()
    {
        buf = NULL;
        global_buffer = NULL;
        local_L_buffer = NULL;
    }

    ~NAPCommData()
    {
        if (global_buffer) delete[] global_buffer;
        if (local_L_buffer) delete[] local_L_buffer;
    }
};

struct NAPData{
    NAPCommData* send_data;
    NAPCommData* recv_data;
    int tag;
    MPI_Comm mpi_comm;

    NAPData()
    {
        send_data = NULL;
        recv_data = NULL;
    }
};

/******************************************
 ****
 **** Forward Declarations
 ****
 ******************************************/
static void MPIX_step_comm(comm_pkg* comm, void* send_data, char** recv_data,
        int tag, MPI_Comm local_comm, MPI_Datatype send_type,
        MPI_Datatype recv_type, MPI_Request* send_requests,
        MPI_Request* recv_requests);
static void MPIX_step_send(comm_pkg* comm, void* send_data,
        int tag, MPI_Comm mpi_comm, MPI_Datatype datatype,
        MPI_Request* send_request, char** send_buffer_ptr);
static void MPIX_step_recv(comm_pkg* comm,
        int tag, MPI_Comm mpi_comm, MPI_Datatype datatype,
        MPI_Request* recv_request, char** recv_buffer_ptr);
static void MPIX_step_waitall(comm_pkg* comm, MPI_Request* send_requests,
        MPI_Request* recv_requests);
static void MPIX_intra_recv_map(comm_pkg* comm, char* intra_recv_data,
        void* inter_recv_data, MPI_Datatype datatype, MPI_Comm mpi_comm);


static char* MPIX_NAP_unpack(char* packed_buf, int size, MPI_Datatype datatype, MPI_Comm comm)
{
    int type_size, pack_size;
    int ctr = 0;

    MPI_Type_size(datatype, &type_size);
    char* unpacked_buf = new char[size*type_size];

    MPI_Pack_size(size, datatype, comm, &pack_size);
    MPI_Unpack(packed_buf, pack_size, &ctr, unpacked_buf, size, datatype, comm);

    return unpacked_buf;
}

/******************************************
 ****
 **** Main Methods
 ****
 ******************************************/

// Node-Aware Version of Isend
static void MPIX_INAPsend(void* buf, NAPComm* nap_comm,
        MPI_Datatype datatype, int tag,
        MPI_Comm comm, NAPData* nap_data)
{
    NAPCommData* nap_send_data = new NAPCommData();
    nap_data->mpi_comm = comm;
    nap_send_data->datatype = datatype;
    nap_send_data->tag = tag;

    int local_S_tag = tag + 1;
    int local_L_tag = tag + 3;
    int size, type_size, ctr;

    char* local_L_recv_data = NULL;
    char* local_S_recv_data = NULL;
    char* global_send_buffer = NULL;
    char* L_send_buffer = NULL;
    MPI_Request* send_requests = nap_comm->send_requests;
    MPI_Request* recv_requests = nap_comm->recv_requests;
    MPI_Request* L_send_requests = NULL;

    MPI_Type_size(datatype, &type_size);

    // Initial intra-node redistribution (step 1 in nap comm)
    MPIX_step_comm(nap_comm->local_S_comm, buf, &local_S_recv_data,
            local_S_tag, nap_comm->topo_info->local_comm, datatype, datatype,
            send_requests, recv_requests);

    if (nap_comm->local_S_comm->recv_data->size_msgs)
    {
        // Unpack previous recv into void*
        char* unpacked_buf = MPIX_NAP_unpack(local_S_recv_data, 
                nap_comm->local_S_comm->recv_data->size_msgs,
                datatype, comm);

        // Initialize Isends for inter-node step (step 2 in nap comm)
        MPIX_step_send(nap_comm->global_comm, local_S_recv_data, tag,
                comm, datatype, send_requests, &global_send_buffer);

        // Free void* buffer (packed into char* buf)
        delete[] unpacked_buf;
    }

    if (nap_comm->local_L_comm->send_data->num_msgs)
    {
        L_send_requests = &(send_requests[nap_comm->global_comm->send_data->num_msgs]);
        MPIX_step_send(nap_comm->local_L_comm, buf, local_L_tag, 
                nap_comm->topo_info->local_comm, datatype, L_send_requests, &L_send_buffer);
    }

    // Store global_send_requests and global_send_buffer, as to not free data
    // before sends are finished
    nap_send_data->global_buffer = global_send_buffer;
    nap_send_data->local_L_buffer = L_send_buffer;
    nap_data->send_data = nap_send_data;

    if (local_S_recv_data) delete[] local_S_recv_data;
}

// Node-Aware Version of Irecv
static void MPIX_INAPrecv(void* buf, NAPComm* nap_comm,
        MPI_Datatype datatype, int tag,
        MPI_Comm comm, NAPData* nap_data)
{
    NAPCommData* nap_recv_data = new NAPCommData();
    nap_data->mpi_comm = comm;
    nap_recv_data->tag = tag;
    nap_recv_data->buf = buf;
    nap_recv_data->datatype = datatype;
    MPI_Request* global_recv_requests = NULL;
    MPI_Request* recv_requests = nap_comm->recv_requests;
    MPI_Request* L_recv_requests = NULL;
    char* global_recv_buffer = NULL;
    char* local_L_buffer = NULL;

    int local_L_tag = tag + 3;

    // Initialize Irecvs for inter-node step (step 2 in nap comm)
    MPIX_step_recv(nap_comm->global_comm, tag, comm, datatype,
            recv_requests, &global_recv_buffer);

    if (nap_comm->local_L_comm->recv_data->num_msgs)
    {
        L_recv_requests = &(recv_requests[nap_comm->global_comm->recv_data->num_msgs]);
        MPIX_step_recv(nap_comm->local_L_comm, local_L_tag, nap_comm->topo_info->local_comm,
                datatype, L_recv_requests, &local_L_buffer);
    }

    // Store global_recv_requests and global_recv_buffer, as to not free data
    // before recvs are finished
    nap_recv_data->global_buffer = global_recv_buffer;
    nap_recv_data->local_L_buffer = local_L_buffer;
    nap_data->recv_data = nap_recv_data;
}

// Wait for Node-Aware Isends and Irecvs to complete
static void MPIX_NAPwait(NAPComm* nap_comm, NAPData* nap_data)
{
    NAPCommData* nap_send_data = nap_data->send_data;
    NAPCommData* nap_recv_data = nap_data->recv_data;
    MPI_Comm mpi_comm = nap_data->mpi_comm;

    char* local_R_recv_data = NULL;
    char* local_L_recv_data = nap_recv_data->local_L_buffer;
    char* global_send_buffer = nap_send_data->global_buffer;
    char* global_recv_buffer = nap_recv_data->global_buffer;
    MPI_Request* send_requests = nap_comm->send_requests;
    MPI_Request* recv_requests = nap_comm->recv_requests;
    MPI_Request* L_send_requests = NULL;
    MPI_Request* L_recv_requests = NULL;
    MPI_Datatype send_type = nap_send_data->datatype;
    MPI_Datatype recv_type = nap_recv_data->datatype;

    int local_R_tag = nap_recv_data->tag + 2;

    MPIX_step_waitall(nap_comm->global_comm, send_requests, recv_requests);
    if (nap_comm->local_L_comm->send_data->num_msgs)
        L_send_requests = &(send_requests[nap_comm->global_comm->send_data->num_msgs]);
    if (nap_comm->local_L_comm->recv_data->num_msgs)
        L_recv_requests = &(recv_requests[nap_comm->global_comm->recv_data->num_msgs]);
    MPIX_step_waitall(nap_comm->local_L_comm, L_send_requests, L_recv_requests);
    if (nap_send_data->local_L_buffer)
    {
        delete[] nap_send_data->local_L_buffer;
        nap_send_data->local_L_buffer = NULL;
    }

    // Map recv buffers from final intra node steps to correct locations in
    // recv_data
    local_L_recv_data = nap_recv_data->local_L_buffer;
    if (local_L_recv_data)
    {
        MPIX_intra_recv_map(nap_comm->local_L_comm, local_L_recv_data, nap_recv_data->buf,
                recv_type, nap_comm->topo_info->local_comm);
        delete[] local_L_recv_data;
        nap_recv_data->local_L_buffer = NULL;
    }

    // Final intra-node redistribution (step 3 in nap comm)
    // Unpack global_recv_buffer ot U[] (but don't know U here...)
    char* unpacked_buf = NULL;
    if (nap_comm->global_comm->recv_data->size_msgs)
    {
        // Unpack previous recv into void*
        unpacked_buf = MPIX_NAP_unpack(global_recv_buffer, 
                nap_comm->global_comm->recv_data->size_msgs,
                recv_type, mpi_comm);
    }

    // Initialize Isends for inter-node step (step 2 in nap comm)
    MPIX_step_comm(nap_comm->local_R_comm, global_recv_buffer, &local_R_recv_data,
            local_R_tag, nap_comm->topo_info->local_comm, recv_type, recv_type,
            send_requests, recv_requests);

    if (unpacked_buf)
        delete[] unpacked_buf;
    
    if (local_R_recv_data)
    {
        MPIX_intra_recv_map(nap_comm->local_R_comm, local_R_recv_data, nap_recv_data->buf,
                recv_type, nap_comm->topo_info->local_comm);
        delete[] local_R_recv_data;
    }
    nap_recv_data->buf = NULL;
    delete nap_send_data;
    delete nap_recv_data;
}


/******************************************
 ****
 **** Helper Methods
 ****
 ******************************************/

// Intra/Inter-Node Waitall
static void MPIX_step_waitall(comm_pkg* comm, MPI_Request* send_requests,
        MPI_Request* recv_requests)
{
    int flag;
    if (comm->send_data->num_msgs)
    {
        MPI_Waitall(comm->send_data->num_msgs, send_requests, MPI_STATUSES_IGNORE);
    }
    if (comm->recv_data->num_msgs)
    {
        MPI_Waitall(comm->recv_data->num_msgs, recv_requests, MPI_STATUSES_IGNORE);
    }
}


// Intra-Node Communication
static void MPIX_step_comm(comm_pkg* comm, void* send_data, char** recv_data,
        int tag, MPI_Comm local_comm, MPI_Datatype send_type, MPI_Datatype recv_type,
        MPI_Request* send_requests, MPI_Request* recv_requests)
{
    char* send_buffer = NULL;
    char* recv_buffer = NULL;

    MPIX_step_send(comm, send_data, tag, local_comm, send_type,
            send_requests, &send_buffer);
    MPIX_step_recv(comm, tag, local_comm, recv_type, recv_requests,
            &recv_buffer);
    MPIX_step_waitall(comm, send_requests, recv_requests); 

    if (send_buffer) delete[] send_buffer;
    if (recv_buffer) *recv_data = recv_buffer;
}

// Inter-node Isend
static void MPIX_step_send(comm_pkg* comm, void* send_data,
        int tag, MPI_Comm mpi_comm, MPI_Datatype mpi_type,
        MPI_Request* send_requests, char** send_buffer_ptr)
{
    int idx, proc, start, end;
    int size, type_size, addr;
    char* send_buffer = NULL;
    char* data = reinterpret_cast<char*>(send_data);
    MPI_Type_size(mpi_type, &type_size);
    if (comm->send_data->size_msgs)
    {
        MPI_Pack_size(comm->send_data->size_msgs, mpi_type, mpi_comm, &size);
        send_buffer = new char[size];
    }

    int ctr = 0;
    int prev_ctr = 0;
    for (int i = 0; i < comm->send_data->num_msgs; i++)
    {
        proc = comm->send_data->procs[i];
        start = comm->send_data->indptr[i];
        end = comm->send_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = comm->send_data->indices[j];
            MPI_Pack(&(data[idx*type_size]), 1, mpi_type, 
                    send_buffer, size, &ctr, mpi_comm);
        }
        MPI_Isend(&send_buffer[prev_ctr], ctr - prev_ctr, MPI_PACKED, proc, tag,
                mpi_comm, &send_requests[i]);
        prev_ctr = ctr;
    }

    if (send_buffer) *send_buffer_ptr = send_buffer;
}

// Inter-Node Irecvs
static void MPIX_step_recv(comm_pkg* comm,
        int tag, MPI_Comm mpi_comm, MPI_Datatype mpi_type,
        MPI_Request* recv_requests, char** recv_buffer_ptr)
{
    int proc, start, end, size;
    char* recv_buffer = NULL;
    if (comm->recv_data->size_msgs)
    {
        MPI_Pack_size(comm->recv_data->size_msgs, mpi_type, mpi_comm, &size);
        recv_buffer = new char[size];
    }

    int ctr = 0;
    for (int i = 0; i < comm->recv_data->num_msgs; i++)
    {
        proc = comm->recv_data->procs[i];
        start = comm->recv_data->indptr[i];
        end = comm->recv_data->indptr[i+1];
        MPI_Pack_size(end - start, mpi_type, mpi_comm, &size);
        MPI_Irecv(&recv_buffer[ctr], size, MPI_PACKED, proc, tag,
                mpi_comm, &recv_requests[i]);
        ctr += size;
    }

    if (recv_buffer) *recv_buffer_ptr = recv_buffer;
}

// Map received values to the appropriate locations
static void MPIX_intra_recv_map(comm_pkg* comm, char* intra_recv_data, void* inter_recv_data,
        MPI_Datatype mpi_type, MPI_Comm mpi_comm)
{
    int idx, addr;
    int size, type_size;
    char* char_ptr = reinterpret_cast<char*>(inter_recv_data);
    int* int_ptr = reinterpret_cast<int*>(inter_recv_data);
    MPI_Type_size(mpi_type, &type_size);
    MPI_Pack_size(comm->recv_data->size_msgs, mpi_type, mpi_comm, &size);

    int ctr = 0;
    for (int i = 0; i < comm->recv_data->size_msgs; i++)
    {
        idx = comm->recv_data->indices[i];
        MPI_Unpack(intra_recv_data, size, &ctr, &(char_ptr[idx*type_size]), 
                1, mpi_type, mpi_comm);
    }
}

#endif
