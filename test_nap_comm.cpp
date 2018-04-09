// Copyright (c) 2015-2017, Node-Aware MPI Development Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include "nap_comm.hpp"
#include <vector>
#include <set>

//TODO -- need to originally send global indices (to remove duplicates)
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n_sends = 7;
    int size_per_send = 25;
    int local_size = 10000;
    std::set<int> send_proc_set;
    std::vector<int> send_procs;
    std::vector<int> send_ptr;
    std::vector<int> send_indices;
    std::vector<int> send_requests;

    int n_recvs;
    std::vector<int> recv_procs;
    std::vector<int> recv_ptr;
    std::vector<int> recv_indices;
    std::vector<int> recv_buffer(size_per_send);
    srand(49352034 + rank);
    int tag = 4935;
    MPI_Status recv_status;
    MPI_Request barrier_request;
    int finished, msg_avail;
    int start, end, proc;
    int size;
    int ctr;

    // Create standard communication
    for (int i = 0; i < n_sends; i++)
    {
        proc = rand() % num_procs;
        while (proc == rank)
        {
            proc = rand() % num_procs;    
        }
        send_proc_set.insert(proc);
    }
    for (std::set<int>::iterator it = send_proc_set.begin();
            it != send_proc_set.end(); ++it)
    {
        send_procs.push_back(*it);
    }
    n_sends = send_procs.size();
    send_requests.resize(n_sends);

    ctr = 0;
    int first_idx = rank * local_size;
    send_ptr.push_back(0);
    for (int i = 0; i < n_sends; i++)
    {
        for (int j = 0; j < size_per_send; j++)
        {
            send_indices.push_back(first_idx + ctr++);
        }
        send_ptr.push_back(send_indices.size());
    }

    recv_ptr.push_back(0);
    for (int i = 0; i < n_sends; i++)
    {
        proc = send_procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        MPI_Isend(&send_indices[start], end - start, MPI_INT, proc, tag, 
                MPI_COMM_WORLD, &send_requests[i]);
    }
    MPI_Testall(n_sends, send_requests.data(), &finished, MPI_STATUSES_IGNORE);
    while (!finished)
    {
        MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &msg_avail, &recv_status);
        if (msg_avail)
        {
            MPI_Get_count(&recv_status, MPI_INT, &size);
            proc = recv_status.MPI_SOURCE;
            MPI_Recv(&recv_buffer[0], size, MPI_INT, proc, tag, MPI_COMM_WORLD, &recv_status);
            recv_procs.push_back(proc);
            for (int i = 0; i < size; i++)
            {
                recv_indices.push_back(recv_buffer[i]);
            }
            recv_ptr.push_back(recv_indices.size());
        }
        MPI_Testall(n_sends, send_requests.data(), &finished, MPI_STATUSES_IGNORE);        
    } 
    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);
    MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
    while (!finished)
    {
        MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &msg_avail, &recv_status);
        if (msg_avail)
        {
            MPI_Get_count(&recv_status, MPI_INT, &size);
            proc = recv_status.MPI_SOURCE;
            MPI_Recv(&recv_buffer[0], size, MPI_INT, proc, tag, MPI_COMM_WORLD, &recv_status);
            recv_procs.push_back(proc);
            for (int i = 0; i < size; i++)
            {
                recv_indices.push_back(recv_buffer[i]);
            }
            recv_ptr.push_back(recv_indices.size());
        }
        MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
    }
    n_recvs = recv_procs.size();

    for (int i = 0; i < n_recvs; i++)
    {
        proc = recv_procs[i];
    } 

    // Initializing node-aware communication package
    NAPComm* nap_comm;
    MPI_NAPinit(n_sends, send_procs.data(), send_ptr.data(), send_indices.data(), 
            n_recvs, recv_procs.data(), recv_ptr.data(), recv_indices.data(), 
            MPI_COMM_WORLD, &nap_comm);
    
    MPI_Finalize();
}
