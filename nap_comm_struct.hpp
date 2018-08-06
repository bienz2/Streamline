#ifndef NAPCOMM_STRUCT_HPP
#define NAPCOMM_STRUCT_HPP

#include <mpi.h>
#include <vector>
#include <map>
#include <algorithm>

// Class Structs
struct comm_data{
    int num_msgs;
    int size_msgs;
    int* procs;
    int* indptr;
    int* indices;

    comm_data()
    {
        num_msgs = 0;
	size_msgs = 0;
	procs = NULL;
	indptr = NULL;
	indices = NULL;
    }

    ~comm_data()
    {
        delete[] procs;
        delete[] indptr;
        delete[] indices;
    }
};

struct comm_pkg{
    comm_data* send_data;
    comm_data* recv_data;
    
    comm_pkg()
    {
        send_data = new comm_data();
        recv_data = new comm_data();
    }

    ~comm_pkg()
    {
        delete send_data;
        delete recv_data;
    }
};

struct NAPComm{
    comm_pkg* local_L_comm;
    comm_pkg* local_R_comm;
    comm_pkg* local_S_comm;
    comm_pkg* global_comm;
    topo_data* topo_info;

    NAPComm(topo_data* _topo_info)
    {
        local_L_comm = new comm_pkg();
        local_R_comm = new comm_pkg();
        local_S_comm = new comm_pkg();
        global_comm = new comm_pkg();
        topo_info = _topo_info;
    }

    ~NAPComm()
    {
        delete local_L_comm;
        delete local_R_comm;
        delete local_S_comm;
        delete global_comm;
    }
};

// forward declarations
void map_procs_to_nodes(NAPComm* nap_comm, const int orig_num_msgs, 
    const int* orig_procs, const int* orig_indptr,
    std::vector<int>& msg_nodes, std::vector<int>& msg_node_to_local, 
    MPI_Comm mpi_comm, bool incr = true);
void form_local_comm(const int orig_num_sends, const int* orig_send_procs, 
    const int* orig_send_ptr, const int* orig_send_indices, 
    const std::vector<int>& nodes_to_local, comm_data* send_data, 
    comm_data* recv_data, comm_data* local_data, 
    std::vector<int>& recv_idx_nodes, MPI_Comm mpi_comm, 
    topo_data* topo_info, const int tag);
void form_global_comm(comm_data* local_data, comm_data* global_data, 
    std::vector<int>& local_data_nodes, MPI_Comm mpi_comm, 
    topo_data* topo_info, int tag);
void update_global_comm(NAPComm* nap_comm, topo_data* topo_info, MPI_Comm mpi_comm);
void update_indices(NAPComm* nap_comm);

// Main Methods (initialize and destroy communicator)
void MPI_NAPinit(const int n_sends, const int* send_procs, const int* send_indptr, 
        const int* send_indices, const int n_recvs, const int* recv_procs, 
        const int* recv_indptr, const int* recv_indices, const MPI_Comm mpi_comm,
        NAPComm** nap_comm_ptr)
{
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &num_procs);

    // Create topo_data object
    topo_data* topology_info = new topo_data(mpi_comm);

    // Initialize structure
    NAPComm* nap_comm = new NAPComm(topology_info);

    // Find global send nodes
    std::vector<int> send_nodes;
    std::vector<int> send_node_to_local;
    map_procs_to_nodes(nap_comm, n_sends, send_procs, send_indptr, 
            send_nodes, send_node_to_local, mpi_comm, true);

    // Form initial send local comm
    std::vector<int> recv_idx_nodes;
    form_local_comm(n_sends, send_procs, send_indptr, send_indices, send_node_to_local,
            nap_comm->local_S_comm->send_data, nap_comm->local_S_comm->recv_data,
            nap_comm->local_L_comm->send_data, recv_idx_nodes, mpi_comm, 
            topology_info, 19483); 

    // Form global send data
    form_global_comm(nap_comm->local_S_comm->recv_data, nap_comm->global_comm->send_data,
            recv_idx_nodes, mpi_comm, topology_info, 93284);
            
    // Find global recv nodes
    std::vector<int> recv_nodes;
    std::vector<int> recv_node_to_local;
    map_procs_to_nodes(nap_comm, n_recvs, recv_procs, recv_indptr,
            recv_nodes, recv_node_to_local, mpi_comm, false);

    // Form final recv local comm
    std::vector<int> send_idx_nodes;
    form_local_comm(n_recvs, recv_procs, recv_indptr, recv_indices, recv_node_to_local,
            nap_comm->local_R_comm->recv_data, nap_comm->local_R_comm->send_data,
            nap_comm->local_L_comm->recv_data, send_idx_nodes, mpi_comm, 
            topology_info, 32048);
    
    // Form global recv data
    form_global_comm(nap_comm->local_R_comm->send_data, nap_comm->global_comm->recv_data,
            send_idx_nodes, mpi_comm, topology_info, 93284);

    // Update procs for global_comm send and recvs
    update_global_comm(nap_comm, topology_info, mpi_comm);

    // Update send and receive indices
    update_indices(nap_comm);

    // Copy to pointer for return
    *nap_comm_ptr = nap_comm;
}

void MPI_NAPDestroy(NAPComm** nap_comm_ptr)
{
    NAPComm* nap_comm = *nap_comm_ptr;
    topo_data* topo_info = nap_comm->topo_info;
    delete topo_info;
    delete nap_comm;
}


/******************************************
 ****
 **** Helper Methods 
 ****
 ******************************************/
void map_procs_to_nodes(NAPComm* nap_comm, const int orig_num_msgs, 
        const int* orig_procs, const int* orig_indptr,
        std::vector<int>& msg_nodes, std::vector<int>& msg_node_to_local, 
        MPI_Comm mpi_comm, bool incr)
{    
    int rank, num_procs;
    int local_rank, local_num_procs;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &num_procs);
    MPI_Comm& local_comm = nap_comm->topo_info->local_comm;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_num_procs);

    int proc, size, node;
    int local_proc;
    int inc;
    std::vector<int> node_sizes;

    int num_nodes = nap_comm->topo_info->num_nodes;
    int rank_node = nap_comm->topo_info->rank_node;

    // Map local msg_procs to local msg_nodes
    node_sizes.resize(num_nodes, 0);
    for (int i = 0; i < orig_num_msgs; i++)
    {
        proc = orig_procs[i];
        size = orig_indptr[i+1] - orig_indptr[i];
        node = get_node(proc, nap_comm->topo_info);
        node_sizes[node] += size;
    }
    
    // Gather all send nodes and sizes among ranks local to node
    MPI_Allreduce(MPI_IN_PLACE, node_sizes.data(), num_nodes, MPI_INT, MPI_SUM, local_comm);
    for (int i = 0; i < num_nodes; i++)
    {
        if (node_sizes[i] && i != rank_node)
        {
            msg_nodes.push_back(i);
        }
    }
    std::sort(msg_nodes.begin(), msg_nodes.end(), 
            [&](const int i, const int j)
            {
                return node_sizes[i] > node_sizes[j];
            });

    // Map send_nodes to local ranks
    msg_node_to_local.resize(num_nodes, -1);
    if (incr)
    {
        local_proc = 0;
        inc = 1;
    }
    else
    {
        local_proc = local_num_procs - 1;
        inc = -1;
    }
    for (int i = 0; i < msg_nodes.size(); i++)
    {
        node = msg_nodes[i];
        msg_node_to_local[node] = local_proc;

        if (local_proc == local_num_procs - 1 && inc == 1)
            inc = -1;
        else if (local_proc == 0 && inc == -1)
           inc = 1; 
        else
            local_proc += inc;
    }
}

void form_local_comm(const int orig_num_sends, const int* orig_send_procs, 
        const int* orig_send_ptr, const int* orig_send_indices, 
        const std::vector<int>& nodes_to_local, comm_data* send_data, 
        comm_data* recv_data, comm_data* local_data, 
        std::vector<int>& recv_idx_nodes, MPI_Comm mpi_comm, 
        topo_data* topo_info, const int tag)
{   
    // MPI_Information
    int rank, num_procs;
    int local_rank, local_num_procs;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &num_procs);
    MPI_Comm& local_comm = topo_info->local_comm;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_num_procs);

    // Declare variables
    int global_proc, local_proc;
    int size, ctr, start_ctr;
    int start, end, node;
    int idx, proc_idx;
    int proc;
    MPI_Status recv_status;

    std::vector<int> send_buffer;
    std::vector<MPI_Request> send_requests;
    std::vector<int> send_sizes;
    std::vector<int> recv_buffer;
    
    std::vector<int> orig_to_node;
    std::vector<int> local_idx;

    // Initialize variables
    orig_to_node.resize(orig_num_sends);
    local_idx.resize(local_num_procs);
    send_sizes.resize(local_num_procs, 0);

    send_data->num_msgs = 0;
    send_data->size_msgs = 0;
    send_data->procs = new int[local_num_procs];
    send_data->indptr = new int[local_num_procs + 1];
    send_data->indices = NULL;
    send_data->indptr[0] = 0;

    recv_data->num_msgs = 0;
    recv_data->size_msgs = 0;
    recv_data->procs = new int[local_num_procs];
    recv_data->indptr = new int[local_num_procs + 1];
    recv_data->indices = NULL;
    recv_data->indptr[0] = 0;

    local_data->num_msgs = 0;
    local_data->size_msgs = 0;
    local_data->procs = new int[local_num_procs];
    local_data->indptr = new int[local_num_procs];
    local_data->indices = NULL;
    local_data->indptr[0] = 0;

    // Form local_S_comm
    for (int i = 0; i < orig_num_sends; i++)
    {
        global_proc = orig_send_procs[i];
        size = orig_send_ptr[i+1] - orig_send_ptr[i];
        node = get_node(global_proc, topo_info);
        if (topo_info->rank_node != node)
        {
            local_proc = nodes_to_local[node];
            if (send_sizes[local_proc] == 0)
            {
                local_idx[local_proc] = send_data->num_msgs;
                send_data->procs[send_data->num_msgs++] = local_proc;
            }
            orig_to_node[i] = node;
            send_sizes[local_proc] += size;
        }
        else 
        {
            orig_to_node[i] = -1;
            local_data->procs[local_data->num_msgs] = get_local_proc(global_proc, topo_info);
            local_data->size_msgs += size;
            local_data->num_msgs++;
            local_data->indptr[local_data->num_msgs] = local_data->size_msgs;
        }
    }        
    local_data->indices = new int[local_data->size_msgs];
    
    for (int i = 0; i < send_data->num_msgs; i++)
    {
        local_proc = send_data->procs[i];
        send_data->indptr[i+1] = send_data->indptr[i] + send_sizes[local_proc];
        send_sizes[local_proc] = 0;
    }
    send_data->size_msgs = send_data->indptr[send_data->num_msgs];
    
    // Allocate send_indices and fill vector
    send_data->indices = new int[send_data->size_msgs];
    std::vector<int> send_idx_node(send_data->size_msgs);
    local_data->size_msgs = 0;
    for (int i = 0; i < orig_num_sends; i++)
    {
        node = orig_to_node[i];
        start = orig_send_ptr[i];
        end = orig_send_ptr[i+1];
        if (node == -1) 
        {
            for (int j = start; j < end; j++)
            {
                local_data->indices[local_data->size_msgs++] = orig_send_indices[j];
            }
        }
        else
        {
            local_proc = nodes_to_local[node];
            proc_idx = local_idx[local_proc];
            for (int j = start; j < end; j++)
            {
                idx = send_data->indptr[proc_idx] + send_sizes[local_proc]++;
                send_data->indices[idx] = orig_send_indices[j];
                send_idx_node[idx] = node;
            }
        }
    }

    // Send 'local_S_comm send' info (to form local_S recv) 
    MPI_Allreduce(MPI_IN_PLACE, send_sizes.data(), local_num_procs, 
            MPI_INT, MPI_SUM, local_comm);
    recv_data->size_msgs = send_sizes[local_rank];
    recv_data->indices = new int[recv_data->size_msgs];
    recv_idx_nodes.resize(recv_data->size_msgs);

    send_buffer.resize(2*send_data->size_msgs);
    send_requests.resize(send_data->num_msgs);
    ctr = 0;
    start_ctr = 0;
    for (int i = 0; i < send_data->num_msgs; i++)
    {
        proc = send_data->procs[i];
        start = send_data->indptr[i];
        end = send_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            send_buffer[ctr++] = send_data->indices[j];
            send_buffer[ctr++] = send_idx_node[j];
        }
        MPI_Isend(&send_buffer[start_ctr], ctr - start_ctr , 
                MPI_INT, proc, tag, local_comm, &send_requests[i]);
        start_ctr = ctr;
    }

    ctr = 0;
    while (ctr < recv_data->size_msgs)
    {
        MPI_Probe(MPI_ANY_SOURCE, tag, local_comm, &recv_status);
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_INT, &size);
        if (size > recv_buffer.size())
            recv_buffer.resize(size);
        MPI_Recv(recv_buffer.data(), size, MPI_INT, proc, tag, local_comm, &recv_status);
        for (int i = 0; i < size; i += 2)
        {
            recv_data->indices[ctr] = recv_buffer[i];
            recv_idx_nodes[ctr++] = recv_buffer[i+1];
        }
        recv_data->procs[recv_data->num_msgs] = proc;
        recv_data->indptr[recv_data->num_msgs + 1] = recv_data->indptr[recv_data->num_msgs] + (size / 2);
        recv_data->num_msgs++;
    }

    if (send_data->num_msgs)
    {
        MPI_Waitall(send_data->num_msgs, send_requests.data(), MPI_STATUSES_IGNORE);
    }
}

void form_global_comm(comm_data* local_data, comm_data* global_data, 
        std::vector<int>& local_data_nodes, MPI_Comm mpi_comm, 
        topo_data* topo_info, int tag)
{
    int num_sends = 0;
    int* send_procs;
    int* send_indptr;
    int* send_indices;
    int size_sends = 0;
    std::vector<int> tmp_send_indices;
    std::vector<int> node_sizes;
    std::vector<int> node_ctr;

    // Get MPI Information
    int rank, num_procs;
    int local_rank, local_num_procs;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &num_procs);
    MPI_Comm& local_comm = topo_info->local_comm;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_num_procs);
    int num_nodes = topo_info->num_nodes;
    int rank_node = topo_info->rank_node;

    int node_idx;
    int start, end, idx;
    int ctr, node, size;

    node_sizes.resize(num_nodes, 0);

    for (int i = 0; i < local_data->num_msgs; i++)
    {
        node = local_data_nodes[i];
        size = local_data->indptr[i+1] - local_data->indptr[i];
        if (node_sizes[node] == 0)
        {
            num_sends++;
        }
        node_sizes[node] += size;
    }
    send_procs = new int[num_sends];
    send_indptr = new int[num_sends + 1];
    node_ctr.resize(num_sends, 0);

    num_sends = 0;
    send_indptr[0] = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        if (node_sizes[i])
        {
            send_procs[num_sends] = i;
            size_sends += node_sizes[i];
            node_sizes[i] = num_sends;
            num_sends++;
            send_indptr[num_sends] = size_sends; 
        }
    }
    tmp_send_indices.resize(size_sends);

    for (int i = 0; i < local_data->num_msgs; i++)
    {
        node = local_data_nodes[i];
        node_idx = node_sizes[node];
        start = local_data->indptr[i];
        end = local_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = send_indptr[node_idx] + node_ctr[node_idx]++;
            tmp_send_indices[idx] = local_data->indices[j];
        }
    }

    // Sort indices
    for (int i = 0; i < num_sends; i++)
    {
        start = send_indptr[i];
        end = send_indptr[i+1];
        std::sort(tmp_send_indices.begin() + start, tmp_send_indices.begin() + end);
    }

    // Remove duplicates
    ctr = 0;
    start = send_indptr[0];
    for (int i = 0; i < num_sends; i++)
    {
        end = send_indptr[i+1];
        tmp_send_indices[ctr++] = tmp_send_indices[start];
        for (int j = start; j < end - 1; j++)
        {
            if (tmp_send_indices[j+1] != tmp_send_indices[j])
            {
                tmp_send_indices[ctr++] = tmp_send_indices[j+1];
            }
        }
        start = end;
        send_indptr[i+1] = ctr;
    }
    size_sends = ctr;
    send_indices = new int[size_sends];
    for (int i = 0; i < size_sends; i++)
    {
        send_indices[i] = tmp_send_indices[i];
    }

    global_data->num_msgs = num_sends;
    global_data->size_msgs = size_sends;
    global_data->procs = send_procs;
    global_data->indptr = send_indptr;
    global_data->indices = send_indices;   
}

void update_global_comm(NAPComm* nap_comm, topo_data* topo_info, MPI_Comm mpi_comm)
{
    int rank, num_procs;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &num_procs);
    MPI_Comm& local_comm = topo_info->local_comm;
    int local_rank, local_num_procs;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_num_procs);
    int num_nodes = topo_info->num_nodes;

    int n_sends = nap_comm->global_comm->send_data->num_msgs;
    int n_recvs = nap_comm->global_comm->recv_data->num_msgs;
    int n_msgs = n_sends + n_recvs;
    MPI_Request* requests = NULL;
    int* send_buffer = NULL;
    int send_tag = 32148532;
    int recv_tag = 52395234;
    int node, global_proc, tag;
    MPI_Status recv_status;
    std::vector<int> send_nodes(num_nodes, 0);
    std::vector<int> recv_nodes(num_nodes, 0);
    if (n_msgs) 
    {
        requests = new MPI_Request[n_msgs];
        send_buffer = new int[n_msgs];
    }

    std::vector<int> comm_procs(num_procs, 0);
    for (int i = 0; i < n_sends; i++)
    {
        node = nap_comm->global_comm->send_data->procs[i];
        global_proc = get_global_proc(node, local_rank, topo_info);
        comm_procs[global_proc]++;
    }
    for (int i = 0; i < n_recvs; i++)
    {
        node = nap_comm->global_comm->recv_data->procs[i];
        global_proc = get_global_proc(node, local_rank, topo_info);
        comm_procs[global_proc]++;
    }

    MPI_Allreduce(MPI_IN_PLACE, comm_procs.data(), num_procs, MPI_INT,
            MPI_SUM, MPI_COMM_WORLD);
    int num_to_recv = comm_procs[rank];

    for (int i = 0; i < n_sends; i++)
    {
        node = nap_comm->global_comm->send_data->procs[i];
        global_proc = get_global_proc(node, local_rank, topo_info);
        send_buffer[i] = node;
        MPI_Isend(&send_buffer[i], 1, MPI_INT, global_proc, send_tag, 
                mpi_comm, &requests[i]);
    }
    for (int i = 0; i < n_recvs; i++)
    {
        node = nap_comm->global_comm->recv_data->procs[i];
        global_proc = get_global_proc(node, local_rank, topo_info);
        send_buffer[n_sends + i] = node;
        MPI_Isend(&send_buffer[n_sends + i], 1, MPI_INT, global_proc, recv_tag, 
                mpi_comm, &requests[n_sends + i]);
    }

    for (int i = 0; i < num_to_recv; i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, mpi_comm, &recv_status);
        global_proc = recv_status.MPI_SOURCE;
        tag = recv_status.MPI_TAG;
        MPI_Recv(&node, 1, MPI_INT, global_proc, tag, mpi_comm, &recv_status);
        if (tag == send_tag)
        {
            send_nodes[node] = global_proc;
        }
        else
        {
            recv_nodes[node] = global_proc;
        }
    }

    if (n_sends + n_recvs)
        MPI_Waitall(n_sends + n_recvs, requests, MPI_STATUSES_IGNORE);

    MPI_Allreduce(MPI_IN_PLACE, send_nodes.data(), num_nodes, MPI_INT, MPI_SUM, local_comm);
    MPI_Allreduce(MPI_IN_PLACE, recv_nodes.data(), num_nodes, MPI_INT, MPI_SUM, local_comm);
    for (int i = 0; i < n_sends; i++)
    {
        node = nap_comm->global_comm->send_data->procs[i];
        nap_comm->global_comm->send_data->procs[i] = send_nodes[node];
    }
    for (int i = 0; i < n_recvs; i++)
    {
        node = nap_comm->global_comm->recv_data->procs[i];
        nap_comm->global_comm->recv_data->procs[i] = recv_nodes[node];
    }
}

// Need local_R_comm->send_data->indices (positions from global->recv that are
// sent)
void update_indices(NAPComm* nap_comm)
{
    int idx;
    std::map<int, int> global_recv_indices;

    for (int i = 0; i < nap_comm->global_comm->recv_data->size_msgs; i++)
    {
        idx = nap_comm->global_comm->recv_data->indices[i];
        global_recv_indices[idx] = i;
    }

    for (int i = 0; i < nap_comm->local_R_comm->send_data->size_msgs; i++)
    {
        idx = nap_comm->local_R_comm->send_data->indices[i];
        nap_comm->local_R_comm->send_data->indices[i] = global_recv_indices[idx];
    }
}

#endif
