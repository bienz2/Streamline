#include <mpi.h>
#include <vector>

using namespace std;

struct topo_data{
    int rank_ordering;
    int ppn;
    int num_nodes;
    int rank_node;
    MPI_Comm local_comm;

    topo_data(MPI_Comm mpi_comm)
    {
        int rank, num_procs;
        MPI_Comm_rank(mpi_comm, &rank);
        MPI_Comm_size(mpi_comm, &num_procs);

        char* proc_layout_c = getenv("MPICH_RANK_REORDER_METHOD");
        char* ppn_char = getenv("PPN");

        rank_ordering = 1;
        if (proc_layout_c)
        {
            rank_ordering = atoi(proc_layout_c);
        }
        ppn = 16;
        if (ppn_char)
        {
            ppn = atoi(ppn_char);
        }

        num_nodes = num_procs / ppn;
        if (num_procs % ppn) num_nodes++;

        if (rank_ordering == 0)
        {
            rank_node = rank % num_nodes;
        }
        else if (rank_ordering == 1)
        {
            rank_node = rank / ppn;
        }
        else if (rank_ordering == 2)
        {
            if ((rank / num_nodes) % 2 == 0)
            {
                rank_node = rank % num_nodes;
            }
            else
            {
                rank_node = num_nodes - (rank % num_nodes) - 1;
            }
        }
        MPI_Comm_split(mpi_comm, rank_node, rank, &local_comm);   
    }
};

struct comm_data{
    int num_msgs;
    int size_msgs;
    int* procs;
    int* indptr;
    int* indices;
};

struct comm_pkg{
    comm_data* send_data;
    comm_data* recv_data;
    
    comm_pkg()
    {
        send_data = new comm_data();
        recv_data = new comm_data();
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
};


/*template <typename T>
typedef struct {
    T* recv_data;
    T* local_L_recv_data;
    T* global_send_buffer;
    T* global_recv_buffer;
    MPI_Request* global_send_requests;
    MPI_Request* global_recv_requests;
} NAPData;*/

static MPI_Datatype get_type(int* buffer)
{
    return MPI_INT;
}
static MPI_Datatype get_type(double* buffer)
{
    return MPI_DOUBLE;
}

/*template <typename T>
void MPI_intra_comm(const comm_pkg* comm, const T* send_data, T** recv_data,
        const int tag, const MPI_Comm local_comm, const MPI_Datatype mpi_type)
{
    if (comm->send_data->num_msgs + comm->recv_data->num_msgs == 0) return;

    MPI_Request* send_requests;
    MPI_Request* recv_requests;
    T* send_buffer;
    T* recv_buffer;

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
        MPI_Isend(&send_buffer[start], end - start, mpi_type, proc, tag, 
                local_comm, &send_requests[i]);
    }
    for (int i = 0; i < comm->recv_data->num_msgs; i++)
    {
        proc = comm->recv_data->procs[i];
        start = comm->recv_data->indptr[i];
        end = comm->send_data->indptr[i+1];
        MPI_Irecv(&recv_buffer[start], end - start, mpi_type, proc, tag, 
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
MPI_inter_send(const comm_pkg* comm, const T* send_data,
        const int tag, const MPI_Comm mpi_comm, const MPI_Datatype mpi_type,
        MPI_Request** send_request_ptr, T** send_buffer_ptr)
{
    MPI_Request* send_requests;
    T* send_buffer;

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
MPI_inter_recv(const comm_pkg* comm,
        const int tag, const MPI_Comm mpi_comm, const MPI_Datatype mpi_type,
        MPI_Request** recv_request_ptr, T** recv_buffer_ptr)
{
    MPI_Request* recv_requests;
    T* recv_buffer;

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

MPI_inter_waitall(const comm_pkg* comm, const MPI_Request* send_requests, 
        const MPI_Request* recv_requests)
{
    if (comm->send_data->n_msgs)
    {
        MPI_Waitall(comm->send_data->n_msgs, send_requests, MPI_STATUSES_IGNORE);
    }

    if (comm->recv_data->n_msgs)
    {
        MPI_Waitall(comm->recv_data->n_msgs, recv_requests, MPI_STATUSES_IGNORE);
    }
}

template <typename T>
MPI_intra_recv_map(const comm_pkg* comm, const T* intra_recv_data, const T* inter_recv_data)
{
    for (int i = 0; i < comm->recv_data->size_msgs; i++)
    {
        idx = comm->recv_data->indices[i];
        inter_recv_data[idx] = intra_recv_data[i];
    }
}

template <typename T>
MPI_INAPsend(const T* send_data, const NAPComm* nap_comm, const int tag,
        const MPI_Comm mpi_comm, NapData* nap_data)
{
    MPI_Datatype mpi_type = get_type(send_data);

    int local_L_tag = 938401304;
    int local_S_tag = 530853024;
    T* local_L_recv_data;
    T* local_S_recv_data;
    T* global_send_buffer;
    MPI_Request* global_send_requests;

    // Fully intra-node communication
    MPI_intra_comm(nap_comm->local_L_par_comm, send_data, &local_L_recv_data,
            local_L_tag, mpi_comm, mpi_type);

    // Initial intra-node redistribution (step 1 in nap comm)
    MPI_intra_comm(nap_comm->local_S_par_comm, send_data, &local_S_recv_data,
            local_S_tag, mpi_comm, mpi_type);

    // Initialize Isends for inter-node step (step 2 in nap comm)
    MPI_inter_send(nap_comm->global_par_comm, local_S_recv_data, tag,
            mpi_comm, mpi_type, &global_send_requests, &global_send_buffer);

    // Store global_send_requests and global_send_buffer, as to not free data
    // before sends are finished
    nap_data->local_L_recv_data = local_L_recv_data;
    nap_data->global_send_requests = global_send_requests;
    nap_data->global_send_buffer = global_send_buffer;

    delete[] local_S_recv_data;
}

template <typename T>
MPI_INAPrecv(T* recv_data, const NAPComm* nap_comm, const int tag, 
        const MPI_Comm mpi_comm, NAPData* nap_data)
{
    MPI_Datatype mpi_type = get_type(send_data);

    MPI_Request* global_recv_requests;
    T* global_recv_buffer;
 
    // Initialize Irecvs for inter-node step (step 2 in nap comm)
    MPI_inter_recv(nap_comm->global_par_comm, tag, mpi_comm, mpi_type,
            &global_recv_requests, &global_recv_buffer);

    // Store global_recv_requests and global_recv_buffer, as to not free data
    // before recvs are finished
    nap_data->global_recv_requests = global_recv_requests;
    nap_data->global_recv_buffer = global_recv_buffer;
    nap_data->recv_data = recv_data;
}

MPI_NAPwait(const NAPComm* nap_comm, NAPData* nap_data)
{
    T* local_R_recv_data;
    T* local_L_recv_data = nap_data->local_L_recv_data;
    T* recv_data = nap_data->recv_data;
    T* global_send_buffer = nap_data->global_send_buffer;
    T* global_recv_buffer = nap_data->global_recv_buffer;
    MPI_Request* global_send_requests;
    MPI_Request* global_recv_requests;
    int local_R_tag = 403582495;

    MPI_inter_waitall(nap_comm->global_par_comm, global_send_requests,
            global_recv_requests);

    // Final intra-node redistribution (step 3 in nap comm)
    MPI_intra_comm(nap_comm->local_R_par_comm, global_recv_buffer, &local_R_recv_data,
            local_R_tag, mpi_comm, mpi_type);

    // Map recv buffers from final intra node steps to correct locations in
    // recv_data
    MPI_intra_recv_map(nap_comm->local_L_par_comm, local_L_recv_data, recv_data);
    MPI_intra_recv_map(nap_comm->local_R_par_comm, local_R_recv_data, recv_data);

    delete[] local_L_recv_data;
    delete[] local_R_recv_data;

    delete[] global_recv_buffer;
    delete[] global_send_buffer;
    delete[] global_send_requests;
    delete[] global_recv_requests;

    nap_data->recv_data = NULL;
}*/



// TODO -- find better way to implement this... don't want to count on
// environment variable


int get_node(const int proc, const topo_data* topo_info)
{
    if (topo_info->rank_ordering == 0)
    {
        return proc % topo_info->num_nodes;
    }
    else if (topo_info->rank_ordering == 1)
    {
        return proc / topo_info->ppn;
    }
    else if (topo_info->rank_ordering == 2)
    {
        if ((proc / topo_info->num_nodes) % 2 == 0)
        {
            return proc % topo_info->num_nodes;
        }
        else
        {
            return topo_info->num_nodes - (proc % topo_info->num_nodes) - 1;
        }
    }
    else
    { 
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
            printf("This MPI rank ordering is not supported!\n");
        }
        return -1;
    }
}
int get_local_proc(const int proc, const topo_data* topo_info)
{
    if (topo_info->rank_ordering == 0 || topo_info->rank_ordering == 2)
    {
        return proc / topo_info->num_nodes;
    }
    else if (topo_info->rank_ordering == 1)
    {
        return proc % topo_info->ppn;
    }
    else
    { 
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
            printf("This MPI rank ordering is not supported!\n");
        }
        return -1;
    }
}

int get_global_proc(const int node, const int local_proc, const topo_data* topo_info)
{
    if (topo_info->rank_ordering == 0)
    {
        return local_proc * topo_info->num_nodes + node;
    }
    else if (topo_info->rank_ordering == 1)
    {
        return local_proc + (node * topo_info->ppn);
    }
    else if (topo_info->rank_ordering == 2)
    {
        if (local_proc % 2 == 0)
        {
            return local_proc * topo_info->num_nodes + node;
        }
        else
        {
            return local_proc * topo_info->num_nodes + topo_info->num_nodes - node - 1;                
        }
    }
    else
    { 
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
            printf("This MPI rank ordering is not supported!\n");
        }
        return -1;
    }
}


void map_procs_to_nodes(NAPComm* nap_comm, const int orig_num_msgs, const int* orig_procs, const int* orig_indptr,
        std::vector<int>& msg_nodes, std::vector<int>& msg_node_to_local, MPI_Comm mpi_comm, 
        int init_inc = 1, int init_local_proc = -1)
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
    msg_node_to_local.resize(msg_nodes.size());
    local_proc = init_local_proc;
    inc = init_inc;
    for (int i = 0; i < msg_nodes.size(); i++)
    {
        msg_node_to_local[i] = local_proc;

        if (local_proc == local_num_procs - 1 && i > 0)
            inc = -1;
        else if (local_proc == 0 && i > 0)
           inc = 1; 
        else
            local_proc += inc;
    }
}

void form_local_comm(int orig_num_sends, int* orig_send_procs, int* orig_send_ptr,
        int* orig_send_indices, std::vector<int>& nodes_to_local, comm_data* send_data, 
        comm_data* recv_data, comm_data* local_data, std::vector<int>& recv_nodes, 
        MPI_Comm mpi_comm, MPI_Comm local_comm, topo_data* topo_info, int tag)
{   
    // Declare variables
    int global_proc, local_proc;
    int size, ctr, start_ctr;
    int start, end, node;
    int idx, proc_idx;
    int proc;
    MPI_Status recv_status;

    int num_sends = 0;
    int size_sends = 0;
    int* send_procs;
    int* send_indptr;
    int* send_indices;
    std::vector<int> send_buffer;
    std::vector<int> send_requests;
    std::vector<int> send_sizes;

    int num_recvs = 0;
    int size_recvs = 0;
    int* recv_procs;
    int* recv_indptr;
    int* recv_indices;
    std::vector<int> recv_idx_nodes;
    std::vector<int> recv_buffer;
    
    std::vector<int> orig_to_node;
    std::vector<int> orig_to_local;
    std::vector<int> local_idx;

    // MPI_Information
    int rank, num_procs;
    int local_rank, local_num_procs;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &num_procs);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_num_procs);

    // Initialize variables
    orig_to_local.resize(orig_num_sends);
    orig_to_node.resize(orig_num_sends);
    local_idx.resize(local_num_procs);
    send_sizes.resize(local_num_procs, 0);
    send_procs = new int[local_num_procs];
    send_indptr = new int[local_num_procs + 1];
    recv_procs = new int[local_num_procs];
    recv_indptr = new int[local_num_procs + 1];

    local_data->num_msgs = 0;
    local_data->procs = new int[local_num_procs];
    local_data->indptr = new int[local_num_procs];

    // Form local_S_comm
    local_data->indptr[0] = 0;
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
                local_idx[local_proc] = num_sends;
                send_procs[num_sends++] = local_proc;
            }
            orig_to_local[i] = local_proc;
            orig_to_node[i] = node;
            send_sizes[local_proc] += size;
        }
        else 
        {
            orig_to_local[i] = -1;
            local_data->procs[local_data->num_msgs] = get_local_proc(global_proc, topo_info);
            local_data->indptr[local_data->num_msgs + 1] = local_data->indptr[local_data->num_msgs] + size;
            local_data->num_msgs++;
        }
    }        
    local_data->size_msgs = local_data->indptr[local_data->num_msgs];
    
    send_indptr[0] = 0;
    for (int i = 0; i < num_sends; i++)
    {
        local_proc = send_procs[i];
        send_indptr[i+1] = send_indptr[i] + send_sizes[local_proc];
        send_sizes[local_proc] = 0;
    }
    size_sends = send_indptr[num_sends];
    local_data->indices = new int[size_sends];
    size_sends = 0;
    
    // Allocate send_indices and fill vector
    send_indices = new int[size_sends];
    for (int i = 0; i < orig_num_sends; i++)
    {
        local_proc = orig_to_local[i];
        proc_idx = local_idx[local_proc];
        start = orig_send_ptr[i];
        end = orig_send_ptr[i+1];
        if (local_proc == -1) 
        {
            for (int j = start; j < end; j++)
            {
                local_data->indices[size_sends++] = orig_send_indices[j];
            }
        }
        else
        {
            for (int j = start; j < end; j++)
            {
                idx = send_indptr[proc_idx] + send_sizes[local_proc]++;
                send_indices[idx] = orig_send_indices[j];
            }
        }
    }

    // Send 'local_S_comm send' info (to form local_S recv) 
    MPI_Allreduce(MPI_IN_PLACE, &send_sizes, local_num_procs, MPI_INT, MPI_SUM, local_comm);
    size_recvs = send_sizes[local_rank];
    recv_indices = new int[size_recvs];
    recv_idx_nodes.resize(size_recvs);

    send_buffer.resize(2*size_sends);
    send_requests.resize(num_sends);
    ctr = 0;
    start_ctr = 0;
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        start = send_indptr[i];
        end = send_indptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = send_indices[j];
            send_buffer[ctr++] = idx;
            send_buffer[ctr++] = orig_to_node[idx];
        }
        MPI_Isend(&send_buffer[start_ctr], ctr - start_ctr , MPI_INT, proc, tag, local_comm, &send_requests[i]);
        start_ctr = ctr;
    }

    ctr = 0;
    recv_indptr[0] = 0;
    while (ctr < size_recvs)
    {
        MPI_Probe(MPI_ANY_SOURCE, tag, local_comm, &recv_status);
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_INT, &size);
        if (size > recv_buffer.size())
            recv_buffer.resize(size);
        MPI_Recv(recv_buffer.data(), size, MPI_INT, proc, tag, local_comm, &recv_status);
        for (int i = 0; i < size; i += 2)
        {
            recv_indices[size_recvs] = recv_buffer[i];
            recv_idx_nodes[size_recvs++] = recv_buffer[i+1];
        }
        recv_procs[num_recvs] = proc;
        recv_indptr[num_recvs+1] = recv_indptr[num_recvs] + (size / 2);
        num_recvs++;
        ctr += size;
    }

    if (num_sends)
    {
        MPI_Waitall(num_sends, send_requests.data(), MPI_STATUSES_IGNORE);
    }

    send_data->num_msgs = num_sends;
    send_data->size_msgs = size_sends;
    send_data->procs = send_procs;
    send_data->indptr = send_indptr;
    send_data->indices = send_indices;

    recv_data->num_msgs = num_recvs;
    recv_data->size_msgs = size_recvs;
    recv_data->procs = recv_procs;
    recv_data->indptr = recv_indptr;
    recv_data->indices = recv_indices;
}

/*void form_global_comm(comm_data* local_data, comm_data* global_data, std::vector<int> local_data_nodes, 
        MPI_Comm mpi_comm, int tag)
{
    int num_nodes;
    int num_sends = 0;
    int* send_procs;
    int* send_indptr;
    int* send_indices;
    std::vector<int> tmp_send_indices;
    std::vector<int> node_sizes;
    std::vector<int> node_ctr;

    // Get MPI Information
    int rank, num_procs;
    int local_rank, local_num_procs;
    MPI_Comm local_comm = nap_comm->local_comm;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &num_procs);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_num_procs);
    int num_nodes = num_procs / local_num_procs;
    if (num_procs % local_num_procs) num_nodes++;
    int rank_node = get_node(rank);

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
            send_indptr[num_sends + 1] = send_indptr[num_sends] + node_sizes[i];
            node_sizes[i] = num_sends;
            num_sends++;
        }
    }
    size_sends = send_indptr[num_sends];
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

    // Remove duplicates
    start = send_indptr[0];
    ctr = 0;
    for (int i = 0; i < num_sends; i++)
    {
        end = send_indptr[i+1];
        std::sort(tmp_send_indices.begin() + start, tmp_send_indices.begin() + end);
        tmp_send_indices[ctr++] = tmp_send_indices[start];
        for (int j = start+1; j < end; j++)
        {
            if (tmp_send_indices[j] != tmp_send_indices[j+1])
            {
                tmp_send_indices[ctr++] = tmp_send_indices[j];
            }
        }
        start = end;
        send_indptr[i+1] = ctr;
    }
    size_sends = ctr;
    send_indices = new int[size_sends];
    for (int i = 0; i < size_sends; i++)
    {
        send_indices[i] = tmp_send_indices[ctr];
    }
     
    global_data->num_msgs = num_sends;
    global_data->size_msgs = size_sends;
    global_data->procs = send_procs;
    global_data->indptr = send_indptr;
    global_data->indices = send_indices;   
}

void update_global_comm(NAPComm* nap_comm)
{
    int n_sends = nap_comm->global_comm->send_data->num_msgs;
    int n_recvs = nap_comm->global_comm->recv_data->num_msgs;
    int n_msgs = n_sends + n_recvs;
    MPI_Request* reqeusts = NULL;
    int* send_buffer = NULL;
    int send_tag = 32148532;
    int recv_tag = 52395234;
    std::vector<int> send_nodes(num_nodes, 0);
    std::vector<int> recv_nodes(num_nodes, 0);
    if (n_msgs) 
    {
        requests = new MPI_Request[n_msgs];
        send_buffer = new int[n_msgs];
    }

    for (int i = 0; i < n_sends; i++)
    {
        node = nap_comm->global_par_comm->send_data->procs[i];
        global_proc = get_global_proc(node, local_rank);
        send_buffer[i] = node;
        MPI_Issend(&send_buffer[i], 1, MPI_INT, global_proc, send_tag, mpi_comm, &requests[i]);
    }
    for (int i = 0; i < n_recvs; i++)
    {
        node = nap_comm->global_par_comm->send_data->procs[i];
        global_proc = get_global_proc(node, local_rank);
        send_buffer[n_sends + i] = node;
        MPI_Issend(&send_buffer[n_sends + i], 1, MPI_INT, global_proc, recv_tag, mpi_comm,
                &requests[n_sends + i]);
    }

    MPI_Testall(n_msgs, requests, &finished, MPI_STATUSES_IGNORE);
    while (!finished)
    {
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, &msg_avail, &recv_status);
        if (msg_avail)
        {
            proc = recv_status.MPI_SOURCE;
            tag = recv_status.MPI_TAG;
            MPI_Recv(&node, 1, MPI_INT, proc, tag, mpi_comm, &recv_status);
            if (tag == send_tag)
            {
                send_nodes[node] = proc;
            }
            else
            {
                recv_nodes[node] = proc;
            }
        }
        MPI_Testall(n_msgs, requests, &finished, MPI_STATUSES_IGNORE);
                int source, int tag, MPI_Comm comm, int *flag, 
               MPI_Status *status
    }
    MPI_Ibarrier(mpi_comm, &barrier_request);
    MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
    while (!finished)
    {
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, &msg_avail, &recv_status);
        if (msg_avail)
        {
            proc = recv_status.MPI_SOURCE;
            tag = recv_status.MPI_TAG;
            MPI_Recv(&node, 1, MPI_INT, proc, tag, mpi_comm, &recv_status);
            if (tag == send_tag)
            {
                send_nodes[node] = proc;
            }
            else
            {
                recv_nodes[node] = proc;
            }
        }
        MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);   
    }

    MPI_Allreduce(MPI_IN_PLACE, &send_nodes, num_nodes, MPI_INT, MPI_SUM, local_comm);
    for (int i = 0; i < n_sends; i++)
    {
        node = nap_comm->global_comm->send_data->procs[i];
        nap_comm->global_comm->send_data->procs[i] = send_nodes[node];
    }
    for (int i = 0; i < n_recvs; i++)
    {
        node = nap_comm->global_comm->recv_data->procs[i];
        nap_comm->global_comm->recv_data>procs[i] = recv_nodes[node];
    }
}

// Need local_R_comm->send_data->indices (positions from global->recv that are
// sent)
void update_indices(NAPComm* nap_comm)
{
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
}*/

/* TODO - NEED METHODS:
 *     form_local_comm()
 */
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
    map_procs_to_nodes(nap_comm, n_sends, send_procs, send_indptr, send_nodes, send_node_to_local, 
            mpi_comm, 1, -1);

    // Find global recv nodes
    std::vector<int> recv_nodes;
    std::vector<int> recv_node_to_local;
    map_procs_to_nodes(nap_comm, n_recvs, recv_procs, recv_indptr, recv_nodes, recv_node_to_local, 
            mpi_comm, -1, -16);

    // Initialize local_S_comm

    *nap_comm_ptr = nap_comm;
}

void MPI_NAPdestroy(NAPComm** nap_comm_ptr)
{
    NAPComm* nap_comm = *nap_comm_ptr;
    delete[] nap_comm;
}


