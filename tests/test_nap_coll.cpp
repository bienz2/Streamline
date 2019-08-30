// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails
// while EXPECT_* variants continue with the run.


#include "gtest/gtest.h"
#include "src/allreduce/nap_allreduce.hpp"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //


TEST(NAPCollTest, TestsInTests)
{
    int ppn = 16;

    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Comm local_comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank / ppn, rank, &local_comm);

    std::vector<int> step_sizes;
    int num_nodes = num_procs / ppn;
    int size = num_nodes;
    while (size > ppn)
    {
        size /= ppn;
        step_sizes.push_back(ppn);
    }
    if (size > 1) step_sizes.push_back(size);
    if (rank == 0) printf("NumNodes %d, NumSteps %d:\n", num_nodes, step_sizes.size());
    if (rank == 0) for (int i = 0; i < step_sizes.size(); i++) printf("Size[%d] = %d\n", i, step_sizes[i]);

    int n_masters = num_procs / ppn;
    int masters[n_masters];
    for (int i = 0; i < n_masters; i++)
        masters[i] = i*ppn;

    MPI_Group world_group;
    MPI_Group master_group;
    MPI_Comm master_comm;

    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, n_masters, masters, &master_group);
    
    if (rank % ppn == 0)
    {
        MPI_Comm_create_group(MPI_COMM_WORLD, master_group, 0, &master_comm);
    }

    int n = 1234;
    int* vals = new int[n];
    int* val_sums = new int[n];
    int* val_sums_RD = new int[n];
    int* val_sums_NAP = new int[n];
    int* val_sums_SMP = new int[n];

    for (int i = 0; i < n; i++)
        vals[i] = rank*n + i;

    MPI_Allreduce(vals, val_sums, n, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPIX_Allreduce_RD(vals, val_sums_RD, n, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < n; i++)
        ASSERT_EQ(val_sums[i], val_sums_RD[i]);

    MPIX_Allreduce_NAP(vals, val_sums_NAP, n, MPI_INT, local_comm, 
        ppn, step_sizes.data(), step_sizes.size());
    for (int i = 0; i < n; i++)
        ASSERT_EQ(val_sums[i], val_sums_NAP[i]);

    MPIX_Allreduce_SMP(vals, val_sums_SMP, n, MPI_INT, local_comm, master_comm);
    for (int i = 0; i < n; i++)
        ASSERT_EQ(val_sums[i], val_sums_SMP[i]);


    delete[] vals;
    delete[] val_sums;
    delete[] val_sums_RD;
    delete[] val_sums_NAP;
    delete[] val_sums_SMP;

    MPI_Comm_free(&local_comm);
    MPI_Group_free(&world_group);
    MPI_Group_free(&master_group);

    if (rank % ppn == 0)
        MPI_Comm_free(&master_comm);
}


