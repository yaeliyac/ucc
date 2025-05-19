/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoall.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_math.h"
#include "tl_ucp_sendrecv.h"

#include <nvcomp/nvcomp.h>
#include "nvcomp/bitcomp.h"


/* TODO: add as parameters */
#define MSG_MEDIUM 66000
#define NP_THRESH 32


nvcompType_t ucc_to_nvcomp_dtype[] = {
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT8)] = NVCOMP_TYPE_CHAR,
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT16)] = NVCOMP_TYPE_SHORT,
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT32)] = NVCOMP_TYPE_INT,
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT64)] = NVCOMP_TYPE_LONGLONG,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT8)] = NVCOMP_TYPE_UCHAR,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT16)] = NVCOMP_TYPE_USHORT,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT32)] = NVCOMP_TYPE_UINT,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT64)] = NVCOMP_TYPE_ULONGLONG,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT16)] = NVCOMP_TYPE_FLOAT16,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT32)] = NVCOMP_TYPE_INT, // TODO: check if this is correct
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT64)] = NVCOMP_TYPE_LONGLONG, // TODO: check if this is correct
    [UCC_DT_PREDEFINED_ID(UCC_DT_BFLOAT16)] = NVCOMP_TYPE_FLOAT16, // TODO: check if this is correct
};

static inline ucc_rank_t get_recv_peer(ucc_rank_t rank, ucc_rank_t size,
                                       ucc_rank_t step)
{
    return (rank + step) % size;
}

static inline ucc_rank_t get_send_peer(ucc_rank_t rank, ucc_rank_t size,
                                       ucc_rank_t step)
{
    return (rank - step + size) % size;
}

static ucc_rank_t get_num_posts(const ucc_tl_ucp_team_t *team,
                                const ucc_coll_args_t *args)
{
    unsigned long posts = UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoall_pairwise_num_posts;
    ucc_rank_t    tsize = UCC_TL_TEAM_SIZE(team);
    size_t data_size;

    data_size = (size_t)args->src.info.count *
                ucc_dt_size(args->src.info.datatype);
    if (posts == UCC_ULUNITS_AUTO) {
        if ((data_size > MSG_MEDIUM) && (tsize > NP_THRESH)) {
            /* use pairwise algorithm */
            posts = 1;
        } else {
            /* use linear algorithm */
            posts = 0;
        }
    }

    posts = (posts > tsize || posts == 0) ? tsize: posts;
    return posts;
}

void ucc_tl_ucp_alltoall_pairwise_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ptrdiff_t          sbuf  = (ptrdiff_t)TASK_ARGS(task).src.info.buffer;
    ptrdiff_t          rbuf  = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  smem  = TASK_ARGS(task).src.info.mem_type;
    ucc_memory_type_t  rmem  = TASK_ARGS(task).dst.info.mem_type;
    ucc_rank_t         grank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         gsize = UCC_TL_TEAM_SIZE(team);
    int                polls = 0;
    ucc_rank_t         peer, nreqs;
    size_t             data_size;

    nreqs     = get_num_posts(team, &TASK_ARGS(task));
    data_size = (size_t)(TASK_ARGS(task).src.info.count / gsize) *
                ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    while ((task->tagged.send_posted < gsize ||
            task->tagged.recv_posted < gsize) &&
           (polls++ < task->n_polls)) {
        ucp_worker_progress(UCC_TL_UCP_TEAM_CTX(team)->worker.ucp_worker);
        while ((task->tagged.recv_posted < gsize) &&
               ((task->tagged.recv_posted - task->tagged.recv_completed) <
                nreqs)) {
            peer = get_recv_peer(grank, gsize, task->tagged.recv_posted);
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nb((void *)(rbuf + peer * data_size),
                                             data_size, rmem, peer, team, task),
                          task, out);
            polls = 0;
        }
        while ((task->tagged.send_posted < gsize) &&
               ((task->tagged.send_posted - task->tagged.send_completed) <
                nreqs)) {
            peer = get_send_peer(grank, gsize, task->tagged.send_posted);
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb((void *)(sbuf + peer * data_size),
                                             data_size, smem, peer, team, task),
                          task, out);
            polls = 0;
        }
    }
    if ((task->tagged.send_posted < gsize) ||
        (task->tagged.recv_posted < gsize)) {
        return;
    }

    task->super.status = ucc_tl_ucp_test(task);
out:
    if (task->super.status != UCC_INPROGRESS) {
        UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task,
                                         "ucp_alltoall_pairwise_done", 0);
    }
}

ucc_status_t ucc_tl_ucp_alltoall_pairwise_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_alltoall_pairwise_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    // fill values in cpu scratch
    uint64_t cpu_scratch[4] = {0, 0, 0, 0};
    uint64_t *cpu_chunks_ptr = (uint64_t*) cpu_scratch;
    cpu_chunks_ptr[0] = (uint64_t)TASK_ARGS(task).src.info.buffer;
    uint64_t *cpu_size_ptr = (uint64_t*) (cpu_scratch + sizeof(uint64_t));
    cpu_size_ptr[0] = (uint64_t)TASK_ARGS(task).src.info.count * ucc_dt_size(TASK_ARGS(task).src.info.datatype);

    uint64_t *cpu_compressed_chunks_ptr = (uint64_t*) (cpu_scratch + sizeof(uint64_t) * 2);
    

    // copy from cpu to device
    ucc_mc_memcpy(task->alltoall_pairwise.scratch, cpu_scratch,
                  sizeof(cpu_scratch), UCC_MEMORY_TYPE_CUDA,
                  UCC_MEMORY_TYPE_HOST);

    void *device_chunks_ptr = task->alltoall_pairwise.scratch;
    void *device_size_ptr = task->alltoall_pairwise.scratch + sizeof(uint64_t);

    void *device_compressed_chunks_ptr = task->alltoall_pairwise.scratch + sizeof(uint64_t) * 2;
    void *device_compressed_size_ptr = task->alltoall_pairwise.scratch + sizeof(uint64_t) * 3;
    
    nvcompBatchedBitcompOpts_t opts = {0};
    opts.algorithm_type = 0;
    opts.data_type = ucc_to_nvcomp_dtype[TASK_ARGS(task).src.info.datatype];

    nvcompStatus_t status = nvcompBatchedBitcompCompressAsync(
        device_chunks_ptr,
        device_size_ptr,
        TASK_ARGS(task).src.info.count * ucc_dt_size(TASK_ARGS(task).src.info.datatype),
        1, // num_chunks, currently only 1 chunk is supported
        NULL, 0, // unused
        device_compressed_chunks_ptr,
        device_compressed_size_ptr,
        opts, NULL);

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        ucc_error("cuda error: %s\n", cudaGetErrorString(error));
        return UCC_ERR_INVALID_PARAM;
    }

    if (status != nvcompSuccess) {
        return UCC_ERR_INVALID_PARAM;
    }

    // copy from device to cpu
    ucc_mc_memcpy(cpu_scratch, task->alltoall_pairwise.scratch,
                  sizeof(cpu_scratch), UCC_MEMORY_TYPE_HOST,
                  UCC_MEMORY_TYPE_CUDA);

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_alltoall_pairwise_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t   *args = &TASK_ARGS(task);
    size_t data_size;

    task->super.post     = ucc_tl_ucp_alltoall_pairwise_start;
    task->super.progress = ucc_tl_ucp_alltoall_pairwise_progress;

    task->n_polls = ucc_max(1, task->n_polls);
    if (UCC_TL_UCP_TEAM_CTX(team)->cfg.pre_reg_mem) {
        data_size =
            (size_t)args->src.info.count * ucc_dt_size(args->src.info.datatype);
        ucc_tl_ucp_pre_register_mem(team, args->src.info.buffer, data_size,
                                    args->src.info.mem_type);
        ucc_tl_ucp_pre_register_mem(team, args->dst.info.buffer, data_size,
                                    args->dst.info.mem_type);
    }

    ucc_status_t status = ucc_mc_alloc(&task->alltoall_pairwise.scratch_mc_header, 4 * sizeof(uint64_t*), UCC_MEMORY_TYPE_CUDA);
    if (status != UCC_OK) {
        return status;
    }

    ucc_status_t status = ucc_mc_alloc(&task->alltoall_pairwise.compressed_mc_header, 1024 * sizeof(uint32_t), UCC_MEMORY_TYPE_CUDA);
    if (status != UCC_OK) {
        return status;
    }

    task->alltoall_pairwise.scratch =
        task->alltoall_pairwise.scratch_mc_header->addr;

    return UCC_OK;
}
