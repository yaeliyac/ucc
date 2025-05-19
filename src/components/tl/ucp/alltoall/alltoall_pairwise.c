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
#include "cuda.h"

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
    ucc_tl_ucp_alltoall_pairwise_metadata_t *metadata = task->alltoall_pairwise.metadata;
    ucc_rank_t tsize = UCC_TL_TEAM_SIZE(team);
    size_t msg_size = TASK_ARGS(task).src.info.count * ucc_dt_size(TASK_ARGS(task).src.info.datatype) / tsize;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_alltoall_pairwise_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    ucc_rank_t i;

    for (i = 0; i < tsize; i++) {
        metadata->device_uncompressed_chunk_ptrs[i] = (void*)PTR_OFFSET(TASK_ARGS(task).src.info.buffer, i * msg_size);
        metadata->device_uncompressed_chunk_bytes[i] = msg_size;
        metadata->device_compressed_chunk_ptrs[i] = (void*)PTR_OFFSET(TASK_ARGS(task).dst.info.buffer, i * msg_size);
        metadata->device_compressed_chunk_bytes[i] = 0;
    }

    nvcompBatchedBitcompOpts_t opts = {0};
    opts.algorithm_type = 0;
    opts.data_type = ucc_to_nvcomp_dtype[TASK_ARGS(task).src.info.datatype];

    nvcompStatus_t status = nvcompBatchedBitcompCompressAsync(
        (const void * const*)metadata->device_uncompressed_chunk_ptrs,
        metadata->device_uncompressed_chunk_bytes,
        TASK_ARGS(task).src.info.count * ucc_dt_size(TASK_ARGS(task).src.info.datatype),
        tsize, // num_chunks, currently only 1 chunk is supported
        NULL, 0, // unused
        metadata->device_compressed_chunk_ptrs,
        metadata->device_compressed_chunk_bytes,
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

    for (i = 0; i < tsize; i++) {
        ucc_print("rank %d chunk %d: uncompressed %zu bytes, compressed %zu bytes\n", UCC_TL_TEAM_RANK(team),
                   i, metadata->device_uncompressed_chunk_bytes[i], metadata->device_compressed_chunk_bytes[i]);
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_alltoall_pairwise_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t   *args = &TASK_ARGS(task);
    size_t data_size;
    ucc_rank_t tsize = UCC_TL_TEAM_SIZE(team);

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

    cudaHostAlloc((void**)&task->alltoall_pairwise.metadata, sizeof(ucc_tl_ucp_alltoall_pairwise_metadata_t), cudaHostAllocDefault);
    cudaHostAlloc((void**)&task->alltoall_pairwise.metadata->device_uncompressed_chunk_ptrs, sizeof(void*) * tsize, cudaHostAllocDefault);
    cudaHostAlloc((void**)&task->alltoall_pairwise.metadata->device_uncompressed_chunk_bytes, sizeof(size_t) * tsize, cudaHostAllocDefault);
    cudaHostAlloc((void**)&task->alltoall_pairwise.metadata->device_compressed_chunk_ptrs, sizeof(void*) * tsize, cudaHostAllocDefault);
    cudaHostAlloc((void**)&task->alltoall_pairwise.metadata->device_compressed_chunk_bytes, sizeof(size_t) * tsize, cudaHostAllocDefault);

    return UCC_OK;
}
