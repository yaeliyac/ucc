/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "allgather.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "components/mc/ucc_mc.h"
#include "cuda.h"
#include <nvcomp/ans.h>


nvcompANSDataType_t ucc_to_nvcompans_dtype[] = {
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT8)] = 0,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT8)] = 0,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT16)] = 1,
};

void ucc_tl_ucp_recv_completion_allgather(void *request, ucs_status_t status,
                                      const ucp_tag_recv_info_t *info, /* NOLINT */
                                      void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    ucc_rank_t sender_rank;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in recv completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }

    sender_rank = UCC_TL_UCP_GET_SENDER(info->sender_tag);
    task->allgather_linear_comp.metadata->device_compressed_chunk_bytes[sender_rank] = info->length;
    ++task->tagged.recv_completed;
    ucp_request_free(request);
}

void ucc_tl_ucp_allgather_linear_comp_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t    *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t    *team      = TASK_TEAM(task);
    ucc_tl_ucp_context_t *ctx       = UCC_TL_UCP_TEAM_CTX(team);
    ucc_rank_t            trank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t            tsize     = UCC_TL_TEAM_SIZE(team);
    void                 *rbuf      = TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t     rmem      = TASK_ARGS(task).dst.info.mem_type;
    ucc_datatype_t        dt        = TASK_ARGS(task).dst.info.datatype;
    size_t                count     = TASK_ARGS(task).dst.info.count;
    size_t                data_size = (count / tsize) * ucc_dt_size(dt);
    int                   nreqs     = task->allgather_linear.nreqs;
    int                   polls     = 0;
    void                 *rbuf_compressed = task->allgather_linear_comp.dst_compressed->addr;
    void                 *sbuf_compressed = task->allgather_linear_comp.src_compressed->addr;
    void                 *tmpsend   = UCC_IS_INPLACE(TASK_ARGS(task))
                                          ? PTR_OFFSET(rbuf_compressed, trank * data_size)
                                          : sbuf_compressed;
    ucc_tl_ucp_compress_metadata_t *metadata = task->allgather_linear_comp.metadata;
    ucc_memory_type_t     smem      = UCC_IS_INPLACE(TASK_ARGS(task))
                                          ? rmem
                                          : TASK_ARGS(task).src.info.mem_type;
    void                 *tmprecv;
    ucc_rank_t            peer;
    ucc_status_t          status;

    while ((task->tagged.send_posted < tsize - 1 ||
            task->tagged.recv_posted < tsize - 1) &&
           (polls++ < task->n_polls)) {

        /* Progress UCP worker */
        ucp_worker_progress(UCC_TL_UCP_TEAM_CTX(team)->worker.ucp_worker);

        /* Try to send data to clockwise peer */
        while ((task->tagged.send_posted < tsize - 1) &&
               ((task->tagged.send_posted - task->tagged.send_completed) <
                nreqs)) {
            peer    = (trank + 1 + task->tagged.send_posted) % tsize;
            /* Send my data to peer */
            UCPCHECK_GOTO(
                ucc_tl_ucp_send_nb(tmpsend, metadata->device_compressed_chunk_bytes[0], smem, peer, team, task),
                task, err);
            polls = 0;
        }

        /* Receive peer's data from counter-clockwise peer to avoid deadlock*/
        while ((task->tagged.recv_posted < tsize - 1) &&
               ((task->tagged.recv_posted - task->tagged.recv_completed) <
                nreqs)) {
            peer    = (tsize + trank - 1 - task->tagged.recv_posted) % tsize;
            tmprecv = PTR_OFFSET(rbuf_compressed, peer * data_size);

            UCPCHECK_GOTO(
                ucc_tl_ucp_recv_cb(tmprecv, data_size, rmem, peer, team, task,
                                  ucc_tl_ucp_recv_completion_allgather, task),
                task, err);
            polls = 0;
        }
    }

    if (task->tagged.send_posted < tsize - 1 ||
        task->tagged.recv_posted < tsize - 1) {
        return;
    }

    task->super.status = ucc_tl_ucp_test(task);
    if (task->super.status != UCC_OK) {
        return;
    }

    /* Need to check copy task if it is not in-place */
    if (task->allgather_linear.copy_task != NULL) {
        status = ctx->copy.test(ctx, task->allgather_linear.copy_task);
        if (status > 0) {
            task->super.status = UCC_INPROGRESS;
            return;
        }
        task->super.status = status;
        ctx->copy.finalize(task->allgather_linear.copy_task);
        task->allgather_linear.copy_task = NULL;
    }

    metadata->device_compressed_chunk_bytes[trank] = metadata->device_compressed_chunk_bytes[0];
    for (int i = 0; i < tsize; i++) {
        metadata->device_compressed_chunk_ptrs[i] = PTR_OFFSET(rbuf_compressed, i * data_size);
        metadata->device_uncompressed_chunk_ptrs[i] = PTR_OFFSET(rbuf, i * data_size);
        metadata->device_uncompressed_chunk_bytes[i] = data_size;
    }

    nvcompStatus_t compstatus = nvcompBatchedANSDecompressAsync(
        (const void * const*)metadata->device_compressed_chunk_ptrs,
        metadata->device_compressed_chunk_bytes,
        metadata->device_uncompressed_chunk_bytes,
        metadata->device_uncompressed_chunk_bytes,
        tsize,
        task->allgather_linear_comp.tmp_memory->addr,
        metadata->tmp_size,
        metadata->device_uncompressed_chunk_ptrs,
        metadata->status,
        team->stream);
    cudaStreamSynchronize(team->stream);
    if (compstatus != nvcompSuccess) {
        ucc_error("nvcomp error: %d\n", (compstatus));
        return;
    }
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_linear_done", 0);
    return;
err:
    ucc_error("allgather linear progress failed with status %d: %s",
              task->super.status, ucc_status_string(task->super.status));
}

ucc_status_t ucc_tl_ucp_allgather_linear_comp_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t    *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t    *team      = TASK_TEAM(task);
    ucc_tl_ucp_context_t *ctx       = UCC_TL_UCP_TEAM_CTX(team);
    size_t                count     = TASK_ARGS(task).dst.info.count;
    void                 *sbuf      = TASK_ARGS(task).src.info.buffer;
    ucc_memory_type_t     smem      = TASK_ARGS(task).src.info.mem_type;
    ucc_memory_type_t     rmem      = TASK_ARGS(task).dst.info.mem_type;
    ucc_datatype_t        dt        = TASK_ARGS(task).dst.info.datatype;
    ucc_rank_t            trank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t            tsize     = UCC_TL_TEAM_SIZE(team);
    size_t                data_size = (count / tsize) * ucc_dt_size(dt);
    ucc_status_t          status;
    ucc_tl_ucp_compress_metadata_t *metadata = task->allgather_linear_comp.metadata;
    nvcompBatchedANSOpts_t opts = {0};
    void *sbuf_compressed = task->allgather_linear_comp.src_compressed->addr;
    void *rbuf_compressed = task->allgather_linear_comp.dst_compressed->addr;

    opts.data_type = ucc_to_nvcompans_dtype[dt];
    opts.type = nvcomp_rANS;


    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_linear_start",
                                     0);

    metadata->device_uncompressed_chunk_ptrs[0] = sbuf;
    metadata->device_uncompressed_chunk_bytes[0] = data_size;
    metadata->device_compressed_chunk_ptrs[0] = sbuf_compressed;
    metadata->device_compressed_chunk_bytes[0] = 0;


    nvcompStatus_t compstatus = nvcompBatchedANSCompressAsync(
        (const void * const*)metadata->device_uncompressed_chunk_ptrs,
        metadata->device_uncompressed_chunk_bytes,
        data_size,
        1,
        task->allgather_linear_comp.tmp_memory->addr, //fix,
        metadata->tmp_size, // fix
        metadata->device_compressed_chunk_ptrs,
        metadata->device_compressed_chunk_bytes,
        opts,
        team->stream);
    cudaStreamSynchronize(team->stream);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        ucc_error("cuda error: %s\n", cudaGetErrorString(error));
        return UCC_ERR_INVALID_PARAM;
    }

    if (compstatus != nvcompSuccess) {
        ucc_error("nvcomp error: %d\n", compstatus);
        return UCC_ERR_INVALID_PARAM;
    }

    ucc_print("rank %d: uncompressed %zu bytes, compressed %zu bytes\n", trank,
              metadata->device_uncompressed_chunk_bytes[0], metadata->device_compressed_chunk_bytes[0]);

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    task->allgather_linear.copy_task = NULL;

    /* Copy local data to the receive buffer if not in-place */
    if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
        status = ctx->copy.post(PTR_OFFSET(rbuf_compressed, data_size * trank), rmem,
                                sbuf_compressed, smem, data_size, task,
                                &task->allgather_linear.copy_task);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

/* Get the number of requests in flight to be used for the allgather batched algorithm
 * If the number of requests is not specified, use the number of team size - 1
 * If number of request is bigger than the team size - 1, use the team size - 1
 */
static unsigned long get_num_reqs(const ucc_tl_ucp_team_t *team)
{
    unsigned long reqs =
        UCC_TL_UCP_TEAM_LIB(team)->cfg.allgather_batched_num_posts;
    ucc_rank_t max_req = UCC_TL_TEAM_SIZE(team) - 1;
    reqs = (reqs > max_req || reqs == UCC_ULUNITS_AUTO) ? max_req : reqs;
    return reqs;
}

ucc_status_t ucc_tl_ucp_allgather_linear_comp_init(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task    = ucc_tl_ucp_init_task(coll_args, team);
    unsigned long nreqs = get_num_reqs(tl_team);
    ucc_rank_t tsize = UCC_TL_TEAM_SIZE(tl_team);
    size_t count = coll_args->args.dst.info.count / tsize;
    size_t dt = coll_args->args.dst.info.datatype;
    nvcompBatchedANSOpts_t opts = {0};
    size_t tmp_size = 0;
    opts.data_type = ucc_to_nvcompans_dtype[dt];
    opts.type = nvcomp_rANS;

    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        ucc_tl_ucp_put_task(task);
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!UCC_IS_INPLACE(coll_args->args)) {
        if (UCC_TL_UCP_TEAM_CTX(tl_team)->cfg.local_copy_type ==
            UCC_TL_UCP_LOCAL_COPY_TYPE_EC) {
            task->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
        }
    }

    nvcompBatchedANSCompressGetTempSizeEx(1, count * ucc_dt_size(dt), opts, &tmp_size, count * ucc_dt_size(dt));
    cudaHostAlloc((void**)&task->allgather_linear_comp.metadata, sizeof(ucc_tl_ucp_compress_metadata_t), cudaHostAllocDefault);
    cudaHostAlloc((void**)&task->allgather_linear_comp.metadata->device_uncompressed_chunk_ptrs, sizeof(void*) * tsize, cudaHostAllocDefault);
    cudaHostAlloc((void**)&task->allgather_linear_comp.metadata->device_uncompressed_chunk_bytes, sizeof(size_t) * tsize, cudaHostAllocDefault);
    cudaHostAlloc((void**)&task->allgather_linear_comp.metadata->device_compressed_chunk_ptrs, sizeof(void*) * tsize, cudaHostAllocDefault);
    cudaHostAlloc((void**)&task->allgather_linear_comp.metadata->device_compressed_chunk_bytes, sizeof(size_t) * tsize, cudaHostAllocDefault);
    cudaHostAlloc((void**)&task->allgather_linear_comp.metadata->status, sizeof(nvcompStatus_t) * tsize, cudaHostAllocDefault);
    task->allgather_linear_comp.metadata->tmp_size = tmp_size;

    ucc_mc_alloc(&task->allgather_linear_comp.src_compressed,
                 count * ucc_dt_size(dt), UCC_MEMORY_TYPE_CUDA);
    ucc_mc_alloc(&task->allgather_linear_comp.dst_compressed,
                 count * ucc_dt_size(dt) * tsize, UCC_MEMORY_TYPE_CUDA);
    ucc_mc_alloc(&task->allgather_linear_comp.tmp_memory,
                 tmp_size, UCC_MEMORY_TYPE_CUDA);
    task->super.post     = ucc_tl_ucp_allgather_linear_comp_start;
    task->super.progress = ucc_tl_ucp_allgather_linear_comp_progress;
    task->allgather_linear.nreqs =
        nreqs == 0 ? UCC_TL_TEAM_SIZE(tl_team) - 1 : nreqs;
    *task_h = &task->super;

    return UCC_OK;
}
