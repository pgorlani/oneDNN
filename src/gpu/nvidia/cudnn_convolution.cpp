/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
* Copyright 2020 Codeplay Software Limited
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/nvidia/cudnn_convolution.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"
#include "sycl_cuda_memory_storage_helper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t cudnn_convolution_fwd_t::execute_convolution(
        const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const {
    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](::sycl::handler &cgh) {
        using scratch_acc_t = ::sycl::accessor<uint8_t, 1,
                ::sycl::access::mode::read_write>;

        auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
        auto arg_weights = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
        auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);

        sycl_memory_arg<::sycl::access::mode::read> arg_bias;
        sycl::sycl_memory_storage_base_t *temp_dst_mem;
        sycl::sycl_memory_storage_base_t *temp_reorder_mem;

        std::shared_ptr<scratch_acc_t> scratch_acc;
        std::shared_ptr<scratch_acc_t> filter_scratch_acc;

        const bool use_temp_dst = pd()->use_temp_dst();

        if (with_scratchpad) {
            scratch_acc = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                    memory_tracking::names::key_conv_cudnn_algo));
        }
        if (with_bias) { arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS); }
        if (pd()->impl_->using_transformed_filter()) {
            filter_scratch_acc
                    = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                            memory_tracking::names::key_conv_cudnn_filter));
        }

        sycl_memory_arg<::sycl::access::mode::read_write> temp_dst;
        sycl_memory_arg<::sycl::access::mode::read_write> temp_reorder;

        if (use_temp_dst) {
            temp_dst_mem = utils::downcast<sycl::sycl_memory_storage_base_t *>(
                    scratch_storage.get());
            temp_reorder_mem
                    = utils::downcast<sycl::sycl_memory_storage_base_t *>(
                            scratch_storage_2.get());
            temp_dst = sycl_memory_arg<::sycl::access::mode::read_write>(
                    temp_dst_mem, cgh);
            temp_reorder = sycl_memory_arg<::sycl::access::mode::read_write>(
                    temp_reorder_mem, cgh);
        }

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            std::vector<void *> args;
            args.push_back(arg_src.get_native_pointer(ih, sc));
            args.push_back(arg_weights.get_native_pointer(ih, sc));
            args.push_back(arg_dst.get_native_pointer(ih, sc));
            args.push_back(
                    with_bias ? arg_bias.get_native_pointer(ih, sc) : nullptr);

            args.push_back(with_scratchpad
                            ? sc.memory<void *>(ih, *scratch_acc.get())
                            : nullptr);
            args.push_back(pd()->impl_->using_transformed_filter()
                            ? sc.memory<void *>(ih, *filter_scratch_acc.get())
                            : nullptr);
            args.push_back(use_temp_dst ? temp_dst.get_native_pointer(ih, sc)
                                        : nullptr);
            args.push_back(use_temp_dst
                            ? temp_reorder.get_native_pointer(ih, sc)
                            : nullptr);
            pd()->impl_->execute(handle, args);
        });
    });
}

status_t cudnn_convolution_bwd_data_t::execute_convolution(
        const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const {
    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](::sycl::handler &cgh) {
        using scratch_acc_t = ::sycl::accessor<uint8_t, 1,
                ::sycl::access::mode::read_write>;

        auto arg_diff_src = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_SRC);
        auto arg_weights = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
        auto arg_diff_dst = CTX_IN_SYCL_MEMORY(DNNL_ARG_DIFF_DST);

        sycl_memory_arg<::sycl::access::mode::read> arg_bias;

        std::shared_ptr<scratch_acc_t> scratch_acc;
        std::shared_ptr<scratch_acc_t> filter_scratch_acc;

        if (with_scratchpad) {
            scratch_acc = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                    memory_tracking::names::key_conv_cudnn_algo));
        }
        if (with_bias) { arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS); }
        if (pd()->impl_->using_transformed_filter()) {
            filter_scratch_acc
                    = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                            memory_tracking::names::key_conv_cudnn_filter));
        }
        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            std::vector<void *> args;
            args.push_back(arg_diff_src.get_native_pointer(ih, sc));
            args.push_back(arg_weights.get_native_pointer(ih, sc));
            args.push_back(arg_diff_dst.get_native_pointer(ih, sc));
            args.push_back(
                    with_bias ? arg_bias.get_native_pointer(ih, sc) : nullptr);
            args.push_back(with_scratchpad
                            ? sc.memory<void *>(ih, *scratch_acc.get())
                            : nullptr);
            args.push_back(pd()->impl_->using_transformed_filter()
                            ? sc.memory<void *>(ih, *filter_scratch_acc.get())
                            : nullptr);
            pd()->impl_->execute(handle, args);
        });
    });
}

status_t cudnn_convolution_bwd_weights_t::execute_zero_dims(
        const exec_ctx_t &ctx) const {
    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_diff_weights = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_WEIGHTS);

        sycl_memory_arg<::sycl::access::mode::write> arg_diff_bias;

        if (pd()->with_bias()) {
            arg_diff_bias = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_BIAS);
        }
        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            void *weights = arg_diff_weights.get_native_pointer(ih, sc);
            void *bias = nullptr;
            if (pd()->with_bias())
                bias = arg_diff_bias.get_native_pointer(ih, sc);
            pd()->impl_->execute_set_weights_bias(handle, weights, bias, 0.f);
        });
    });
}

status_t cudnn_convolution_bwd_weights_t::execute_convolution(
        const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const {
    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](::sycl::handler &cgh) {
        using scratch_acc_t = ::sycl::accessor<uint8_t, 1,
                ::sycl::access::mode::read_write>;
        auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
        auto arg_diff_weights = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_WEIGHTS);
        auto arg_diff_dst = CTX_IN_SYCL_MEMORY(DNNL_ARG_DIFF_DST);

        sycl_memory_arg<::sycl::access::mode::write> arg_diff_bias;

        std::shared_ptr<scratch_acc_t> scratch_acc;
        std::shared_ptr<scratch_acc_t> filter_scratch_acc;

        if (with_scratchpad) {
            scratch_acc = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                    memory_tracking::names::key_conv_cudnn_algo));
        }
        if (with_bias) {
            arg_diff_bias = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_BIAS);
        }
        if (pd()->impl_->using_transformed_filter()) {
            filter_scratch_acc
                    = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                            memory_tracking::names::key_conv_cudnn_filter));
        }

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            std::vector<void *> args;
            args.push_back(arg_src.get_native_pointer(ih, sc));
            args.push_back(arg_diff_weights.get_native_pointer(ih, sc));
            args.push_back(arg_diff_dst.get_native_pointer(ih, sc));
            args.push_back(with_bias ? arg_diff_bias.get_native_pointer(ih, sc)
                                     : nullptr);
            args.push_back(with_scratchpad
                            ? sc.memory<void *>(ih, *scratch_acc.get())
                            : nullptr);
            args.push_back(pd()->impl_->using_transformed_filter()
                            ? sc.memory<void *>(ih, *filter_scratch_acc.get())
                            : nullptr);
            pd()->impl_->execute(handle, args);
        });
    });
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
