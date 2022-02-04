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
        auto *x_mem = CTX_IN_MEMORY(DNNL_ARG_SRC);
        auto *weights_mem = CTX_IN_MEMORY(DNNL_ARG_WEIGHTS);
        auto *y_mem = CTX_OUT_MEMORY(DNNL_ARG_DST);

        sycl::sycl_memory_storage_base_t *bias_mem;
        sycl::sycl_memory_storage_base_t *temp_dst_mem;
        sycl::sycl_memory_storage_base_t *temp_reorder_mem;

        std::shared_ptr<scratch_acc_t> scratch_acc;
        std::shared_ptr<scratch_acc_t> filter_scratch_acc;

        std::optional<::sycl::accessor<uint8_t, 1, ::sycl::access::mode::read>>
                bias_acc;
        std::optional<scratch_acc_t> temp_dst_acc;
        std::optional<scratch_acc_t> temp_reorder_acc;

        const bool use_temp_dst = pd()->use_temp_dst();

        auto x_acc = CTX_IN_OPTIONAL_ACCESSOR(DNNL_ARG_SRC, x_mem);
        auto weights_acc
                = CTX_IN_OPTIONAL_ACCESSOR(DNNL_ARG_WEIGHTS, weights_mem);
        auto y_acc = CTX_OUT_OPTIONAL_ACCESSOR(DNNL_ARG_DST, y_mem);

        if (with_scratchpad) {
            scratch_acc = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                    memory_tracking::names::key_conv_cudnn_algo));
        }
        if (with_bias) {
            bias_mem = CTX_IN_MEMORY(DNNL_ARG_BIAS);
            bias_acc = CTX_IN_OPTIONAL_ACCESSOR(DNNL_ARG_BIAS, bias_mem);
        }
        if (pd()->impl_->using_transformed_filter()) {
            filter_scratch_acc
                    = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                            memory_tracking::names::key_conv_cudnn_filter));
        }

        if (use_temp_dst) {
            temp_dst_mem = utils::downcast<sycl::sycl_memory_storage_base_t *>(
                    scratch_storage.get());
            temp_reorder_mem
                    = utils::downcast<sycl::sycl_memory_storage_base_t *>(
                            scratch_storage_2.get());
            temp_dst_acc = get_cudnn_accessor<scratch_acc_t>(temp_dst_mem, cgh);
            temp_reorder_acc
                    = get_cudnn_accessor<scratch_acc_t>(temp_reorder_mem, cgh);
        }

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            std::vector<void *> args;
            args.push_back(get_cudnn_ptr(sc, ih, x_acc, x_mem));
            args.push_back(get_cudnn_ptr(sc, ih, weights_acc, weights_mem));
            args.push_back(get_cudnn_ptr(sc, ih, y_acc, y_mem));
            args.push_back(with_bias ? get_cudnn_ptr(sc, ih, bias_acc, bias_mem)
                                     : nullptr);
            args.push_back(with_scratchpad
                            ? sc.memory<void *>(ih, *scratch_acc.get())
                            : nullptr);
            args.push_back(pd()->impl_->using_transformed_filter()
                            ? sc.memory<void *>(ih, *filter_scratch_acc.get())
                            : nullptr);
            args.push_back(use_temp_dst
                            ? get_cudnn_ptr(sc, ih, temp_dst_acc, temp_dst_mem)
                            : nullptr);
            args.push_back(use_temp_dst ? get_cudnn_ptr(
                                   sc, ih, temp_reorder_acc, temp_reorder_mem)
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

        auto *x_mem = CTX_OUT_MEMORY(DNNL_ARG_DIFF_SRC);
        auto *weights_mem = CTX_IN_MEMORY(DNNL_ARG_WEIGHTS);
        auto *y_mem = CTX_IN_MEMORY(DNNL_ARG_DIFF_DST);

        sycl::sycl_memory_storage_base_t *bias_mem;

        std::optional<::sycl::accessor<uint8_t, 1, ::sycl::access::mode::read>>
                bias_acc;
        std::shared_ptr<scratch_acc_t> scratch_acc;
        std::shared_ptr<scratch_acc_t> filter_scratch_acc;

        auto x_acc = CTX_OUT_OPTIONAL_ACCESSOR(DNNL_ARG_DIFF_SRC, x_mem);
        auto weights_acc
                = CTX_IN_OPTIONAL_ACCESSOR(DNNL_ARG_WEIGHTS, weights_mem);
        auto y_acc = CTX_IN_OPTIONAL_ACCESSOR(DNNL_ARG_DIFF_DST, y_mem);

        if (with_scratchpad) {
            scratch_acc = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                    memory_tracking::names::key_conv_cudnn_algo));
        }
        if (with_bias) {
            bias_mem = CTX_IN_MEMORY(DNNL_ARG_BIAS);
            bias_acc = CTX_IN_OPTIONAL_ACCESSOR(DNNL_ARG_BIAS, bias_mem);
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
            args.push_back(get_cudnn_ptr(sc, ih, x_acc, x_mem));
            args.push_back(get_cudnn_ptr(sc, ih, weights_acc, weights_mem));
            args.push_back(get_cudnn_ptr(sc, ih, y_acc, y_mem));
            args.push_back(with_bias ? get_cudnn_ptr(sc, ih, bias_acc, bias_mem)
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
status_t cudnn_convolution_bwd_weights_t::execute_zero_dims(
        const exec_ctx_t &ctx) const {
    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](::sycl::handler &cgh) {
        auto *weights_mem = CTX_OUT_MEMORY(DNNL_ARG_DIFF_WEIGHTS);
        auto weights_acc
                = CTX_OUT_OPTIONAL_ACCESSOR(DNNL_ARG_DIFF_WEIGHTS, weights_mem);
        sycl::sycl_memory_storage_base_t *bias_mem;
        std::optional<::sycl::accessor<uint8_t, 1, ::sycl::access::mode::write>>
                bias_acc;
        if (pd()->with_bias()) {
            bias_mem = CTX_OUT_MEMORY(DNNL_ARG_DIFF_BIAS);
            bias_acc = CTX_OUT_OPTIONAL_ACCESSOR(DNNL_ARG_DIFF_BIAS, bias_mem);
        }
        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            auto weights = get_cudnn_ptr(sc, ih, weights_acc, weights_mem);
            void *bias = nullptr;
            if (pd()->with_bias())
                bias = get_cudnn_ptr(sc, ih, bias_acc, bias_mem);
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
        auto *x_mem = CTX_IN_MEMORY(DNNL_ARG_SRC);
        auto *weights_mem = CTX_OUT_MEMORY(DNNL_ARG_DIFF_WEIGHTS);
        auto *y_mem = CTX_IN_MEMORY(DNNL_ARG_DIFF_DST);

        sycl::sycl_memory_storage_base_t *bias_mem;

        std::optional<::sycl::accessor<uint8_t, 1, ::sycl::access::mode::write>>
                bias_acc;
        std::shared_ptr<scratch_acc_t> scratch_acc;
        std::shared_ptr<scratch_acc_t> filter_scratch_acc;

        auto x_acc = CTX_IN_OPTIONAL_ACCESSOR(DNNL_ARG_SRC, x_mem);
        auto weights_acc
                = CTX_OUT_OPTIONAL_ACCESSOR(DNNL_ARG_DIFF_WEIGHTS, weights_mem);
        auto y_acc = CTX_IN_OPTIONAL_ACCESSOR(DNNL_ARG_DIFF_DST, y_mem);

        if (with_scratchpad) {
            scratch_acc = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                    memory_tracking::names::key_conv_cudnn_algo));
        }
        if (with_bias) {
            bias_mem = CTX_OUT_MEMORY(DNNL_ARG_DIFF_BIAS);
            bias_acc = CTX_OUT_OPTIONAL_ACCESSOR(DNNL_ARG_DIFF_BIAS, bias_mem);
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
            args.push_back(get_cudnn_ptr(sc, ih, x_acc, x_mem));
            args.push_back(get_cudnn_ptr(sc, ih, weights_acc, weights_mem));
            args.push_back(get_cudnn_ptr(sc, ih, y_acc, y_mem));
            args.push_back(with_bias ? get_cudnn_ptr(sc, ih, bias_acc, bias_mem)
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
