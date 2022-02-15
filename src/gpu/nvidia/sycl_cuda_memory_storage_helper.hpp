/*******************************************************************************
* Copyright 2022 Intel Corporation
* Copyright 2022 Codeplay Software Limited
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

#ifndef GPU_NVIDIA_SYCL_CUDA_MEMORY_STORAGE_HELPER_HPP
#define GPU_NVIDIA_SYCL_CUDA_MEMORY_STORAGE_HELPER_HPP

#include <optional>
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "sycl/sycl_memory_storage.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

#define CTX_IN_SYCL_MEMORY(arg) \
    sycl_memory_arg_t<::sycl::access::mode::read>( \
            static_cast<sycl::sycl_memory_storage_base_t *>( \
                    &CTX_IN_STORAGE(arg)), \
            cgh)

#define CTX_OUT_SYCL_MEMORY(arg) \
    sycl_memory_arg_t<::sycl::access::mode::write>( \
            static_cast<sycl::sycl_memory_storage_base_t *>( \
                    &CTX_OUT_STORAGE(arg)), \
            cgh)

template <::sycl::access_mode mode>
class sycl_memory_arg_t {
public:
    sycl_memory_arg_t() = default;
    sycl_memory_arg_t(
            sycl::sycl_memory_storage_base_t *mem, ::sycl::handler &cgh) {
        switch (mem->memory_kind()) {
            case sycl::memory_kind::buffer:
                acc_.emplace(
                        utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                                mem)
                                ->buffer(),
                        cgh);
                break;
            case sycl::memory_kind::usm:
                raw_ptr_ = utils::downcast<
                        const sycl::sycl_usm_memory_storage_t *>(mem)
                                   ->usm_ptr();
                break;
            default: assert(!"unexpected memory kind");
        }
    }

    template <::sycl::backend be = ::sycl::backend::ext_oneapi_cuda,
            typename T = void>
    T *get_native_pointer(const compat::interop_handle &ih) const {
        void *raw_ptr;
        if (acc_.has_value()) {
            raw_ptr = reinterpret_cast<void *>(ih.get_native_mem<be>(acc_.value()));
        } else {
            raw_ptr = raw_ptr_;
        }
        return reinterpret_cast<T *>(raw_ptr);
    }

    template <::sycl::backend be = ::sycl::backend::ext_oneapi_cuda,
            typename T = void>
    T *get_native_pointer(
            const compat::interop_handle &ih, size_t offset) const {
        return reinterpret_cast<T *>(
                reinterpret_cast<uint8_t *>(get_native_pointer<be, T>(ih))
                + offset);
    }

private:
    void *raw_ptr_ = nullptr;
    std::optional<::sycl::accessor<uint8_t, 1, mode>> acc_;
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
