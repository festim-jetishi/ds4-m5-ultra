// DeepSeek V4.1 keeps the RoPE tail in the quantized KV vector. Unlike V4,
// indexer Q/K have no Hadamard transform, and compressed KV has E4M3 scales.
struct ds4_metal_args_dsv41_quantize {
    uint width;
    uint rows;
    uint mode;
};

static inline float dsv41_bf16(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}

static inline float dsv41_pow2_ceil(float x) {
    const uint bits = as_type<uint>(x);
    return as_type<float>((bits & 0x7f800000u) +
                         ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

// Word copy inside the compute encoder: the same bytes as a blit copy,
// without ending the compute pass for a blit pass and starting a new one.
kernel void kernel_ds4_copy_words(
        constant uint &count,
        device const uint *src,
        device uint *dst,
        uint gid [[thread_position_in_grid]]) {
    if (gid < count) dst[gid] = src[gid];
}

kernel void kernel_dsv41_bf16_linear(
        constant ulong &count,
        device uint *x,
        uint gid [[thread_position_in_grid]]) {
    const ulong first = (ulong)gid * 4u;
    if (first + 4u <= count) {
        uint4 bits = *((device uint4 *)(x + first));
        const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
        bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
        *((device uint4 *)(x + first)) = bits & 0xffff0000u;
    } else {
        for (ulong i = first; i < count; i++) {
            uint bits = x[i];
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16u) & 1u);
            x[i] = bits & 0xffff0000u;
        }
    }
}

struct ds4_metal_args_dsv41_rope {
    uint width, heads, rows, start, inverse, stride;
    float frequencies[32];
};

kernel void kernel_dsv41_rope_rows(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        constant uint *pos,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(pos[group.y]) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong i = ((ulong)group.y * args.heads + group.x) * args.width +
                    args.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

// The BF16 pass over whole heads followed by the RoPE of their last 64
// values, in one dispatch: same rounding and rotation as kernel_dsv41_bf16_linear
// then kernel_dsv41_rope / kernel_dsv41_rope_rows. ROWS reads per-row positions.
template <bool ROWS>
static inline void dsv41_bf16_rope_impl(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        constant uint *pos,
        uint2 group,
        uint lane) {
    const ulong base = ((ulong)group.y * args.heads + group.x) * args.width;
    const uint rope0 = args.width - 64u;
    for (uint j = lane; j < rope0; j += 32u) x[base + j] = dsv41_bf16(x[base + j]);
    const float theta = float(ROWS ? pos[group.y] : args.start + group.y * args.stride) *
        args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong i = base + rope0 + 2u * lane;
    const float re = dsv41_bf16(x[i]), im = dsv41_bf16(x[i + 1u]);
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

kernel void kernel_dsv41_bf16_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    dsv41_bf16_rope_impl<false>(args, x, nullptr, group, lane);
}

kernel void kernel_dsv41_bf16_rope_rows(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        constant uint *pos,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    dsv41_bf16_rope_impl<true>(args, x, pos, group, lane);
}

// out = BF16(a + b): kernel_add2_f32 followed by the BF16 pass.
kernel void kernel_dsv41_add_bf16(
        constant uint &n,
        device const float *a,
        device const float *b,
        device float *out,
        uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = dsv41_bf16(a[i] + b[i]);
}

// Greedy argmax of one logits row, as the host's sample_argmax: the first
// index of the largest value (NaN never wins; none at all gives 0). One
// threadgroup of 1024 threads; out[slot] receives the index.
kernel void kernel_dsv41_argmax_f32(
        constant uint &n,
        constant uint &slot,
        device const float *x,
        device int *out,
        threadgroup float *sv [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    float bv = -INFINITY;
    int bi = -1;
    for (uint i = tid; i < n; i += 1024u) {
        const float v = x[i];
        if (v > bv) { bv = v; bi = (int)i; }
    }
    for (ushort d = 16; d > 0; d >>= 1) {
        const float ov = simd_shuffle_down(bv, d);
        const int oi = simd_shuffle_down(bi, d);
        if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
    }
    threadgroup int *si = (threadgroup int *)(sv + 32);
    if (lane == 0) { sv[sg] = bv; si[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        bv = sv[lane];
        bi = si[lane];
        for (ushort d = 16; d > 0; d >>= 1) {
            const float ov = simd_shuffle_down(bv, d);
            const int oi = simd_shuffle_down(bi, d);
            if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
        }
        if (lane == 0) out[slot] = bi < 0 ? 0 : bi;
    }
}

// DSpark Markov input: the F32 markov_w1 row of the previous token (tokens[slot]
// when tokens is bound, else token) into the embedding and the tail of the
// confidence input.
kernel void kernel_dsv41_markov_embed_f32(
        constant uint &rank,
        constant uint &vocab,
        constant uint &slot,
        constant int &token,
        constant uint &use_tokens,
        device const float *w1,
        device const int *tokens,
        device float *embed,
        device float *conf_tail,
        uint i [[thread_position_in_grid]]) {
    if (i >= rank) return;
    int t = use_tokens ? tokens[slot] : token;
    if (t < 0 || (uint)t >= vocab) t = 0;
    const float v = w1[(ulong)t * rank + i];
    embed[i] = v;
    conf_tail[i] = v;
}

kernel void kernel_dsv41_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(args.start + group.y * args.stride) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong i = ((ulong)group.y * args.heads + group.x) * args.width +
                    args.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

kernel void kernel_dsv41_quantize(
        constant ds4_metal_args_dsv41_quantize &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    const ulong index = (ulong)group.y * args.width + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    if (valid) x[index] = dsv41_bf16(result);
}

struct ds4_metal_args_dsv41_engram {
    uint width;
    uint rows;
    float eps;
    uint masked;
};

kernel void kernel_dsv41_engram_add(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (args.masked && !mask[group.x]) return;
    const ulong offset = ((ulong)group.x * 4u + group.y) * args.width;
    const ulong key_offset = ((ulong)group.x * 5u + group.y) * args.width;
    const ulong value_offset = ((ulong)group.x * 5u + 4u) * args.width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint i = lane; i < args.width; i += 32u) {
        const float h = residual[offset + i];
        const float k = dsv41_bf16(kv[key_offset + i]);
        const uint wi = group.y * args.width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (q_weight[wi] * k_weight[wi]) * k;
    }
    h2 = simd_sum(h2);
    k2 = simd_sum(k2);
    dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
          rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
    const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
    for (uint i = lane; i < args.width; i += 32u)
        residual[offset + i] = dsv41_bf16(residual[offset + i] +
            gate * dsv41_bf16(kv[value_offset + i]));
}

struct ds4_metal_args_dsv41_pool {
    uint width;
    uint pairs;
    uint tail;
};

kernel void kernel_dsv41_pool2(
        constant ds4_metal_args_dsv41_pool &args,
        device float *out,
        device const float *kv,
        device const float *scores,
        device const float *previous_kv,
        device const float *previous_scores,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.pairs) return;
    const long a = (long)index.y * 2 - args.tail;
    const ulong b = (ulong)(a + 1) * args.width + index.x;
    const float ka = a < 0 ? previous_kv[index.x] : kv[(ulong)a * args.width + index.x];
    const float sa = a < 0 ? previous_scores[index.x] : scores[(ulong)a * args.width + index.x];
    const float sb = scores[b], peak = max(sa, sb);
    const float ea = exp(sa - peak), eb = exp(sb - peak);
    out[(ulong)index.y * args.width + index.x] = dsv41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
}

struct ds4_metal_args_dsv41_candidates {
    uint width;
    uint rows;
    uint start;
    uint ratio;
};

kernel void kernel_dsv41_candidate_blocks(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *blocks,
        device const float *unused,
        uint2 index [[thread_position_in_grid]]) {
    (void)unused;
    const uint count = (args.width + 7u) / 8u;
    if (index.x >= count || index.y >= args.rows) return;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    float best = -INFINITY;
    for (uint i = index.x * 8u; i < min(visible, (index.x + 1u) * 8u); i++)
        best = max(best, scores[(ulong)index.y * args.width + i]);
    if (visible && index.x == (visible - 1u) / 8u) best = INFINITY;
    blocks[(ulong)index.y * count + index.x] = best;
}

kernel void kernel_dsv41_candidate_filter(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *out,
        device const float *block_mask,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.rows) return;
    const ulong offset = (ulong)index.y * args.width + index.x;
    const uint blocks = (args.width + 7u) / 8u;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    out[offset] = index.x < visible &&
        block_mask[(ulong)index.y * blocks + index.x / 8u] == 0.0f
        ? scores[offset] : -INFINITY;
}

struct ds4_metal_args_dsv41_carry {
    uint width, rows, words, format, pack;
};

kernel void kernel_dsv41_carry_copy(
        constant ds4_metal_args_dsv41_carry &args,
        device uint *packed, device float *plain,
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint col = group.x * 128u + tid;
    const ulong row = group.y;
    if (args.format == 0u) {
        if (col >= args.width) return;
        device ushort *p = (device ushort *)(packed + row * args.words);
        if (args.pack) p[col] = ushort(as_type<uint>(plain[row * args.width + col]) >> 16);
        else plain[row * args.width + col] = as_type<float>(uint(p[col]) << 16);
    } else {
        const uint word = col / 32u;
        if (args.pack) {
            const bool allowed = col < args.width && plain[row * args.width + col] == 0.0f;
            const uint bits = simd_sum(allowed ? 1u << lane : 0u);
            if (!lane && word < args.words) packed[row * args.words + word] = bits;
        } else if (col < args.width) {
            const uint bits = packed[row * args.words + word];
            plain[row * args.width + col] = bits & (1u << lane) ? 0.0f : -INFINITY;
        }
    }
}

#ifdef DS4_METAL_HAS_TENSOR
kernel void kernel_dsv41_indexer_pack(
        constant uint4 &args,
        device const float *q, device const float *keys,
        device uint *flags, device bfloat *packed_q, device bfloat *packed_keys,
        threadgroup uint *valid [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const bool query = group < args.y;
    const uint count = query ? 32u * 128u : 64u * 128u;
    const ulong offset = query ? (ulong)group * count : (ulong)(group - args.y) * count;
    bool exact = true;
    for (uint i = tid; i < count; i += 128u) {
        const float value = query ? q[offset + i] :
            offset + i < (ulong)args.x * 128u ? keys[offset + i] : 0.0f;
        const bfloat converted = bfloat(value);
        if (query) packed_q[offset + i] = converted;
        else packed_keys[offset + i] = converted;
        exact = exact && float(converted) == value;
    }
    const bool same = simd_all(exact);
    if (!(tid % 32u)) valid[tid / 32u] = same;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) flags[group] = valid[0] && valid[1] && valid[2] && valid[3];
}

kernel void kernel_dsv41_indexer_scores_packed(
        constant uint4 &args, constant uint2 &range,
        device const float *q, device const float *weights,
        device const float *keys, device float *scores,
        device const uint *flags, device const bfloat *packed_q,
        device const bfloat *packed_keys,
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int HEADS = 32, KEYS = 64, DIM = 128;
    const uint width = args.x, token = group.y, row0 = group.x * KEYS;
    const uint ratio = args.w & 0xffu;
    const bool keys_bf16 = (args.w >> 8u) != 0u;
    const uint visible = (args.z + token + 1u) / ratio;
    if (row0 >= visible) {
        if (tid < KEYS && row0 + tid < width)
            scores[(ulong)token * width + row0 + tid] = -INFINITY;
        return;
    }
    matmul2d<matmul2d_descriptor(HEADS, KEYS, DIM, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto result = tensor(scratch, dextents<int32_t, 2>(KEYS, HEADS));
    if (keys_bf16 && flags[range.y + token]) {
        auto query = tensor((device bfloat *)packed_q + (ulong)(range.y + token) * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto all_keys = tensor((device bfloat *)keys, dextents<int32_t, 2>(DIM, int(width)));
        auto key = all_keys.slice(0, int(row0));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    } else if (!keys_bf16 && flags[range.y + token] && flags[range.x + group.x]) {
        auto query = tensor((device bfloat *)packed_q + (ulong)(range.y + token) * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto key = tensor((device bfloat *)packed_keys + (ulong)row0 * DIM,
                           dextents<int32_t, 2>(DIM, KEYS));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    } else if (keys_bf16) {
        auto query = tensor((device float *)q + (ulong)token * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto all_keys = tensor((device bfloat *)keys, dextents<int32_t, 2>(DIM, int(width)));
        auto key = all_keys.slice(0, int(row0));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    } else {
        auto query = tensor((device float *)q + (ulong)token * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto all_keys = tensor((device float *)keys, dextents<int32_t, 2>(DIM, int(width)));
        auto key = all_keys.slice(0, int(row0));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEYS && row0 + tid < width) {
        float sum = 0;
        for (uint h = 0; h < HEADS; h++)
            sum += max(scratch[h * KEYS + tid] * (1.0f / 64.0f), 0.0f) * weights[token * HEADS + h];
        scores[(ulong)token * width + row0 + tid] = row0 + tid < visible ? sum : -INFINITY;
    }
}
#endif

struct ds4_metal_args_dsv41_stage_rows {
    uint n_rows;
    uint n_keys;
    uint raw_cap;
    uint sel_stride;
    uint comp_direct;
    uint comp_f16;
    uint n_raw[8];
    uint raw_start[8];
    uint first_pos[8];
    uint new_from[8];
    uint new_row0[8];
    uint n_comp[8];
};

// FlashAttention K/V staging for 1-8 decode rows in one dispatch: each row's
// raw keys (ring order) then its compressed keys, converted to F16 exactly as
// the copy kernels do. Raw positions from new_from on come from the batch kv
// rows (the window writes for this batch happen after staging), and
// compressed keys are gathered by the row's selection (or read in order when
// comp_direct). Grid: x = key vectors of a row, y = row.
kernel void kernel_dsv41_stage_rows_f16(
        constant ds4_metal_args_dsv41_stage_rows & args [[buffer(0)]],
        device const char * kv_rows [[buffer(1)]],
        device const int  * sel     [[buffer(2)]],
        device       char * dst     [[buffer(3)]],
        device const char * w0 [[buffer(4)]],  device const char * w1 [[buffer(5)]],
        device const char * w2 [[buffer(6)]],  device const char * w3 [[buffer(7)]],
        device const char * w4 [[buffer(8)]],  device const char * w5 [[buffer(9)]],
        device const char * w6 [[buffer(10)]], device const char * w7 [[buffer(11)]],
        device const char * c0 [[buffer(12)]], device const char * c1 [[buffer(13)]],
        device const char * c2 [[buffer(14)]], device const char * c3 [[buffer(15)]],
        device const char * c4 [[buffer(16)]], device const char * c5 [[buffer(17)]],
        device const char * c6 [[buffer(18)]], device const char * c7 [[buffer(19)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint r = gid.y;
    const uint v = gid.x;
    if (r >= args.n_rows || v >= args.n_keys * 128u) return;
    const uint k = v >> 7;
    const uint col = v & 127u;
    float4 value;
    if (k < args.n_raw[r]) {
        if (args.first_pos[r] + k >= args.new_from[r]) {
            const uint row = args.new_row0[r] + (args.first_pos[r] + k - args.new_from[r]);
            value = float4(((device const packed_float4 *)kv_rows)[row * 128u + col]);
        } else {
            device const char * w = r == 0 ? w0 : r == 1 ? w1 : r == 2 ? w2 : r == 3 ? w3 :
                                    r == 4 ? w4 : r == 5 ? w5 : r == 6 ? w6 : w7;
            uint slot = args.raw_start[r] + k;
            if (slot >= args.raw_cap) slot -= args.raw_cap;
            value = float4(((device const packed_float4 *)w)[slot * 128u + col]);
        }
    } else {
        const uint i = k - args.n_raw[r];
        if (i >= args.n_comp[r]) {
            /* padding up to the batch key count: zero keys, masked later */
            ((device packed_half4 *)dst)[(r * args.n_keys + k) * 128u + col] = packed_half4(half4(0.0h));
            return;
        }
        device const char * c = r == 0 ? c0 : r == 1 ? c1 : r == 2 ? c2 : r == 3 ? c3 :
                                r == 4 ? c4 : r == 5 ? c5 : r == 6 ? c6 : c7;
        const uint src = args.comp_direct ? i : (uint)sel[r * args.sel_stride + i];
        if (args.comp_f16) {
            ((device packed_half4 *)dst)[(r * args.n_keys + k) * 128u + col] =
                ((device const packed_half4 *)c)[src * 128u + col];
            return;
        }
        value = float4(((device const packed_float4 *)c)[src * 128u + col]);
    }
    ((device packed_half4 *)dst)[(r * args.n_keys + k) * 128u + col] = packed_half4(half4(value));
}

// ds4_flash_attn_vec_packed8_reduce_f16_512 with NSG physical simdgroups
// (only which simdgroup runs which virtual split changes).
template <short NSG>
static inline void dsv41_flash_attn_packed_f16_512(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        threadgroup  char * shmem,
        uint head,
        int ne11,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NW = 32;
    constexpr short C = 32;
    constexpr short NWG = 32;
    constexpr short DK4 = 128;
    constexpr short DV4 = 128;
    constexpr short SH = 128;

    /* 24,448 dynamic bytes: shared Q, eight score/mask banks, all 32
     * split-local weights and stats, sink scales, and a padded 32x33 F32
     * float4 partial plane. */
    threadgroup half4 *q_shared = (threadgroup half4 *)shmem;
    threadgroup half *score_banks =
        (threadgroup half *)(q_shared + DK4);
    threadgroup volatile float *weights =
        (threadgroup volatile float *)(score_banks + NSG * SH);
    threadgroup volatile float *stats = weights + NWG * C;
    threadgroup volatile float *sink_scale = stats + 2 * NWG;
    threadgroup volatile float4 *partial_plane =
        (threadgroup volatile float4 *)(sink_scale + NWG);

    const short lane = (short)tiisg;
    threadgroup half *bank = score_banks + (short)sgitg * SH;
    threadgroup float *ss = (threadgroup float *)bank;
    threadgroup half *sm = bank + 2 * C;

    device const float4 *q4 =
        (device const float4 *)(q + (uint64_t)head * args.nb02);
    if (sgitg == 0) {
        for (short i = lane; i < DK4; i += NW) {
            q_shared[i] = (half4)q4[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Eight physical simdgroups time-slice the exact 32 legacy split-K
     * workgroups. The official <=1024-key gate gives each virtual split at
     * most one 32-row block, so its value partial can be formed later from
     * these materialized weights without changing online-softmax order. */
    for (short iwg = (short)sgitg; iwg < NWG; iwg += NSG) {
        float S = 0.0f;
        float M = -FLT_MAX / 2;
        float out_scale = 1.0f;
        const int ic_original = (int)iwg * C;

        weights[(uint)iwg * C + (uint)lane] = 0.0f;
        ss[lane] = 0.0f;
        sm[lane] = (half)0.0h;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (ic_original < ne11) {
            device const char *k_block = k;
            device const char *v_block = v;
            device const half *pm = (device const half *)mask;
            int ic = ic_original;

            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > ne11) {
                k_block = pad;
                const uint64_t k_pad_bytes =
                    args.nb11 * (uint64_t)C *
                    (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
                const uint64_t v_pad_bytes =
                    args.nb21 * (uint64_t)C *
                    (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
                if (FC_flash_attn_ext_vec_shared_kvpad) {
                    v_block = k_block;
                    pm = (device const half *)(k_block +
                                                k_pad_bytes + v_pad_bytes);
                } else {
                    v_block = k_block + k_pad_bytes;
                    pm = (device const half *)(v_block + v_pad_bytes);
                }
                ic = 0;
            }

            /* A row staged with fewer keys than ne11's block masks the rest as
             * the pad kernel does (its keys are staged as zeros). */
            sm[lane] = FC_flash_attn_ext_vec_has_kvpad || ic_original + lane < ne11 ?
                pm[ic + lane] : (half)-MAXHALF;
            if (simd_max(sm[lane]) > -MAXHALF) {
                device const half4 *pk4 =
                    (device const half4 *)(k_block +
                                           (uint64_t)ic * args.nb11);
                threadgroup const half4 *pq4 = q_shared;
                pk4 += lane;
                pq4 += lane;

                float lane_mqk = 0.0f;
                FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                    float mqk = 0.0f;
                    FOR_UNROLL (short ii = 0; ii < DK4 / NW; ++ii) {
                        mqk += dot((float4)pk4[cc * DK4 + ii * NW],
                                   (float4)pq4[ii * NW]);
                    }
                    mqk = simd_sum(mqk);
                    if (lane == cc) {
                        lane_mqk = mqk;
                    }
                }

                ss[lane] = fma(lane_mqk, args.scale,
                               (float)sm[lane]);
                simdgroup_barrier(mem_flags::mem_threadgroup);

                const float old_m = M;
                const float score = ss[lane];
                M = simd_max(max(M, score));
                const float ms = exp(old_m - M);
                const float vs = exp(score - M);
                S = S * ms + simd_sum(vs);
                ss[lane] = vs;
                simdgroup_barrier(mem_flags::mem_threadgroup);

                weights[(uint)iwg * C + (uint)lane] = ss[lane];
            }

            if (FC_flash_attn_ext_vec_has_sinks && iwg == 0) {
                const float old_m = M;
                const float sink = lane == 0
                    ? ((device const float *)sinks)[head]
                    : -FLT_MAX / 2;
                M = simd_max(max(M, sink));
                const float ms = exp(old_m - M);
                const float vs = exp(sink - M);
                S = S * ms + simd_sum(vs);
                out_scale = ms;
            }
        } else if (FC_flash_attn_ext_vec_has_sinks && iwg == 0) {
            const float old_m = M;
            const float sink = lane == 0
                ? ((device const float *)sinks)[head]
                : -FLT_MAX / 2;
            M = simd_max(max(M, sink));
            const float ms = exp(old_m - M);
            const float vs = exp(sink - M);
            S = S * ms + simd_sum(vs);
            out_scale = ms;
        }

        if (lane == 0) {
            stats[2 * (uint)iwg + 0] = S;
            stats[2 * (uint)iwg + 1] = M;
            sink_scale[(uint)iwg] = out_scale;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Recreate the legacy 32-lane reducer exactly: lane is the virtual split
     * index, including neutral idle splits in their original tree positions. */
    const short split = lane;
    float reduce_S = stats[2 * (uint)split + 0];
    float reduce_M = stats[2 * (uint)split + 1];
    const float reduce_max = simd_max(reduce_M);
    const float reduce_ms = exp(reduce_M - reduce_max);
    reduce_S = simd_sum(reduce_S * reduce_ms);
    const float reduce_inv =
        reduce_S == 0.0f ? 0.0f : 1.0f / reduce_S;

    device float4 *dst4 =
        (device float4 *)(dst +
                          (uint64_t)head * 512u * sizeof(float));

    /* Form one 32-float4 output quadrant at a time. During production each
     * physical simdgroup time-slices four virtual splits while SIMD lanes are
     * contiguous output columns, exactly matching the legacy V loads and
     * cc-major accumulation. A padded 33-column plane avoids a 32-way TG-bank
     * conflict when the reducer transposes lanes back to virtual splits. */
    for (short quadrant = 0; quadrant < 4; ++quadrant) {
        for (short iwg = (short)sgitg; iwg < NWG; iwg += NSG) {
            float4 lo = float4(0.0f);
            const int ic_original = (int)iwg * C;
            if (ic_original < ne11) {
                device const char *v_block = v;
                int ic = ic_original;
                if (FC_flash_attn_ext_vec_has_kvpad && ic + C > ne11) {
                    device const char *k_block = pad;
                    const uint64_t k_pad_bytes =
                        args.nb11 * (uint64_t)C *
                        (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
                    if (FC_flash_attn_ext_vec_shared_kvpad) {
                        v_block = k_block;
                    } else {
                        v_block = k_block + k_pad_bytes;
                    }
                    ic = 0;
                }

                device const half4 *pv4 =
                    (device const half4 *)(v_block +
                                           (uint64_t)ic * args.nb21);
                threadgroup volatile float *split_weights =
                    weights + (uint)iwg * C;
                const short oc = quadrant * NW + lane;
                FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                    lo += float4(pv4[cc * DV4 + oc]) *
                          float4(split_weights[cc]);
                }

                float4 acc = float4(0.0f);
                acc += lo;
                if (iwg == 0) {
                    acc *= sink_scale[0];
                }
                lo = acc;
            }
            partial_plane[(uint)iwg * 33u + (uint)lane] = lo;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* lane is now the legacy split index and each physical simdgroup
         * reduces four output columns through the identical simd_sum tree. */
        for (short out_lane = (short)sgitg; out_lane < NW; out_lane += NSG) {
            const float4 materialized =
                (float4)partial_plane[(uint)lane * 33u + (uint)out_lane];
            const float4 reduced = simd_sum(materialized * reduce_ms);
            if (lane == 0) {
                dst4[quadrant * NW + out_lane] = reduced * reduce_inv;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// V4.1 decode attention in one dispatch: ds4's packed M5 FlashAttention (the
// 32 split-K workgroups and their reduce, with the arithmetic of the vec and
// reduce kernels) followed by kernel_dsv41_bf16_rope_rows' BF16 pass and
// inverse RoPE. Grid: x = head, y = row; row r reads its row_keys[r] keys at
// r*kv_row_bytes (padding keys past them up to args.ne11 are masked).
template <short NSG>
static inline void dsv41_flash_attn_packed_bf16_rope_impl(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char * q,
        device const char * kv,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        constant ds4_metal_args_dsv41_rope & rope,
        constant uint * pos,
        constant ulong & kv_row_bytes,
        constant uint * row_keys,
        threadgroup char * shmem,
        uint2  tg,
        ushort tiisg,
        ushort sgitg) {
    if (!FC_flash_attn_ext_vec_has_mask || !FC_flash_attn_ext_vec_has_sinks ||
         FC_flash_attn_ext_vec_has_bias || FC_flash_attn_ext_vec_has_scap ||
         FC_flash_attn_ext_vec_nsg != 1 || FC_flash_attn_ext_vec_nwg != 32 ||
         FC_flash_attn_ext_vec_ns10 != 512 || FC_flash_attn_ext_vec_ns20 != 512 ||
         args.ne01 != 1 || args.ne02 != 64 || args.ne11 <= 0 || args.ne11 > 1024 ||
         args.nb02 != 2048 || args.nb11 != 1024 || args.nb21 != 1024 ||
         rope.width != 512 || rope.heads != 64) {
        return;
    }
    const int ne11 = (int)row_keys[tg.y];
    if (ne11 <= 0 || ne11 > args.ne11) return;
    const ulong row_bytes = 64ul * 2048ul;
    device const char * k = kv + (ulong)tg.y * kv_row_bytes;
    dsv41_flash_attn_packed_f16_512<NSG>(args, q + (ulong)tg.y * row_bytes, k, k, mask,
        sinks, pad, dst + (ulong)tg.y * row_bytes, shmem, tg.x, ne11, tiisg, sgitg);
    threadgroup_barrier(mem_flags::mem_device);
    if (sgitg == 0) dsv41_bf16_rope_impl<true>(rope, (device float *)dst, pos, tg, tiisg);
}

#define DSV41_PACKED_ATTN(N) kernel void kernel_dsv41_flash_attn_packed_bf16_rope_sg##N(         constant ds4_metal_args_flash_attn_ext_vec & args [[buffer(0)]],         device const char * q     [[buffer(1)]],         device const char * kv    [[buffer(2)]],         device const char * mask  [[buffer(3)]],         device const char * sinks [[buffer(4)]],         device const char * pad   [[buffer(5)]],         device       char * dst   [[buffer(6)]],         constant ds4_metal_args_dsv41_rope & rope [[buffer(7)]],         constant uint * pos [[buffer(8)]],         constant ulong & kv_row_bytes [[buffer(9)]],         constant uint * row_keys [[buffer(10)]],         threadgroup char * shmem [[threadgroup(0)]],         uint2  tg    [[threadgroup_position_in_grid]],         ushort tiisg [[thread_index_in_simdgroup]],         ushort sgitg [[simdgroup_index_in_threadgroup]]) {     dsv41_flash_attn_packed_bf16_rope_impl<N>(args, q, kv, mask, sinks, pad, dst, rope, pos,         kv_row_bytes, row_keys, shmem, tg, tiisg, sgitg); }
DSV41_PACKED_ATTN(8)
DSV41_PACKED_ATTN(16)
DSV41_PACKED_ATTN(32)
