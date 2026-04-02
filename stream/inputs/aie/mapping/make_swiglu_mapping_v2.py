import copy
import os

import yaml


def make_swiglu_mapping_v2(
    seq_len,
    embedding_dim,
    hidden_dim,
    last_gemm_down,
    seq_len_tile_options: list[int],
    embedding_tile_options: list[int],
    hidden_tile_options: list[int],
):  # noqa: N803
    """
    V2 mapping for SwiGLU with tile_options format supporting variable tile size DSE.

    Each *_tile_options parameter is a list of possible tile sizes for that dimension.
    For Phase 1 (baseline), each list contains exactly one element.

    Models after make_swiglu_mapping2() but uses tile_options instead of fixed tile values
    in the intra_core_tiling section.
    """
    name = f"swiglu_{seq_len}_{embedding_dim}_{hidden_dim}"
    output_file = os.path.join(os.path.dirname(__file__), f"{name}_v2.yaml")

    seq_len_tile_size = seq_len_tile_options[0]
    embedding_tile_size = embedding_tile_options[0]
    hidden_tile_size = hidden_tile_options[0]

    assert seq_len % 4 == 0, "seq_len must be divisible by 4 for this mapping"
    assert seq_len >= seq_len_tile_size * 4, f"seq_len must be at least {seq_len_tile_size * 4} for this mapping"
    assert embedding_dim % embedding_tile_size == 0, (
        f"embedding_dim must be divisible by embedding_tile_size ({embedding_tile_size})"
    )
    assert hidden_dim % hidden_tile_size == 0, (
        f"hidden_dim must be divisible by hidden_tile_size ({hidden_tile_size})"
    )

    # Kernel selection based on seq_len tile size (matvec for seq_len_tile_size==1, gemm otherwise)
    if seq_len_tile_size == 1:
        kernel_gemm = {"name": "matvec", "kwargs": {"utilization": 61.8, "layout": "default"}}
    else:
        kernel_gemm = {
            "name": "gemm",
            "kwargs": {
                "m": seq_len_tile_size,
                "k": embedding_tile_size,
                "n": hidden_tile_size,
                "utilization": 61.8,
                "layout": "default",
            },
        }

    # Left Gemm
    inter_core_tiling_gemm_left = [
        [{"dim": "D0", "split": 4}, {"dim": "D2", "split": 2}],
    ]
    compute_allocation_gemm_left = [
        [2, 3, 4, 5, 8, 9, 10, 11],
    ]
    gemm_left = {
        "name": "Gemm_Left",
        "core_allocation": copy.deepcopy(compute_allocation_gemm_left),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling_gemm_left),
        "kernel": copy.deepcopy(kernel_gemm),
    }

    # Right Gemm
    inter_core_tiling_gemm_right = [
        [{"dim": "D0", "split": 4}, {"dim": "D2", "split": 2}],
    ]
    compute_allocation_gemm_right = [
        [14, 15, 16, 17, 20, 21, 22, 23],
    ]
    gemm_right = {
        "name": "Gemm_Right",
        "core_allocation": copy.deepcopy(compute_allocation_gemm_right),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling_gemm_right),
        "kernel": copy.deepcopy(kernel_gemm),
    }

    # SiLU
    compute_allocation_silu = [
        [26, 27, 28, 29],
    ]
    inter_core_tiling_silu = [
        [{"dim": "D0", "split": 4}],
    ]
    kernel_silu = {"name": "silu", "kwargs": {"utilization": 50.0, "layout": "default"}}
    silu = {
        "name": "Silu",
        "core_allocation": copy.deepcopy(compute_allocation_silu),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling_silu),
        "kernel": copy.deepcopy(kernel_silu),
    }

    # Elementwise Mul
    compute_allocation_mul = [
        [32, 33, 34, 35],
    ]
    inter_core_tiling_mul = [
        [{"dim": "D0", "split": 4}],
    ]
    kernel_mul = {"name": "eltwise_mul", "kwargs": {"utilization": 50.0, "layout": "default"}}
    mul = {
        "name": "Elt_Mul",
        "core_allocation": copy.deepcopy(compute_allocation_mul),
        "inter_core_tiling": copy.deepcopy(inter_core_tiling_mul),
        "kernel": copy.deepcopy(kernel_mul),
    }

    # Final down projection Gemm (k and n are reversed: k=hidden, n=embedding)
    if last_gemm_down:
        inter_core_tiling_gemm_down = [
            [{"dim": "D0", "split": 4}, {"dim": "D1", "split": 2}],
        ]
        compute_allocation_gemm_down = [
            [38, 39, 40, 41, 44, 45, 46, 47],
        ]
        if seq_len_tile_size == 1:
            kernel_gemm_down = {"name": "matvec", "kwargs": {"utilization": 61.8, "layout": "default"}}
        else:
            kernel_gemm_down = {
                "name": "gemm",
                "kwargs": {
                    "m": seq_len_tile_size,
                    "k": hidden_tile_size,
                    "n": embedding_tile_size,
                    "utilization": 61.8,
                    "layout": "default",
                },
            }
        gemm_down = {
            "name": "Gemm_Down",
            "core_allocation": copy.deepcopy(compute_allocation_gemm_down),
            "inter_core_tiling": copy.deepcopy(inter_core_tiling_gemm_down),
            "kernel": copy.deepcopy(kernel_gemm_down),
        }
        layers = [gemm_left, gemm_right, silu, mul, gemm_down]
        runtime_args = {
            "input": {},
            "weights_1": {"layout": "(d0, d1) -> (d1, d0)"},
            "weights_2": {"layout": "(d0, d1) -> (d1, d0)"},
            "weights_3": {"layout": "(d0, d1) -> (d1, d0)"},
            "output": {},
        }
    else:
        layers = [gemm_left, gemm_right, silu, mul]
        runtime_args = {
            "input": {},
            "weights_1": {"layout": "(d0, d1) -> (d1, d0)"},
            "weights_2": {"layout": "(d0, d1) -> (d1, d0)"},
            "output": {},
        }

    # Fused groups with tile_options format
    fused_groups = {
        "name": "Fused_Group_1",
        "layers": [layer["name"] for layer in layers],
        "intra_core_tiling": [
            {"dim": "Gemm_Left.D1", "tile_options": embedding_tile_options},
            {"dim": "Gemm_Left.D2", "tile_options": hidden_tile_options},
            {"dim": "Gemm_Left.D0", "tile_options": seq_len_tile_options},
        ],
    }
    if last_gemm_down:
        fused_groups["intra_core_tiling"].insert(1, {"dim": "Gemm_Down.D2", "tile_options": embedding_tile_options})

    mapping = {
        "layers": layers,
        "fused_groups": [fused_groups],
        "runtime_args": runtime_args,
    }

    with open(output_file, "w") as f:
        yaml.dump(mapping, f, default_flow_style=False, sort_keys=False)
    print(f"SWIGLU v2 mapping file created: {output_file}")
    return output_file
