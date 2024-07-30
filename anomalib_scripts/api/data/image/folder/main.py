from pathlib import Path

from utils import (
    get_filter_from_name,
    from_subdir_name_to_dict,
    apply_filter,
    filter_subdirs,
    get_filters_from_combinations,
    get_filters_to_subdirs_map
)


source_dirs_comb = [["type_1"],
                    ["type_2"],]

source_sub_dirs_comb = [["1", "2"],
                        ["2"],
                        ["3"],
                        ["4"],
                        ["5"],
                        ["7"],
                        ["8"],
                        ["9"],
                        ["10"],
                        ["12"],
                        ["13"],
                        ["14"],
                        ["15"],
                        ["16"],
                        ["17"],]

cropping_variants_comb = [["without_background"],
                          ["with_background"],
                          ["without_background", "with_background"],]

tiling_variants_comb = [["normal_tiles"], ]

tiling_params_comb = [["512-512_32-32"],
                      ["384-384_32-32"],
                      ["256-256_32-32"],
                      ["128-128_4-4"],
                      ["108-108_4-4"],
                      ["98-98_4-4"],]


if __name__ == "__main__":

    filters = get_filters_from_combinations(
        source_dirs_comb=source_dirs_comb,
        source_sub_dirs_comb=source_sub_dirs_comb,
        cropping_variants_comb=cropping_variants_comb,
        tiling_variants_comb=tiling_variants_comb,
        tiling_params_comb=tiling_params_comb
    )

    dataset_root = Path.cwd() / "custom_datasets" / "cabinet_metallic_surface" / "blue"

    input_dir = dataset_root / "normal"

    filters_to_subdirs_map = get_filters_to_subdirs_map(
        filters=filters,
        input_dir=input_dir,
        output_subdir_prefix="normal/"
    )

    for item in filters_to_subdirs_map.items():
        print(item)

