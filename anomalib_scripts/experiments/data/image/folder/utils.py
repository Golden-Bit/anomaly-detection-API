# flake8: noqa
import os
from typing import Dict, Any

import numpy as np
from PIL import Image
from anomalib.data.utils import TestSplitMode, ValSplitMode
from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2.functional import to_pil_image

from anomalib.data.image.folder import Folder, FolderDataset
from anomalib import TaskType

from pathlib import Path


def get_filter_from_name(filter_name: str,
                         split_fields_by: str,
                         split_variants_by: str,
                         keys: list[str]):

    #print(filter_name)
    fields_values = filter_name.split(split_fields_by)
    output_dict = dict()

    for field_key, field_value in zip(keys, fields_values):

        variants_values = field_value.split(split_variants_by)
        if variants_values == ["all"]:
            output_dict[field_key] = {"include": variants_values[0]}
        else:
            output_dict[field_key] = {"include": list()}
            for variant_value in variants_values:
                output_dict[field_key]["include"].append(variant_value)

    return output_dict


def from_subdir_name_to_dict(subdir_name: str,
                             split_by: str,
                             keys: list[str]):

    values = subdir_name.split(split_by)

    output_dict = dict()
    for key, value in zip(keys, values):
        output_dict[key] = value

    return output_dict


def apply_filter(dict_to_filter: Dict[str, Any],
                 dict_filter: Dict[str, Any]):

    filter_result = True
    for field, field_filter in list(dict_filter.items()):
        if isinstance(field_filter["include"], list):
            if dict_to_filter[field] not in field_filter["include"]:
                # print(dict_to_filter[field_to_filter], filter["include"])
                filter_result = False
                break

        if "exclude" in field_filter:
            if dict_to_filter[field] in field_filter["exclude"]:
                # print(dict_to_filter[field_to_filter], filter["exclude"])
                filter_result = False
                break

    return filter_result


def filter_subdirs(
        input_dir: Any,
        dict_filter: Dict[str, Any] = None
):
    filtered_subdirs = list()
    subdir_names = os.listdir(input_dir)

    for subdir_to_filter in subdir_names:

        dict_to_filter = from_subdir_name_to_dict(
            subdir_name=subdir_to_filter,
            split_by="___",
            keys=["source_dir", "source_sub_dir", "cropping_variant", "tiling_variant", "tiling_params"]
        )

        filter_result = apply_filter(
            dict_to_filter=dict_to_filter,
            dict_filter=dict_filter
        )

        if filter_result:
            filtered_subdirs.append(subdir_to_filter)

    return filtered_subdirs


def get_filters_from_combinations(
        source_dirs_comb: list[list[str] | str] = ['all'],
        source_sub_dirs_comb: list[list[str] | str] = ['all'],
        cropping_variants_comb: list[list[str] | str] = ['all'],
        tiling_variants_comb: list[list[str] | str] = ['all'],
        tiling_params_comb: list[list[str] | str] = ['all'],
):
    filters_keys = list()
    filters_dicts = dict()

    for source_dirs in source_dirs_comb:

        for source_sub_dirs in source_sub_dirs_comb:

            for cropping_variants in cropping_variants_comb:

                for tiling_variants in tiling_variants_comb:

                    for tiling_params in tiling_params_comb:

                        filter = dict()

                        source_dir = {
                            "include": source_dirs,
                            "exclude": [],
                        }

                        source_sub_dir = {
                            "include": source_sub_dirs,
                            "exclude": [],
                        }

                        cropping_variant = {
                            "include": cropping_variants,
                            "exclude": [],
                        }

                        tiling_variant = {
                            "include": tiling_variants,
                            "exclude": [],
                        }

                        tiling_params = {
                            "include": tiling_params,
                            "exclude": [],
                        }

                        filter["source_dir"] = source_dir
                        filter["source_sub_dir"] = source_sub_dir
                        filter["cropping_variant"] = cropping_variant
                        filter["tiling_variant"] = tiling_variant
                        filter["tiling_params"] = tiling_params

                        filter_name_parts = list()
                        for field_to_filter, filter_value in list(filter.items()):
                            if isinstance(filter_value['include'], list):
                                filter_name_parts.append(f"{'__'.join(filter_value['include'])}")
                            else:
                                filter_name_parts.append(f"{filter_value['include']}")

                        filter_name = "___".join(filter_name_parts)
                        filters_keys.append(filter_name)
                        filters_dicts[filter_name] = filter

    return {
        "filters_keys": filters_keys,
        "filters_dicts": filters_dicts
    }


def get_filters_to_subdirs_map(
        input_dir: str | Path,
        filters: Dict[str, Any],
        output_subdir_prefix: str,
):

    filters_to_subdirs_map = dict()

    filters_keys = filters["filters_keys"]
    filters_dicts = filters["filters_dicts"]

    for filter_key in filters_keys:

        dict_filter_1 = get_filter_from_name(
            filter_name=filter_key,
            split_fields_by="___",
            split_variants_by="__",
            keys=["source_dir", "source_sub_dir", "cropping_variant", "tiling_variant", "tiling_params"]
        )

        dict_filter_2 = filters_dicts[filter_key]

        #for item_1, item_2  in zip(list(dict_filter_1.items()), list(dict_filter_2.items())):
        #    print(item_1, item_2)

        filtered_subdirs_1 = filter_subdirs(
            input_dir=input_dir,
            dict_filter=dict_filter_1
        )

        filtered_subdirs_2 = filter_subdirs(
            input_dir=input_dir,
            dict_filter=dict_filter_2
        )

        if filtered_subdirs_1 == filtered_subdirs_2:

            if filtered_subdirs_1 and filtered_subdirs_2:
                #print("=" * 100)
                #print(filter_key)
                #print("-" * 100)
                #for subdir_1, subdir_2 in zip(filtered_subdirs_1, filtered_subdirs_2):
                #    print("-" * 100)
                #    print(f"{subdir_1}\n{subdir_2}")
                #print("=" * 100)

                filtered_subdirs = [f"{output_subdir_prefix}{filtered_subdir}" for filtered_subdir in filtered_subdirs_1]

                filtered_subdirs = set(filtered_subdirs)

                is_break = False

                for _filter_key, _filtered_subdirs in list(filters_to_subdirs_map.items()):

                    if filtered_subdirs == _filtered_subdirs:
                        if len(_filter_key) <= len(filter_key):
                            is_break = True
                            break
                        else:
                            del filters_to_subdirs_map[_filter_key]
                            filters_to_subdirs_map[filter_key] = filtered_subdirs
                            is_break = True
                            break

                if not is_break:
                    filters_to_subdirs_map[filter_key] = filtered_subdirs

        else:
            print(filtered_subdirs_1, filtered_subdirs_2)
            input("[ERROR MESSAGE] conversion 'filters_keys from/to filters_dicts' failed!")

    for key, value in filters_to_subdirs_map.items():
        filters_to_subdirs_map[key] = list(value)

    return filters_to_subdirs_map


if __name__ == "__main__":

    filter_name = "type_1__type_2___1__3__7__15___without_background__with_background___normal_tiles___384-384_32-32__256-256_32-32__128-128_4-4"

    dict_filter = get_filter_from_name(
        filter_name=filter_name,
        split_fields_by="___",
        split_variants_by="__",
        keys=["source_dir", "source_sub_dir", "cropping_variant", "tiling_variant", "tiling_params"]
    )

    for item in dict_filter.items():
        print(item)

    dataset_root = Path.cwd() / "custom_datasets" / "cabinet_metallic_surface" / "blue"
    subset_name = "normal"
    input_dir = dataset_root / subset_name

    filtered_subdirs = filter_subdirs(
        input_dir=input_dir,
        dict_filter=dict_filter
    )

    filtered_subdirs = [f"{subset_name}/{filtered_subdir}" for filtered_subdir in filtered_subdirs]

    for subdir in filtered_subdirs:
        print(subdir)

