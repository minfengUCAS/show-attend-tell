package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//show_tell_attend/...",
    ],
)

py_binary(
    name = "build_mscoco_data",
    srcs = [
        "data/build_mscoco_data.py",
    ],
)

sh_binary(
    name = "download_and_preprocess_mscoco",
    srcs = ["data/download_and_preprocess_mscoco.sh"],
    data = [
        ":build_mscoco_data",
    ],
)

py_library(
    name = "configuration",
    srcs = ["configuration.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "show_attend_tell_model",
    srcs = ["show_attend_tell_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//show_tell_attend/ops:image_embedding",
        "//show_tell_attend/ops:image_processing",
        "//show_tell_attend/ops:inputs",
    ],
)

py_test(
    name = "show_attend_tell_model_test",
    size = "large",
    srcs = ["show_attend_tell_model_test.py"],
    deps = [
        ":configuration",
        ":show_attend_tell_model",
    ],
)

py_library(
    name = "inference_wrapper",
    srcs = ["inference_wrapper.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":show_attend_tell_model",
        "//show_tell_attend/inference_utils:inference_wrapper_base",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":show_attend_tell_model",
    ],
)

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":show_attend_tell_model",
    ],
)

py_binary(
    name = "run_inference",
    srcs = ["run_inference.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":inference_wrapper",
        "//show_tell_attend/inference_utils:caption_generator",
        "//show_tell_attend/inference_utils:vocabulary",
    ],
)
