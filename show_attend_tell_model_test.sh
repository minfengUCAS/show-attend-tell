bazel build show_tell_attend/show_attend_tell_model_test

export CUDA_VISIBLE_DEVICES="gpu:0"

bazel-bin/show_tell_attend/show_attend_tell_model_test

