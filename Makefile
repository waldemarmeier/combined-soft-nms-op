CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

COMBINED_SOFT_NMS_SRCS = $(wildcard tensorflow_combined_soft_nms/cc/kernels/*.cc) $(wildcard tensorflow_combined_soft_nms/cc/ops/*.cc)
ZERO_OUT_SRCS = $(wildcard tensorflow_zero_out/cc/kernels/*.cc) $(wildcard tensorflow_zero_out/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

ZERO_OUT_TARGET_LIB = tensorflow_zero_out/python/ops/_zero_out_ops.so
COMBINED_SOFT_NMS_TARGET_LIB = tensorflow_zero_out/python/ops/_combined_soft_nms_ops.so

# combined soft nms op for CPU
combined_soft_nms_op: $(COMBINED_SOFT_NMS_TARGET_LIB)

$(COMBINED_SOFT_NMS_TARGET_LIB): $(COMBINED_SOFT_NMS_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS} -undefined dynamic_lookup

# zero_out_test: tensorflow_zero_out/python/ops/zero_out_ops_test.py tensorflow_zero_out/python/ops/zero_out_ops.py $(ZERO_OUT_TARGET_LIB)
# 	$(PYTHON_BIN_PATH) tensorflow_zero_out/python/ops/zero_out_ops_test.py

combined_soft_nms_pip_pkg: $(COMBINED_SOFT_NMS_TARGET_LIB)
	./build_pip_pkg.sh make artifacts


# zero_out op for CPU
zero_out_op: $(ZERO_OUT_TARGET_LIB)

$(ZERO_OUT_TARGET_LIB): $(ZERO_OUT_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS} -undefined dynamic_lookup

zero_out_test: tensorflow_zero_out/python/ops/zero_out_ops_test.py tensorflow_zero_out/python/ops/zero_out_ops.py $(ZERO_OUT_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_zero_out/python/ops/zero_out_ops_test.py

zero_out_pip_pkg: $(ZERO_OUT_TARGET_LIB)
	./build_pip_pkg.sh make artifacts

clean:
	rm -f $(ZERO_OUT_TARGET_LIB) $(COMBINED_SOFT_NMS_TARGET_LIB)
