CXX := g++
PYTHON_BIN_PATH = python

COMBINED_SOFT_NMS_SRCS = $(wildcard tensorflow_combined_soft_nms/cc/kernels/*.cc) $(wildcard tensorflow_combined_soft_nms/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14
LDFLAGS = -shared ${TF_LFLAGS}

COMBINED_SOFT_NMS_TARGET_LIB = tensorflow_combined_soft_nms/python/ops/_combined_soft_nms_ops.so

# combined soft nms op for CPU
combined_soft_nms_op: $(COMBINED_SOFT_NMS_TARGET_LIB)

# -undefined dynamic_lookup
$(COMBINED_SOFT_NMS_TARGET_LIB): $(COMBINED_SOFT_NMS_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

combined_soft_nms_test: tensorflow_combined_soft_nms/python/ops/combined_soft_nms_ops_test.py tensorflow_combined_soft_nms/python/ops/combined_soft_nms_ops.py $(COMBINED_SOFT_NMS_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_combined_soft_nms/python/ops/combined_soft_nms_ops_test.py

combined_soft_nms_pip_pkg: $(COMBINED_SOFT_NMS_TARGET_LIB)
	./build_pip_pkg.sh make artifacts

clean:
	rm -f $(COMBINED_SOFT_NMS_TARGET_LIB)
