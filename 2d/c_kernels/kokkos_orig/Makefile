vpath %.cpp $(KERNELS_PATH)

# User definable parameters
KOKKOS_PATH   ?= $(HOME)/kokkos
KOKKOS_TARGET ?= OPENMP

# Kokkos target device settings
# Currently constrained CUDA compilation
ifeq ($(KOKKOS_TARGET), CUDA)
  CXX			  = $(NVCC_WRAPPER)
  KERNEL_COMPILER = nvcc
  KERNEL_FLAGS	  = -ccbin g++ -std=c++11 -arch=sm_35 --expt-extended-lambda \
					-restrict -Xcompiler "-O3 -fopenmp" $(KOKKOS_CPPFLAGS) \
 					-I$(KERNELS_PATH) --x cu $(OPTIONS) -DCUDA
  TL_LDFLAGS	 += -lcudart
  KOKKOS_DEVICES  = "Cuda"
  KOKKOS_ARCH 	  = "Kepler35"
endif
ifeq ($(KOKKOS_TARGET), OPENMP)
  CXX 			  = $(CPP)
  KERNEL_COMPILER = $(CPP)
  KERNEL_FLAGS	  = $(CPPFLAGS) $(OPTIONS) -DOPENMP
  KOKKOS_DEVICES  = "OpenMP"
  KOKKOS_ARCH 	  = "SNB"
endif
ifeq ($(KOKKOS_TARGET), MIC)
  CXX 			  = $(CPP)
  KERNEL_COMPILER = $(CPP)
  KERNEL_FLAGS	  = $(CPPFLAGS) $(OPTIONS) -DKNC -mmic
  KOKKOS_DEVICES  = "OpenMP"
  KOKKOS_ARCH 	  = "KNC"
endif

include $(KOKKOS_PATH)/Makefile.kokkos
INCLUDE_DIRS := ./ $(KERNELS_PATH) $(KOKKOS_PATH)
CPPFLAGS	  += $(KOKKOS_CPPFLAGS) $(addprefix -I, $(INCLUDE_DIRS))

# Override the default compiler and flags
TL_LINK	    := $(CPP)
TL_COMPILER := $(KERNEL_COMPILER)
TL_FLAGS	:= $(KERNEL_FLAGS)
TL_LDFLAGS 	+= $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS) -Lobj/$(KERNELS) -fopenmp

# Locate the kernel files to compile
KERNEL_SRCS := $(wildcard $(KERNELS_PATH)/*.cpp)
KERNEL_OBJS := $(patsubst $(KERNELS_PATH)/%.cpp, obj/$(KERNELS)/%.o, $(KERNEL_SRCS))

build_kernels: $(KERNEL_OBJS) $(KOKKOS_LINK_DEPENDS) 
	-@mv *.o *.a obj/$(KERNELS)/
	@echo "Built Kokkos kernels"

obj/$(KERNELS)/%.o: %.cpp $(KOKKOS_CPP_DEPENDS) Makefile make.deps
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) -c $< -o $@

# Have to build the chunk independently
obj/$(KERNELS)/chunk.o: chunk.c $(KOKKOS_CPP_DEPENDS) Makefile make.deps
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) -c $< -o $@

