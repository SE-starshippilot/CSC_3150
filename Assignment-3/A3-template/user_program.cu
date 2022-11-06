#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size) {

  for (int i = 0; i < input_size; i++){
    vm_write(vm, i, input[i]);
    printf("Writing value %d to address %d", input[i], i);
  }

  for (int i = input_size - 1; i >= input_size - 32769; i--){
    int value = vm_read(vm, i);
    printf("Read value %d from %d", value, i);
  }

  vm_snapshot(vm, results, 0, input_size);
}
