﻿#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

struct LRUNode{
  int page_id;
  struct LRUNode* prev;
  struct LRUNode* next;
};

struct VirtualMemory {
  uchar* buffer;
  uchar* storage;
  u32* invert_page_table;
  int* swap_table;
  int* pagefault_num_ptr;

  struct LRUNode* lru_head;
  struct LRUNode* lru_tail;
  int lru_size;

  int PAGESIZE;
  int INVERT_PAGE_TABLE_SIZE;
  int SWAP_TABLE_SIZE;
  int PHYSICAL_MEM_SIZE;
  int STORAGE_SIZE;
  int PAGE_ENTRIES;
};

struct TableQuery {
  int frame_id;
  int empty_frame;
};


// TODO
__device__ void vm_init(VirtualMemory* vm, uchar* buffer, uchar* storage,
  u32* invert_page_table,int* pagefault_num_ptr,
  int PAGESIZE, int INVERT_PAGE_TABLE_SIZE, int SWAP_TABLE_SIZE,
  int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
  int PAGE_ENTRIES);
__device__ uchar vm_read(VirtualMemory* vm, u32 addr);
__device__ void vm_write(VirtualMemory* vm, u32 addr, uchar value);
__device__ void vm_snapshot(VirtualMemory* vm, uchar* results, int offset,
  int input_size);

#endif
