#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void init_invert_page_table(VirtualMemory* vm) {
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = -1;  // Take the second 1K space as the LRU list 
  }
}

__device__ void init_swap_table(VirtualMemory* vm) {
  for (int i = 0; i < vm->SWAP_TABLE_SIZE; i++) {
    vm->swap_table[i] = -1; // invalid :-1
  }
}

__device__ int update_lru(VirtualMemory* vm, int page_id) {
  /* If lru recorded the page, find it and place it at the top of LRU array */
  int move_pos = -1; //index of page_id in LRU array
  for (int i = vm->PAGE_ENTRIES; i <= vm->lru_oldest; i++) {
    if (vm->invert_page_table[i] == page_id) {
      move_pos = i; // if page_id is in the array, then everything (starting at )
      break;
    }
  }//find
  if (move_pos == -1) {
    vm->lru_oldest++;
    move_pos == vm->lru_oldest;
  }// the page is previously not in LRU
  for (int i = vm->PAGE_ENTRIES; i < move_pos; i++) {
    vm->invert_page_table[i + 1] = vm->invert_page_table[i];
  }//move
  int retval = vm->invert_page_table[vm->lru_oldest];
  vm->invert_page_table[vm->PAGE_ENTRIES] = page_id;
  return retval;
}

__device__ PageTableQuery get_page_table_entry(VirtualMemory* vm, int page_id) {
  /* Get the index of page entry using sequential search. If not found, return 0. (basically just a sequential search) */
  int empty_frame, frame_id;
  empty_frame = -1;
  frame_id = -1;
  for (int idx = 0; idx < vm->PAGE_ENTRIES; idx++) {
    if (vm->invert_page_table[idx] >> 31 == 0 &&        // find first empty frame
      empty_frame == -1) empty_frame = idx;
    if (vm->invert_page_table[idx] == page_id &&        // find the frame of page_id
      vm->invert_page_table[idx] >> 31 == 1) frame_id = idx;
    if (empty_frame != -1 && frame_id != -1) break;     // break if both are found
  }
  PageTableQuery query = { .frame_id = frame_id, .empty_frame = empty_frame };
  return query;
}

__device__ int search_swap_table(VirtualMemory* vm, int page_id){
  for(int i=0; i<vm->SWAP_TABLE_SIZE; i++){
    if(vm->swap_table[i] == -1) break;
    if(vm->swap_table[i] == page_id) return i;
  }
  return -1;
}

__device__ void swap_page(VirtualMemory* vm, int disk_page_id, int ram_page, uchar temp[]) {
  if (disk_page_id == -1) {
    // Nothing from disk->ram, allocate a place in disk, update the swap table.
    for (int i = 0; i < vm->SWAP_TABLE_SIZE; i++) {
      if (vm->swap_table[i] == -1) {
        disk_page_id = i;
        vm->swap_table[i] = ram_page;
        break;
      }
    }
  }
  else {
    // The requested page is stored in disk. Perform disk->ram.
    for (int i = 0; i < vm->PAGESIZE; i++) {
      vm->storage[disk_page_id*vm->PAGESIZE + i] = vm->buffer[vm->PAGESIZE * ram_page + i];
    }
  }
  for (int i=0; i<vm->PAGESIZE; i++) vm->storage[disk_page_id*vm->PAGESIZE + i] = temp[i];
}
__device__ void vm_init(VirtualMemory* vm, uchar* buffer, uchar* storage,
  u32* invert_page_table, u32* swap_table, int* pagefault_num_ptr,
  int PAGESIZE, int INVERT_PAGE_TABLE_SIZE, int SWAP_TABLE_SIZE,
  int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
  int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->swap_table = swap_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;
  vm->swap_table = NULL;
  vm->lru_oldest = PAGESIZE;

  // init constants
  vm->PAGESIZE = PAGESIZE-1;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->SWAP_TABLE_SIZE = SWAP_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}


__device__ uchar vm_read(VirtualMemory* vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  int page_id = addr / vm->PAGESIZE;
  int page_offset = addr % vm->PAGESIZE;
  int physical_addr;

  PageTableQuery res = get_page_table_entry(vm, page_id);
  int victim_page_id = update_lru(vm, page_id);//get the least recently used page as victim
  if (res.frame_id != -1) {
    physical_addr = res.frame_id * vm->PAGESIZE + page_offset;
  }
  else {
    (*vm->pagefault_num_ptr)++;
    if (res.empty_frame != -1){
      physical_addr = res.empty_frame * vm->PAGESIZE + page_offset;
      vm->invert_page_table[res.empty_frame] = vm->invert_page_table[res.empty_frame] & 0x00000000 + page_id;
    }
    else {//the RAM is full and the requested page is in the disk
      uchar temp_frame[vm->PAGESIZE];
      PageTableQuery victim_res = get_page_table_entry(vm, victim_page_id);
      if (victim_res.frame_id != -1){
        for(int i=0; i<vm->PAGESIZE; i++){ // transfer the frame to the buffer
          int _physical_addr = victim_res.frame_id * vm->PAGESIZE + i;
          temp_frame[i] = vm->buffer[_physical_addr];
        }
      }
      else perror("cannot find victim page in page table!\n");

      int target_disk_page_id = search_swap_table(vm, page_id);
      swap_page(vm, target_disk_page_id, victim_page_id, temp_frame);
      physical_addr = victim_res.frame_id * vm->PAGESIZE + page_offset;
    }
  }
  return vm->storage[physical_addr];
}

__device__ void vm_write(VirtualMemory* vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  int page_id = addr / vm->PAGESIZE;
  int page_offset = addr % vm->PAGESIZE;
  int physical_addr;

  PageTableQuery res = get_page_table_entry(vm, page_id);
  int victim_page_id = update_lru(vm, page_id);//get the least recently used page as victim
  if (res.frame_id != -1) {
    physical_addr = res.frame_id * vm->PAGESIZE + page_offset;
  }
  else {
    (*vm->pagefault_num_ptr)++; //gotta dereference the pointer first!
    if (res.empty_frame != -1) { //there are still vacant space in the RAM
      vm->invert_page_table[res.empty_frame] = vm->invert_page_table[res.empty_frame] & 0x00000000 + page_id; //update inverted page table, SET valid bit
      physical_addr = res.empty_frame * vm->PAGESIZE + page_offset;
    }
    else {                      //no more vacant space in the RAM, requires swapping
      uchar temp_frame[vm->PAGESIZE];
      PageTableQuery victim_res = get_page_table_entry(vm, victim_page_id);
      if (victim_res.frame_id != -1){
        for(int i=0; i<vm->PAGESIZE; i++){ // transfer the frame to the buffer
          int _physical_addr = victim_res.frame_id * vm->PAGESIZE + i;
          temp_frame[i] = vm->buffer[_physical_addr];
        }
      }
      else perror("cannot find victim page in page table!\n");

      int target_disk_page_id = search_swap_table(vm, page_id);
      swap_page(vm, target_disk_page_id, victim_page_id, temp_frame);
      physical_addr = victim_res.frame_id * vm->PAGESIZE + page_offset;
    }
  }
  vm->buffer[physical_addr] = value;
  printf("writing value %d to address %d\n", value, addr);
}

__device__ void vm_snapshot(VirtualMemory* vm, uchar* results, int offset,
  int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  for (int i = offset; i < input_size; ++i) {
    results[i] = vm_read(vm, i);
  }
}

