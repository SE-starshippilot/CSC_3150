#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

__device__ void init_invert_page_table(VirtualMemory* vm) {
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    // vm->invert_page_table[i + vm->PAGE_ENTRIES] = 0xffffffff;  // Take the second 1K space to record frame number 
    // vm->invert_page_table[i+ 2*vm->PAGE_ENTRIES] = 0xffffffff; // Take the third 1K space as the LRU list
  }
}

__device__ void init_swap_table(VirtualMemory* vm) {
  for (int i = 0; i < vm->SWAP_TABLE_SIZE; i++) {
    vm->swap_table[i] = -1; // invalid :-1
  }
}
__device__ void show_lru(VirtualMemory* vm) {
  // int size = 4;
  struct LRUNode* _tmp = vm->lru_head;
  while (_tmp != NULL) {
    printf("%d->", _tmp->page_id);
    _tmp = _tmp->next;
  }
  printf("\n");
}

__device__ int update_lru(VirtualMemory* vm, int page_id, int is_oldpage) {
  /* If lru recorded the page, find it and place it at the top of LRU array */
  printf("Trying to put %d in list.Before update:", page_id);
  show_lru(vm);
  int ret_val = -1;  // index of page_id in LRU array
  if (is_oldpage) {                  // save some time. no need to search LRU array if it is a new page
    for (LRUNode* i = vm->lru_head; i != NULL; i = i->next) {
      if (i->page_id == page_id) {
        if (i->prev != NULL)i->prev->next = i->next;
        if (i->next != NULL)i->next->prev = i->prev;
        else {
          vm->lru_tail = i->prev;
          vm->lru_tail->next = NULL;
        }
        i->next = vm->lru_head;
        i->prev = NULL;
        vm->lru_head->prev = i;
        vm->lru_head = i;
        break;
      }
    }//find
  }
  else {
    struct LRUNode* _node = (struct LRUNode*)malloc(sizeof(struct LRUNode));
    _node->page_id = page_id;
    _node->next = vm->lru_head;
    _node->prev = NULL;
    if (vm->lru_size == 0)vm->lru_tail = _node;
    else vm->lru_head->prev = _node;
    vm->lru_head = _node;
    vm->lru_size++;
    printf("Current LRU size:%d", vm->lru_size);
    if (vm->lru_size > vm->PAGE_ENTRIES) {
      printf("Current tail, page_id is %d, prev is %p, next is %p\n", vm->lru_tail->page_id, vm->lru_tail->prev, vm->lru_tail->next);
      ret_val = vm->lru_tail->page_id;
      vm->lru_tail = vm->lru_tail->prev;
      vm->lru_tail->next = NULL;
      vm->lru_size--;
      printf("After update, tail page_id is %d, prev is %p, next is %p\n", vm->lru_tail->page_id, vm->lru_tail->prev, vm->lru_tail->next);
    }
  }
  // printf("After update:");
  // show_lru(vm);
  return ret_val;
}

__device__ PageTableQuery query_page_table(VirtualMemory* vm, int page_id) {
  /* Get the index of page entry using sequential search. If not found, return 0. (basically just a sequential search) */
  int empty_frame, frame_id;
  empty_frame = -1;
  frame_id = -1;
  for (int idx = 0; idx < vm->PAGE_ENTRIES; idx++) {
    if (vm->invert_page_table[idx] >> 31 == 1 &&        // find first empty frame
      empty_frame == -1) empty_frame = idx;
    if (vm->invert_page_table[idx] == page_id &&        // find the frame of page_id
      vm->invert_page_table[idx] >> 31 == 0) frame_id = idx;
    if (frame_id != -1) break;     // break if the frame is found
  }
  PageTableQuery query = { .frame_id = frame_id, .empty_frame = empty_frame };
  return query;
}

__device__ int search_swap_table(VirtualMemory* vm, int page_id) {
  for (int i = 0; i < vm->SWAP_TABLE_SIZE; i++) {
    if (vm->swap_table[i] == page_id) return i;
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
      vm->storage[disk_page_id * vm->PAGESIZE + i] = vm->buffer[vm->PAGESIZE * ram_page + i];
    }
  }
  for (int i = 0; i < vm->PAGESIZE; i++) vm->storage[disk_page_id * vm->PAGESIZE + i] = temp[i];
}

__device__ void vm_init(VirtualMemory* vm, uchar* buffer, uchar* storage,
  u32* invert_page_table, int* pagefault_num_ptr,
  int PAGESIZE, int INVERT_PAGE_TABLE_SIZE, int SWAP_TABLE_SIZE,
  int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
  int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->swap_table = (int*)malloc(sizeof(int) * SWAP_TABLE_SIZE);
  vm->pagefault_num_ptr = pagefault_num_ptr;

  vm->lru_head = NULL;
  vm->lru_tail = NULL;
  vm->lru_size = 0;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->SWAP_TABLE_SIZE = SWAP_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
  init_swap_table(vm);
}

__device__ uchar vm_read(VirtualMemory* vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  int page_id = addr / vm->PAGESIZE;
  int page_offset = addr % vm->PAGESIZE;
  int physical_addr;
  printf("[Read] VA: %d Pid:%d Poff:%d ", addr, page_id, page_offset);
  PageTableQuery res = query_page_table(vm, page_id);
  if (res.frame_id != -1) {
    physical_addr = res.frame_id * vm->PAGESIZE + page_offset;
    if (vm->lru_head->page_id != page_id) update_lru(vm, page_id, 1);//get the least recently used page as victim
    printf("[HIT!!:%d] Fid: %d ", (*vm->pagefault_num_ptr), res.frame_id);
  }
  else {
    (*vm->pagefault_num_ptr)++;
    printf("[FAULT:%d]", (*vm->pagefault_num_ptr));
    int victim_page_id = update_lru(vm, page_id, 0);
    if (res.empty_frame != -1) {
      printf(" [Empty Frame:%d]", res.empty_frame);
      vm->invert_page_table[res.empty_frame] = page_id;
      printf("Fid: %d ", res.empty_frame);
      physical_addr = res.empty_frame * vm->PAGESIZE + page_offset;
    }
    else {//the RAM is full and the requested page is in the disk
      printf("Victim frame:%d", victim_page_id);
      uchar temp_frame[32];
      PageTableQuery victim_res = query_page_table(vm, victim_page_id);
      printf("Found victim on frame:%d", victim_res.frame_id);
      assert(victim_res.frame_id != -1);
      for (int i = 0; i < vm->PAGESIZE; i++) { // transfer the frame to the buffer
        int _physical_addr = victim_res.frame_id * vm->PAGESIZE + i;
        temp_frame[i] = vm->buffer[_physical_addr];
      }

      int target_disk_frame_id = search_swap_table(vm, page_id);
      printf("Found target page on disk frame:%d", target_disk_frame_id);
      swap_page(vm, target_disk_frame_id, victim_page_id, temp_frame);
      vm->invert_page_table[victim_res.frame_id] = page_id;
      printf("Fid: %d ", victim_res.frame_id);
      physical_addr = victim_res.frame_id * vm->PAGESIZE + page_offset;
    }
  }
  printf("Val %d PA %d\n", vm->buffer[physical_addr], physical_addr);
  return vm->buffer[physical_addr];
}

__device__ void vm_write(VirtualMemory* vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  int page_id = addr / vm->PAGESIZE;
  int page_offset = addr % vm->PAGESIZE;
  int physical_addr;
  printf("[Write]Val: %d VA: %d Pid:%d Poff:%d ", value, addr, page_id, page_offset);
  PageTableQuery res = query_page_table(vm, page_id);
  if (res.frame_id != -1) {
    physical_addr = res.frame_id * vm->PAGESIZE + page_offset;
    if (vm->lru_head->page_id != page_id) update_lru(vm, page_id, 1);
    printf("[HIT!!:%d] Fid: %d ", (*vm->pagefault_num_ptr), res.frame_id);
  }
  else {
    (*vm->pagefault_num_ptr)++; //gotta dereference the pointer first!
    printf("[FAULT:%d]", (*vm->pagefault_num_ptr));
    int victim_page_id = update_lru(vm, page_id, 0);//if it's not in page table, it must not be in LRU. pick the victim
    if (res.empty_frame != -1) { //there are still vacant space in the RAM
      vm->invert_page_table[res.empty_frame] = page_id; //update inverted page table, SET valid bit
      printf("Fid: %d ", res.empty_frame);
      physical_addr = res.empty_frame * vm->PAGESIZE + page_offset;
    }
    else {                      //no more vacant space in the RAM, requires swapping
      uchar temp_frame[32];
      PageTableQuery victim_res = query_page_table(vm, victim_page_id);
      assert(victim_res.frame_id != -1);
      for (int i = 0; i < vm->PAGESIZE; i++) { // transfer the frame to the buffer
        int _physical_addr = victim_res.frame_id * vm->PAGESIZE + i;
        temp_frame[i] = vm->buffer[_physical_addr];
      }

      int target_disk_frame_id = search_swap_table(vm, page_id);
      swap_page(vm, target_disk_frame_id, victim_page_id, temp_frame);
      vm->invert_page_table[victim_res.frame_id] = page_id;
      printf("Fid: %d ", victim_res.frame_id);
      physical_addr = victim_res.frame_id * vm->PAGESIZE + page_offset;
    }
  }
  printf("PA %d. LRU size:%d.\n", physical_addr, vm->lru_size);
  vm->buffer[physical_addr] = value;
}

__device__ void vm_snapshot(VirtualMemory* vm, uchar* results, int offset,
  int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  for (int i = offset; i < input_size; i++) {
    int read_val = vm_read(vm, i);
    // printf("Read val: %d\n", read_val);
    results[i] = read_val;
  }
}

