﻿#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define CREATE_TIME_ATTR_OFFSET 2
#define MODIFY_TIME_ATTR_OFFSET 4
#define STARTBLK_ATTR_OFFSET 6
#define SIZE_ATTR_OFFSET 8
#define NAME_ATTR_OFFSET 12
#define CREATE_TIME_ATTR_LENGTH 2
#define MODIFY_TIME_ATTR_LENGTH 2
#define STARTBLK_ATTR_LENGTH 2
#define SIZE_ATTR_LENGTH 4
#define FCB_VALID 0b10000000
#define FCB_INVALID 0b00000000
#define FP_INVALID 1024

__device__ __managed__ u32 gtime = 0; // increasing. larger means newer
__device__ __managed__ u32 gfilenum = 0;
__device__ __managed__ u32 glastblock = 0;

__device__ void fcb_init(FileSystem* fs) {
  for (u32 i = 0; i < fs->FCB_ENTRIES; i++) {
    set_file_attr(fs, i, 0, 1, FCB_INVALID);// MSB in the first byte of FCB is valid bit. 0 indicates invalid.
  }
}

__device__ void superblock_init(FileSystem* fs) {
  // Initialize superblock. In my implementation, 0 means free and 1 means used.
  for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++) {
    fs->volume[i] = (uchar)0x00;
  }
}

__device__ void fs_init(FileSystem* fs, uchar* volume, int SUPERBLOCK_SIZE,
  int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
  int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
  int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

  // init superblock
  superblock_init(fs);

  // init file control block
  fcb_init(fs);
}

__device__ int str_cmp(char* str1, char* str2) {
  /* Compare two strings. Return 1 if they are the same. */
  while (*str1 != '\0' && *str2 != '\0') {
    if (*str1 != *str2) return 0;
    str1++;
    str2++;
  }
  if (*str1 == '\0' && *str2 == '\0') return 1;
  else return 0;
}

__device__ int str_len(const char* str) {
  /* Return the length of a string. */
    const char *s;
    for (s = str; *s; ++s);
    return(s - str);
}

__device__ char* get_file_attr(FileSystem* fs, u32 fp, int attr_offset) {
  u32 fcb_attr_addr = fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + attr_offset;
  char* file_name = (char*)fs->volume + fcb_attr_addr;
  return file_name;
}

__device__ int get_file_attr(FileSystem* fs, u32 fp, int attr_offset, int attr_length) {
  /* Read file attribute from FCB. */
  u32 fcb_attr_addr = fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + attr_offset;
  int result = 0;
  memcpy(&result, fs->volume + fcb_attr_addr, attr_length);
  return result;
}

__device__ void set_file_attr(FileSystem* fs, u32 fp, int attr_offset, int attr_length, int value) {
  /* Set file attribute. */
  // u32 fcb_attr_addr = fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + attr_offset;
  memcpy(fs->volume + fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + attr_offset, &value, attr_length);
}

__device__ void set_file_attr(FileSystem* fs, u32 fp, int attr_offset, int attr_length, char* value) {
  /* Set file attribute. This reloaded function is for setting file name only. */
  // int fcb_attr_addr = fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + attr_offset;
  memcpy(fs->volume + fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + attr_offset, value, attr_length);
  memset(fs->volume + fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + attr_offset + attr_length,  0, 1);
}

__device__ FCBQuery search_file(FileSystem* fs, char* s) {
  /* Find the FCB of the file with name s.
   * Return a query result containing the fcb index (if found) and first empty fcb index (if found).
   */
  FCBQuery ret_val = { FP_INVALID, FP_INVALID };
  int valid_fcb_traversed = 0;
  for (u32 i = 0; i < fs->FCB_ENTRIES; i++) {
    if (get_file_attr(fs, i, 0, 1) == FCB_VALID) { // valid bit is set
      valid_fcb_traversed++;
      if (str_cmp(s, get_file_attr(fs, i, NAME_ATTR_OFFSET))) {
        ret_val.FCB_index = i;
        break;
      }
    }
    else if (ret_val.empty_index == FP_INVALID) {
      ret_val.empty_index = i;
    }
    if (valid_fcb_traversed == gfilenum && ret_val.empty_index != FP_INVALID) break;
  }
  return ret_val;
}

__device__ u32 get_file_base_addr(FileSystem* fs, u32 fp) {
  /* Given a file pointer, return the base address of the file*/
  u32 file_start_block = get_file_attr(fs, fp, STARTBLK_ATTR_OFFSET, STARTBLK_ATTR_LENGTH);
  return fs->FILE_BASE_ADDRESS + file_start_block * fs->STORAGE_BLOCK_SIZE;
}

__device__ u32 get_block_idx(FileSystem* fs, u32 addr) {
  /* Given an address(in the volume), return the corresponding block ID*/
  // printf("addr=%d;\tbase_addr=%d\t;\tdelta=%d.\n", addr, fs->FILE_BASE_ADDRESS, addr - fs->FILE_BASE_ADDRESS);
  return (addr - fs->FILE_BASE_ADDRESS) / fs->STORAGE_BLOCK_SIZE;
}

__device__ u32 get_file_end_block(FileSystem* fs, u32 fp) {
  /* Given a file pointer, return the end block of the file*/
  u32 file_start_block = get_file_attr(fs, fp, STARTBLK_ATTR_OFFSET, STARTBLK_ATTR_LENGTH);
  u32 file_size = get_file_attr(fs, fp, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
  u32 file_block_count = ceil((float)file_size / fs->STORAGE_BLOCK_SIZE);
  return file_start_block + file_block_count - 1;
}

__device__ void vcb_set(FileSystem* fs, int fp, int val) {
  /* Set the corresponding VCB bits to 0 */
  int file_size = get_file_attr(fs, fp, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
  if (file_size == 0) return;
  int file_start_block = get_file_attr(fs, fp, STARTBLK_ATTR_OFFSET, STARTBLK_ATTR_LENGTH);
  int file_end_block = get_file_end_block(fs, fp);
  int start_byte = file_start_block / 8, start_offset = file_start_block % 8, end_byte = file_end_block / 8, end_offset = file_end_block % 8;
  int start_mask = 0xff >> start_offset, end_mask = (0xff >> (7 - end_offset)) << (7 - end_offset);
  if (start_byte == end_byte) {
    int mask = start_mask & end_mask;
    if (val) {
      fs->volume[start_byte] |= mask;
    }
    else {
      fs->volume[start_byte] &= ~mask;
    }
  }
  else {
    if (val) {
      fs->volume[start_byte] |= start_mask;
      fs->volume[end_byte] |= end_mask;
    }
    else {
      fs->volume[start_byte] &= ~(start_mask);
      fs->volume[end_byte] &= ~(end_mask);
    }
    for (int i = start_byte + 1; i < end_byte; i++) fs->volume[i] = val;
  }
}

__device__ int count_occupied_blocks(int VCB_Byte) {
  int count = 0;
  while (VCB_Byte) {
    count++;
    VCB_Byte &= VCB_Byte - 1;
  }
  return count;
}

__device__ int has_enough_space(FileSystem* fs, int block_size) {
  /* Check if there is enough space to put $block_size blocks*/
  int used_blocks = 0;
  for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++)
    used_blocks += count_occupied_blocks(fs->volume[i]);
  return fs->SUPERBLOCK_SIZE * 8 - used_blocks >= block_size;
}

__device__ int move_file(FileSystem* fs, u32 fp, int new_start_block_idx) {
  /* move file and return the next vacant block id after moving the file*/
  u32 old_file_base_addr = get_file_base_addr(fs, fp);
  u32 new_file_base_addr = fs->FILE_BASE_ADDRESS + new_start_block_idx * fs->STORAGE_BLOCK_SIZE;
  vcb_set(fs, fp, 0); // first, clear the original VCB bits
  set_file_attr(fs, fp, STARTBLK_ATTR_OFFSET, STARTBLK_ATTR_LENGTH, new_start_block_idx);
  vcb_set(fs, fp, 1); // then, set the new VCB bits
  int file_size = get_file_attr(fs, fp, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
  for (int i = 0; i < file_size; i++) {
    fs->volume[new_file_base_addr + i] = fs->volume[old_file_base_addr + i];
  }
  int file_end_block = get_file_end_block(fs, fp);
  return file_end_block + 1;
}

__device__ int fs_compress(FileSystem* fs) {
  /* Compress volume and retrun the first vacant block's index*/
  int next_vacant_block_idx = 0, prev_smallest_start_block = 0;
  for (int i = 0; i < gfilenum - 1; i++) { // we need to exclude the new file created
    int curr_lowset_start_block_idx = 8 * fs->SUPERBLOCK_SIZE;
    int curr_lowest_start_block_fp;
    for (int j = 0; j < fs->FCB_ENTRIES; j++) {
      if (get_file_attr(fs, j, 0, 1) == FCB_INVALID) continue;
      int file_start_block_idx = get_file_attr(fs, j, STARTBLK_ATTR_OFFSET, STARTBLK_ATTR_LENGTH);
      if (file_start_block_idx <= prev_smallest_start_block) continue;
      if (file_start_block_idx < curr_lowset_start_block_idx) {
        curr_lowset_start_block_idx = file_start_block_idx;
        curr_lowest_start_block_fp = j;
      }
    }
    prev_smallest_start_block = curr_lowset_start_block_idx;
    next_vacant_block_idx = move_file(fs, curr_lowest_start_block_fp, next_vacant_block_idx);
  }
  return next_vacant_block_idx;
}

__device__ u32 fs_allocate(FileSystem* fs, int block_num) {
  /* Return the index of first block that can hold $block_num blocks*/
  /* Use first fit algirthm. First, check if the volume has enough space.*/
  /* If there are enough space */
  int count = 0;
  int t_block_idx = 0;
  /* Use first fit to find the starting block index*/
  for (int i = glastblock; i < fs->SUPERBLOCK_SIZE * 8; i++) {
    int curr_block_byte = i / 8, curr_block_offset = i % 8;
    int curr_block_status = fs->volume[curr_block_byte] & (1 << (7 - curr_block_offset));
    if (curr_block_status == 0) {
      count++;
      if (count == 1) t_block_idx = i;
      if (count == block_num) {
        glastblock = i;
        return t_block_idx;
      }
    }
    else {
      count = 0;
    }
  }
  /* If no such block is found, compress volume*/
  glastblock = fs_compress(fs);
  if (glastblock + block_num > fs->SUPERBLOCK_SIZE * 8) {
    printf("No enough space to allocate %d blocks\n", block_num);
    return fs->SUPERBLOCK_SIZE * 8;
  }
  return glastblock;
}

__device__ u32 fs_open(FileSystem* fs, char* s, int op)
{
  /* Implement open operation here */
  int file_name_length = str_len(s);
  if (file_name_length > fs->MAX_FILENAME_SIZE) return FP_INVALID<<1;
  FCBQuery query = search_file(fs, s);
  int ret_val = query.FCB_index;
  if (op == G_WRITE) {
    if (ret_val == FP_INVALID) {
      if (query.empty_index == FP_INVALID) {
        printf("Maximum #file reached.\n");
      }
      else {
        ret_val = query.empty_index;
        set_file_attr(fs, query.empty_index, 0, 1, FCB_VALID);
        set_file_attr(fs, query.empty_index, NAME_ATTR_OFFSET, file_name_length, s); // set file name
        set_file_attr(fs, query.empty_index, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH, 0); // set file size
        set_file_attr(fs, query.empty_index, CREATE_TIME_ATTR_OFFSET, CREATE_TIME_ATTR_LENGTH, gtime); // set create time
        set_file_attr(fs, query.empty_index, MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH, gtime); // set modify time
        gtime++;
        gfilenum++;
      }
    }
  }
  else if (op != G_READ) {
    printf("Invalid operation code.\n");
    ret_val = FP_INVALID;
  }
  return (ret_val << 1) + op;
}

__device__ void fs_read(FileSystem* fs, uchar* output, u32 size, u32 fp)
{
  /* Implement read operation here */
  int mode = fp & 1;
  fp >>= 1;
  if (fp == fs->FCB_ENTRIES || mode != G_READ) {
    printf("File not found.\n");
    return;
  }
  int file_size = get_file_attr(fs, fp, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
  if (size > file_size) {
    printf("Read size exceeds file size.\n");
    return;
  }
  u32 file_base_addr = get_file_base_addr(fs, fp);
  memcpy(output, fs->volume + file_base_addr, size);
}

__device__ u32 fs_write(FileSystem* fs, uchar* input, u32 size, u32 fp)
{
  /* Implement write operation here */
  /* return 1 means error, 0 means success*/
  int mode = fp & 1;
  fp >>= 1;
  if (fp == fs->FCB_ENTRIES || mode != G_WRITE) {
    printf("Invalid fp.\n");
    return 1;
  }
  u32 orgn_file_size = get_file_attr(fs, fp, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
  int orgn_pos_max_size = floor((float)orgn_file_size / fs->STORAGE_BLOCK_SIZE) * fs->STORAGE_BLOCK_SIZE; // the maximum size the previous location can hold 
  u32 new_file_start_block;
  if (size < orgn_file_size) { // If the new size is smaller than the original file, clear VCB and set according to new size
    vcb_set(fs, fp, 0); // clear the VCB bits
    set_file_attr(fs, fp, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH, size); // update file size
    vcb_set(fs, fp, 1); // set the VCB bits
  }
  else if (size > orgn_pos_max_size)
  { // need to reallocate space for file.Clear previous VCB and allocate new space.
    vcb_set(fs, fp, 0);
    set_file_attr(fs, fp, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH, size); // update file size
    int new_block_num = ceil((float)size / fs->STORAGE_BLOCK_SIZE);
    new_file_start_block = fs_allocate(fs, new_block_num);
    if (new_file_start_block == fs->SUPERBLOCK_SIZE * 8) {
      printf("No enough space.\n");
      // roll back
      set_file_attr(fs, fp, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH, orgn_file_size);
      vcb_set(fs, fp, 1);
      return 1;
    }
    set_file_attr(fs, fp, STARTBLK_ATTR_OFFSET, STARTBLK_ATTR_LENGTH, new_file_start_block); // update file start block
    vcb_set(fs, fp, 1);
  }
  else {
    set_file_attr(fs, fp, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH, size); // update file size
  }
  u32 new_file_base_addr = get_file_base_addr(fs, fp);
  // write $size bytes to the new starting position 
  memcpy(fs->volume + new_file_base_addr, input, size);
  set_file_attr(fs, fp, MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH, gtime); // set modify time
  gtime++;
  return 0;
}

__device__ void fs_gsys(FileSystem* fs, int op)
{
  /* Implement LS_D and LS_S operation here */
  if (op == LS_D) {
    /* Since at each time only one file is modified, two files cannot have the same modification time.*/
    printf("===sort by modified time===\n");
    int prev_youngest_modtime = gtime;
    for (int i = 0; i < gfilenum; i++) {
      int curr_youngest_modtime = -1;
      char* curr_file_name;
      for (int j = 0; j < fs->FCB_ENTRIES; j++) {
        if (get_file_attr(fs, j, 0, 1) == FCB_INVALID) continue;
        int file_modtime = get_file_attr(fs, j, MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH);
        if (file_modtime >= prev_youngest_modtime) continue;
        if (file_modtime > curr_youngest_modtime) {
          curr_youngest_modtime = file_modtime;
          curr_file_name = get_file_attr(fs, j, NAME_ATTR_OFFSET);

        }
      }
      printf("%-20s\n", curr_file_name);
      prev_youngest_modtime = curr_youngest_modtime;
    }
  }
  else if (op == LS_S) {
    printf("===sort by file size===\n");
    int prev_max_size = fs->MAX_FILE_SIZE, prev_oldest_create_time = -1;
    for (int i = 0; i < gfilenum; i++) {
      int curr_max_size = -1, curr_oldest_create_time = gtime;
      char* curr_file_name;
      for (int j = 0; j < fs->FCB_ENTRIES; j++) {
        if (get_file_attr(fs, j, 0, 1) == FCB_INVALID) continue;
        int file_size = get_file_attr(fs, j, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
        int file_create_time = get_file_attr(fs, j, CREATE_TIME_ATTR_OFFSET, CREATE_TIME_ATTR_LENGTH);
        if (file_size > prev_max_size || (file_size == prev_max_size) && file_create_time <= prev_oldest_create_time) continue;
        if (file_size > curr_max_size || (file_size == curr_max_size && file_create_time < curr_oldest_create_time)) {
          curr_max_size = file_size;
          curr_oldest_create_time = file_create_time;
          curr_file_name = get_file_attr(fs, j, NAME_ATTR_OFFSET);
        }
      }
      printf("%-20s\t %d\n", curr_file_name, curr_max_size);
      prev_max_size = curr_max_size;
      prev_oldest_create_time = curr_oldest_create_time;
    }
  }
  else if (op == LS_DR) {
    printf("===sort by start block index===\n");
    int prev_smallest_start_block = -1;
    for (int i = 0; i < gfilenum; i++) {
      int curr_smallest_start_block = fs->SUPERBLOCK_SIZE * 8;
      int curr_fp;
      for (int j = 0; j < fs->FCB_ENTRIES; j++) {
        if (get_file_attr(fs, j, 0, 1) == FCB_INVALID) continue;
        int file_startblock = get_file_attr(fs, j, STARTBLK_ATTR_OFFSET, STARTBLK_ATTR_LENGTH);
        if (file_startblock <= prev_smallest_start_block) continue;
        if (file_startblock < curr_smallest_start_block) {
          curr_smallest_start_block = file_startblock;
          curr_fp = j;
        }
      }
      char* curr_file_name = get_file_attr(fs, curr_fp, NAME_ATTR_OFFSET);
      int curr_file_modtime = get_file_attr(fs, curr_fp, MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH);
      int curr_file_size = get_file_attr(fs, curr_fp, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
      int curr_file_createtime = get_file_attr(fs, curr_fp, CREATE_TIME_ATTR_OFFSET, CREATE_TIME_ATTR_LENGTH);
      int curr_file_startblock = get_file_attr(fs, curr_fp, STARTBLK_ATTR_OFFSET, STARTBLK_ATTR_LENGTH);
      int curr_file_endblock = get_file_end_block(fs, curr_fp);
      printf("#%4d FCB Index:%-4d\tFile name:%-20s\tSize:%-10d\tStarts on block:%-5d\tEnds on block:%-5d\tTime created:%-5d\tTime modified:%-5d\n", i, curr_fp, curr_file_name, curr_file_size, curr_file_startblock, curr_file_endblock, curr_file_createtime, curr_file_modtime);
      prev_smallest_start_block = curr_smallest_start_block;
    }
  }
  else {
    printf("Invalid operation code [%d]\n", op);
  }
}

__device__ void fs_gsys(FileSystem* fs, int op, char* s)
{
  /* Implement rm operation here */
  if (op != RM) {
    printf("Invalid operation [%d].\n", op);
    return;
  }
  FCBQuery query = search_file(fs, s);
  if (query.FCB_index == -1) {
    printf("No file named %s to delete.\n", s);
  }
  vcb_set(fs, query.FCB_index, 0);
  set_file_attr(fs, query.FCB_index, 0, 1, FCB_INVALID);
  gfilenum--;
}
