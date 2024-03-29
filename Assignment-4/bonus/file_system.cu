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
#define PARDIR_ATTR_LENGTH 2
#define SIZE_ATTR_LENGTH 4
#define MISC_ATTR_LENGTH 2
#define FILE 0x8000
#define FCB_INVALID 0x0000
#define DIR 0xc000
#define FP_INVALID 1024
#define FILE_STATUS(x) x&0xc000
#define FORMAT_PARENT_FP(x) x&0x03ff

__device__ __managed__ u32 gtime = 0;       // increasing. larger means newer. Initially set to -1
__device__ __managed__ u32 gfilenum = 0;    // number of files present in the file system
__device__ __managed__ u32 glastblock = 0;  // used in next-fit algorithm
__device__ __managed__ u32 gcwd = 0;        // current working directory. Default is root. root directory is always at fcb#0
__device__ __managed__ u32 glevel = 1;      // the root directory is at level 0, maxmimum level is 3.

__device__ void fcb_init(FileSystem* fs) {
  for (u32 i = 0; i < fs->FCB_ENTRIES; i++) {
    set_file_attr(fs, i, 0, MISC_ATTR_LENGTH, FCB_INVALID);// MSB in the first byte of FCB is valid bit. 0 indicates invalid.
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

  // create the fcb for root directory
  fs_gsys(fs, MKDIR, "/\0");
  gtime = 0; // reset gtime
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
  const char* s;
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
  int result = 0;
  memcpy(&result, fs->volume + fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + attr_offset, attr_length);
  return result;
}

__device__ void set_file_attr(FileSystem* fs, u32 fp, int attr_offset, int attr_length, int value) {
  /* Set file attribute. */
  memcpy(fs->volume + fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + attr_offset, &value, attr_length);
}

__device__ void set_file_attr(FileSystem* fs, u32 fp, int attr_offset, int attr_length, char* value) {
  /* Set file attribute. This reloaded function is for setting file name only. */
  memcpy(fs->volume + fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + attr_offset, value, attr_length);
  memset(fs->volume + fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + attr_offset + attr_length, 0, 1);
}

__device__ void append_parent_content(FileSystem* fs, char* s) {
  int parent_fp = gcwd;
  if (str_cmp(s, ".\0")) return;
  int new_filename_length = str_len(s);
  int orgn_parent_size = get_file_attr(fs, parent_fp, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
  char* new_parent_content = new char[orgn_parent_size + new_filename_length + 1];
  if (orgn_parent_size) {
    fs_read(fs, (uchar*)new_parent_content, orgn_parent_size, (parent_fp << 1) + G_READ);
  }
  memcpy(new_parent_content + orgn_parent_size, s, new_filename_length);
  memset(new_parent_content + orgn_parent_size + new_filename_length, 0, 1);
  gtime--; // we need to decrease here, because the following fs_write will increase it.
  fs_write(fs, (uchar*)new_parent_content, orgn_parent_size + new_filename_length + 1, (parent_fp << 1) + G_WRITE); // update name info
  delete[] new_parent_content;
}

__device__ void pop_parent_content(FileSystem* fs, char* s) {
  int parent_fp = gcwd;
  if (str_cmp(s, ".\0")) return;
  int grandparent_fp = get_file_attr(fs, parent_fp, 0, MISC_ATTR_LENGTH) & 0x3fff;
  int grandparent_modtime  = get_file_attr(fs, grandparent_fp, MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH);
  int pop_filelength = str_len(s) + 1;
  int orgn_parent_size = get_file_attr(fs, parent_fp, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
  char* new_parent_content = new char[orgn_parent_size];
  fs_read(fs, (uchar*)new_parent_content, orgn_parent_size, (parent_fp << 1) + G_READ);
  int bits_traversed = 0;
  while (true) {
    char* tmp_char = &new_parent_content[bits_traversed];
    if (str_cmp(tmp_char, s)) {
      memcpy(new_parent_content + bits_traversed, new_parent_content + bits_traversed + pop_filelength, orgn_parent_size - bits_traversed - pop_filelength);
      orgn_parent_size -= pop_filelength;
      // for (int i = 0; i < orgn_parent_size; i++) {
      //   printf("Byte%d is %c\n", i, new_parent_content[i]);
      // }
      break;
    }
    else {
      bits_traversed += str_len(tmp_char) + 1;
    }
  }
  fs_write(fs, (uchar*)new_parent_content, orgn_parent_size, (parent_fp << 1) + G_WRITE); // update name info
  set_file_parent_attr(fs, parent_fp, MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH, grandparent_modtime); // reset grandparent modification time.
  gtime--;
  delete[] new_parent_content;
}

__device__ void set_file_parent_attr(FileSystem* fs, u32 fp, int attr_offset, int attr_length, int value) {
  int raw_parent_fp, parent_fp;
  if (fp == 0) return;
  raw_parent_fp = get_file_attr(fs, fp, 0, PARDIR_ATTR_LENGTH);
  parent_fp = raw_parent_fp&0x3fff;
  set_file_attr(fs, parent_fp, attr_offset, attr_length, value);
}

__device__ FCBQuery search_file(FileSystem* fs, char* s) {
  /* Find the FCB of the file with name s.
   * Return a query result containing the fcb index (if found) and first empty fcb index (if found).
   */
  FCBQuery ret_val = { FP_INVALID, FP_INVALID };
  int valid_fcb_traversed = 0;
  for (u32 i = 0; i < fs->FCB_ENTRIES; i++) {
    int file_status = get_file_attr(fs, i, 0, MISC_ATTR_LENGTH);
    int file_parent = file_status & 0x3fff;
    if (file_status >> 15) { // valid bit is set
      valid_fcb_traversed++;
      if (str_cmp(s, get_file_attr(fs, i, NAME_ATTR_OFFSET)) && file_parent == gcwd) {
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

__device__ int move_file(FileSystem* fs, u32 fp, int new_start_block_idx) {
  /* move file and return the next vacant block id after moving the file*/
  u32 old_file_base_addr = get_file_base_addr(fs, fp);
  u32 new_file_base_addr = fs->FILE_BASE_ADDRESS + new_start_block_idx * fs->STORAGE_BLOCK_SIZE;
  vcb_set(fs, fp, 0); // first, clear the original VCB bits
  set_file_attr(fs, fp, STARTBLK_ATTR_OFFSET, STARTBLK_ATTR_LENGTH, new_start_block_idx);
  vcb_set(fs, fp, 1); // then, set the new VCB bits
  int file_size = get_file_attr(fs, fp, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
  memcpy(fs->volume + new_file_base_addr, fs->volume + old_file_base_addr, file_size);
  int file_end_block = get_file_end_block(fs, fp);
  return file_end_block + 1;
}

__device__ int fs_compress(FileSystem* fs) {
  /* Compress volume and retrun the first vacant block's index*/
  u32* fcb_arr = new u32[gfilenum];
  u32* startblk_arr = new u32[gfilenum];
  int files_found = 0;
  for (int i = 0; i < fs->FCB_ENTRIES; i++) {
    if (get_file_attr(fs, i, 0, MISC_ATTR_LENGTH) >> 15) {
      fcb_arr[files_found] = i;
      startblk_arr[files_found] = get_file_attr(fs, i, STARTBLK_ATTR_OFFSET, STARTBLK_ATTR_LENGTH);
      files_found++;
    }
    if (files_found == gfilenum) break;
  }
  for (int i = 0; i < gfilenum; i++) {
    int curr_min = i;
    for (int j = i + 1; j < gfilenum; j++) {
      if (startblk_arr[curr_min] > startblk_arr[j]) {
        curr_min = j;
      }
    }
    int tmp = startblk_arr[i];
    startblk_arr[i] = startblk_arr[curr_min];
    startblk_arr[curr_min] = tmp;
    tmp = fcb_arr[i];
    fcb_arr[i] = fcb_arr[curr_min];
    fcb_arr[curr_min] = tmp;
  }
  int prev_smallest_start_block = 0;
  for (int i = 0; i < gfilenum; i++) {
    if (startblk_arr[i] != prev_smallest_start_block) {
      prev_smallest_start_block = move_file(fs, fcb_arr[i], prev_smallest_start_block);
    }
  }
  delete[] fcb_arr;
  delete[] startblk_arr;
  return prev_smallest_start_block;
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
  if (file_name_length > fs->MAX_FILENAME_SIZE) return FP_INVALID << 1;
  FCBQuery query = search_file(fs, s);
  int ret_val = query.FCB_index;
  if (op == G_WRITE) {
    if (ret_val == FP_INVALID) {
      if (query.empty_index == FP_INVALID) {
        printf("Maximum #file reached.\n");
      }
      else {
        ret_val = query.empty_index;
        int parent_fp = gcwd;
        int new_file_misc = FILE + parent_fp;
        append_parent_content(fs, s);
        set_file_attr(fs, query.empty_index, 0, MISC_ATTR_LENGTH, new_file_misc);
        set_file_attr(fs, query.empty_index, NAME_ATTR_OFFSET, file_name_length, s); // set file name
        set_file_attr(fs, query.empty_index, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH, 0); // set file size
        set_file_attr(fs, query.empty_index, CREATE_TIME_ATTR_OFFSET, CREATE_TIME_ATTR_LENGTH, gtime); // set create time
        set_file_attr(fs, query.empty_index, MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH, gtime); // set modify time
        set_file_parent_attr(fs, query.empty_index, MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH, gtime); // update direct parents'
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
  int orgn_pos_max_size = ceil((float)orgn_file_size / fs->STORAGE_BLOCK_SIZE) * fs->STORAGE_BLOCK_SIZE; // the maximum size the previous location can hold 
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
  memcpy(fs->volume + new_file_base_addr, input, size);  // write $size bytes to the new starting position 
  set_file_attr(fs, fp, MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH, gtime); // update modify time
  set_file_parent_attr(fs, fp, MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH, gtime); // set modify time of parent
  gtime++;
  return 0;
}

__device__ void count_cwd_filenum(FileSystem* fs, char* cwd_content, int* cwd_file_count, int cwd_content_size) {
  while (cwd_content_size) {
    char* curr_filename = cwd_content;
    int curr_filename_length = str_len(curr_filename) + 1;
    (*cwd_file_count)++;
    cwd_content_size -= curr_filename_length;
    cwd_content += curr_filename_length;
  }
}

__device__ void fs_gsys(FileSystem* fs, int op)
{
  /* Implement LS_D and LS_S operation here */
  switch (op)
  {
  case LS_D:
  {
    int cwd_file_count = 0;
    int cwd_size = get_file_attr(fs, gcwd, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
    char* cwd_conent = new char[cwd_size];
    fs_read(fs, (uchar*)cwd_conent, cwd_size, (gcwd << 1) + G_READ);
    count_cwd_filenum(fs, cwd_conent, &cwd_file_count, cwd_size);
    if (cwd_file_count == 0) {
      printf("===Sort %d files by modified time===\n", cwd_file_count);
      break;
    }

    int* fcb_arr = new int[cwd_file_count];
    int* modtime_arr = new int[cwd_file_count];
    int bytes_traversed = 0;
    for (int i = 0; i < cwd_file_count; i++) {
      char* curr_file_name = &cwd_conent[bytes_traversed];
      int curr_file_name_length = str_len(curr_file_name);
      FCBQuery curr_query = search_file(fs, curr_file_name);
      fcb_arr[i] = curr_query.FCB_index;
      modtime_arr[i] = get_file_attr(fs, curr_query.FCB_index, MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH);
      bytes_traversed += (str_len(curr_file_name) + 1);
    }
    for (int i = 0; i < cwd_file_count; i++) {
      if (cwd_file_count == 1) break;
      int curr_max = i;
      for (int j = i + 1; j < cwd_file_count; j++) {
        if (modtime_arr[curr_max] < modtime_arr[j]) {
          curr_max = j;
        }
      }
      int tmp = modtime_arr[i];
      modtime_arr[i] = modtime_arr[curr_max];
      modtime_arr[curr_max] = tmp;
      tmp = fcb_arr[i];
      fcb_arr[i] = fcb_arr[curr_max];
      fcb_arr[curr_max] = tmp;
    }
    printf("===Sort %d files by modified time===\n", cwd_file_count);
    for (int i = 0; i < cwd_file_count; i++) {
      int file_status = get_file_attr(fs, fcb_arr[i], 0, MISC_ATTR_LENGTH);
      char is_dir = (file_status & 0x4000) ? 'd' : ' ';
      printf("%-20s\t%-8d\t%c\n", get_file_attr(fs, fcb_arr[i], NAME_ATTR_OFFSET), modtime_arr[i], is_dir);
    }
    delete[] fcb_arr;
    delete[] modtime_arr;
    delete[] cwd_conent;
    break;
  }
  case LS_S:
  {
    int cwd_file_count = 0;
    int cwd_size = get_file_attr(fs, gcwd, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
    char* cwd_conent = new char[cwd_size];
    fs_read(fs, (uchar*)cwd_conent, cwd_size, (gcwd << 1) + G_READ);
    count_cwd_filenum(fs, cwd_conent, &cwd_file_count, cwd_size);
    if (cwd_file_count == 0) {
      printf("===Sort %d files by size===\n", cwd_file_count);
      break;
    }
    int* fcb_arr = new int[cwd_file_count];
    int* size_arr = new int[cwd_file_count];
    int bytes_traversed = 0;
    for (int i = 0; i < cwd_file_count; i++) {
      char* curr_file_name = &cwd_conent[bytes_traversed];
      int curr_file_name_length = str_len(curr_file_name);
      FCBQuery curr_query = search_file(fs, curr_file_name);
      fcb_arr[i] = curr_query.FCB_index;
      size_arr[i] = get_file_attr(fs, curr_query.FCB_index, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
      bytes_traversed += (str_len(curr_file_name) + 1);
    }
    for (int i = 0; i < cwd_file_count; i++) {
      if (cwd_file_count == 1) break;
      int curr_max = i;
      for (int j = i + 1; j < cwd_file_count; j++) {
        if (size_arr[curr_max] < size_arr[j] ||
          (size_arr[curr_max] == size_arr[j] && (get_file_attr(fs, fcb_arr[curr_max], CREATE_TIME_ATTR_OFFSET, CREATE_TIME_ATTR_LENGTH) > get_file_attr(fs, fcb_arr[j], CREATE_TIME_ATTR_OFFSET, CREATE_TIME_ATTR_LENGTH)))
          ) {
          curr_max = j;
        }
      }
      int temp = size_arr[i];
      size_arr[i] = size_arr[curr_max];
      size_arr[curr_max] = temp;
      temp = fcb_arr[i];
      fcb_arr[i] = fcb_arr[curr_max];
      fcb_arr[curr_max] = temp;
    }
    printf("===Sort %d files by size===\n", cwd_file_count);
    for (int i = 0; i < cwd_file_count; i++) {
      int file_status = get_file_attr(fs, fcb_arr[i], 0, MISC_ATTR_LENGTH);
      char is_dir = (file_status & 0x4000) ? 'd' : ' ';
      printf("%-20s\t%-8d\t%c\n", get_file_attr(fs, fcb_arr[i], NAME_ATTR_OFFSET), get_file_attr(fs, fcb_arr[i], SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH), is_dir);
    }
    delete[] cwd_conent;
    delete[] fcb_arr;
    delete[] size_arr;
    break;
  }
  case CD_P:
  {
    int raw_pd = get_file_attr(fs, gcwd, 0, 2);
    gcwd = raw_pd & 0x3FFF;
    if (gcwd!=1) glevel--;
    break;
  }
  case PWD:
  {
    int tcwd = gcwd;
    if (tcwd == 0) {
      printf("/\n");
      break;
    }
    int tlevel = glevel;
    int* working_dir = new int[tlevel - 1];
    for (int i = 0; i < tlevel - 1; i++) {
      working_dir[i] = tcwd;
      int tparent = get_file_attr(fs, tcwd, 0, PARDIR_ATTR_LENGTH);
      tcwd = FORMAT_PARENT_FP(tparent);
    }
    for (int i = tlevel - 1; i > 0; i--)
      printf("/%s", get_file_attr(fs, working_dir[i - 1], NAME_ATTR_OFFSET));
    printf("\n");
    delete[] working_dir;
    break;
  }
  default:
  {
    printf("Invalid operation code [%d]\n", op);
    break;
  }
  }
}

__device__ void fs_gsys(FileSystem* fs, int op, char* s)
{
  /* Implement rm operation here */
  FCBQuery query = search_file(fs, s);
  if (query.FCB_index == FP_INVALID && op != MKDIR) {
    printf("No file named %s");
    return;
  }
  else if (query.FCB_index != FP_INVALID && op == MKDIR) {
    printf("File named %s already exists\n", s);
    return;
  }
  switch (op) {
  case RM:
  {
    int file_status = get_file_attr(fs, query.FCB_index, 0, MISC_ATTR_LENGTH);
    if (((file_status & 0x4000) >> 14)) {
      printf("Cannot delete a directory using RM,\n");
      return;
    }
    vcb_set(fs, query.FCB_index, 0);
    set_file_attr(fs, query.FCB_index, 0, MISC_ATTR_LENGTH, FCB_INVALID);
    pop_parent_content(fs, s);
    gfilenum--;
    gtime++;
    break;
  }
  case RM_RF:
  {
    int file_status = get_file_attr(fs, query.FCB_index, 0, MISC_ATTR_LENGTH), ttime = gtime;
    if (((file_status & 0x4000) >> 14) == 0) {
      printf("Cannot delete a file using RM_RF.\n");
      return;
    }
    else if (query.FCB_index == 0) {
      printf("Cannot remove root directory.\n");
      return;
    }
    int dir_size = get_file_attr(fs, query.FCB_index, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH), bytes_traversed = 0;
    char* dir_content = new char[dir_size];
    fs_read(fs, (uchar*) dir_content, dir_size, (query.FCB_index << 1) + G_READ);
    fs_gsys(fs, CD, s); // cd into the directory
    while (bytes_traversed != dir_size) { // delete all files/subdirectories
      char* t_name = &dir_content[bytes_traversed];
      FCBQuery t_query = search_file(fs, t_name);
      int t_fp = t_query.FCB_index;
      int file_status = get_file_attr(fs, t_fp, 0, MISC_ATTR_LENGTH);
      if ((file_status&0x4000) >> 14) {
        fs_gsys(fs, RM_RF, t_name); // recursively remove all the files within the directory
      }
      else {
        fs_gsys(fs, RM, t_name);
      }
      bytes_traversed += str_len(t_name)+1;
    }
    fs_gsys(fs, CD_P); // cd back to the parent directory
    vcb_set(fs, query.FCB_index, 0); // clear vcb bits
    set_file_attr(fs, query.FCB_index, 0, MISC_ATTR_LENGTH, FCB_INVALID); // invalidate fcb entry
    gtime = ttime;
    pop_parent_content(fs, s); // remove the directory from the parent directory
    gfilenum--; // decrease by 1 (deleting the directory itself)
    gtime++;
    delete[] dir_content;
    break;
  }
  case CD:
  {
    if (get_file_attr(fs, query.FCB_index, 0, MISC_ATTR_LENGTH) & 0x4000 == 0) {
      printf("Cannot CD into a file.\n");
    }
    gcwd = query.FCB_index;
    glevel++;
    break;
  }
  case MKDIR:
  {
    if (gfilenum == fs->MAX_FILE_NUM || query.empty_index == FP_INVALID) {
      printf("Cannot create more directoryes.\n");
      return;
    }
    int new_fcb = query.empty_index;
    append_parent_content(fs, s);
    int new_misc_info = DIR + gcwd;
    set_file_attr(fs, new_fcb, NAME_ATTR_OFFSET, str_len(s), s);
    set_file_attr(fs, new_fcb, 0, MISC_ATTR_LENGTH, new_misc_info);
    set_file_attr(fs, new_fcb, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH, 0);
    set_file_attr(fs, new_fcb, CREATE_TIME_ATTR_OFFSET, CREATE_TIME_ATTR_LENGTH, gtime);
    set_file_attr(fs, new_fcb, MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH, gtime);
    set_file_parent_attr(fs, new_fcb, MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH, gtime);
    gfilenum++;
    gtime++;
    break;
  }
  default:
  {
    printf("Invalid operation code [%d]\n", op);
    break;
  }
  }
}

__device__ void file_diagnose(FileSystem* fs, u32 fp) {
  int fcb_status = get_file_attr(fs, fp, 0, MISC_ATTR_LENGTH);
  if ((fcb_status >> 15) == FCB_INVALID) {
    return;
  }
  char is_file = ((fcb_status & 0x4000) >> 14) ? 'd' : 'f';
  int parent_fcb = fcb_status & 0x3fff;
  char* parent_name = get_file_attr(fs, parent_fcb, NAME_ATTR_OFFSET);
  char* file_name = get_file_attr(fs, fp, NAME_ATTR_OFFSET);
  short file_modtime = get_file_attr(fs, fp, MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH);
  int file_size = get_file_attr(fs, fp, SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
  int file_createtime = get_file_attr(fs, fp, CREATE_TIME_ATTR_OFFSET, CREATE_TIME_ATTR_LENGTH);
  short file_startblock, file_endblock;
  if (file_size) {
    file_startblock = get_file_attr(fs, fp, STARTBLK_ATTR_OFFSET, STARTBLK_ATTR_LENGTH);
    file_endblock = get_file_end_block(fs, fp);
  }
  else {
    file_startblock = -1;
    file_endblock = -1;
  }
  printf("Index:%-4d\tFile name:%-20s\tType:%c\tParent:%-20s\tSize:%-10d\tStart block:%-5d\tEnd block:%-5d\tTime created:%-5d\tTime modified:%-5d\n", fp, file_name, is_file, parent_name, file_size, file_startblock, file_endblock, file_createtime, file_modtime);
}

__device__ void fs_diagnose(FileSystem* fs) {
  printf("===File System Diagnose===\n");
  printf("Current system time:%d\n", gtime);
  printf("Current file  count:%d\n", gfilenum);
  printf("Current working dir:");
  fs_gsys(fs, PWD);
  for (int i = 0; i < fs->FCB_ENTRIES; i++) {
    file_diagnose(fs, i);
  }
  printf("==========================\n");
}