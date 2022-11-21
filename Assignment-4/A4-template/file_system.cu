#include "file_system.h"
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

__device__ __managed__ u32 gtime = 0; // increasing. larger means newer
__device__ __managed__ int gfilenum = 0;

__device__ void fcb_init(FileSystem* fs) {
  for (u32 i = 0; i < fs->FCB_ENTRIES; i++) {
    fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE] = 0x00; // MSB in the first byte of FCB is valid bit. 0 indicates invalid.
  }
}

__device__ void superblock_init(FileSystem* fs) {
  // Initialize superblock. In my implementation, 0 means free and 1 means used.
  for (uchar i = 0; i < fs->SUPERBLOCK_SIZE; i++) {
    fs->volume[i] = 0;
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

__device__ int cmp_str(char* str1, char* str2) {
  /* Compare two strings. Return 1 if they are the same. */
  while (*str1 != '\0' && *str2 != '\0') {
    if (*str1 != *str2) return 0;
    str1++;
    str2++;
  }
  if (*str1 == '\0' && *str2 == '\0') return 1;
  else return 0;
}

__device__ char* read_file_attr(FileSystem* fs, u32 fcb_offset_addr) {
  int file_name_len = 0;
  while (fs->volume[fcb_offset_addr + file_name_len] != '\0') {
    file_name_len++;
  }
  file_name_len++;
  char file_name[file_name_len];
  memcpy(file_name, fs->volume + fcb_offset_addr, file_name_len);
  return file_name;
}

__device__ int read_file_attr(FileSystem* fs, u32 fcb_offset_addr, int length) {
  /* Read file attribute from FCB. */
  printf("[Read Attr from %d length %d]\n", fcb_offset_addr, length);
  int result = 0;
  for (int i = 0; i < length; i++) {
    printf("reading byte %d:\t curr_result:%d\t", i, result);
    result = result << 8;
    printf("result << 8: %d\t curr_byte: %d\t", result, (int)fs->volume[fcb_offset_addr + i]);
    result += (int)fs->volume[fcb_offset_addr + i];
    printf("result after shifting: %d\n", result);
  }
  return result;
}

__device__ void set_file_attr(FileSystem* fs, u32 fcb_offset_addr, int length, int value) {
  /* Set file attribute. */
  for (int i = length - 1; i >= 0; i--) {
    fs->volume[fcb_offset_addr + i] = value & 0xFF;
    value = value >> 8;
  }
}

__device__ void set_file_attr(FileSystem* fs, u32 fcb_offset_addr, char* value) {
  /* Set file attribute. */
  int count = 0;
  while (value != '\0') {
    fs->volume[fcb_offset_addr + count] = value[count];
    count++;
    if (count == fs->MAX_FILENAME_SIZE) break;
  }
}

__device__ FCBQuery search_file(FileSystem* fs, char* s) {
  /* Find the FCB of the file with name s.
   * Return a query result containing the fcb index (if found) and first empty fcb index (if found).
   */
  FCBQuery ret_val = { -1, -1 };
  for (u32 i = 0; i < fs->FCB_ENTRIES; i++) {
    if (fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE] & FCB_VALID == FCB_VALID) { // valid bit is set
      if (cmp_str(s, (char*)fs->volume + fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + 1)) {
        ret_val.FCB_index = i;
      }
    }
    else if (ret_val.FCB_index == -1) {
      ret_val.empty_index = i;
    }
    if (ret_val.FCB_index != -1 && ret_val.empty_index != -1) break;
  }
  return ret_val;
}

__device__ u32 get_file_base_addr(FileSystem* fs, u32 fp) {
  /* Given a file pointer, return the base address of the file*/
  u32 file_start_block = read_file_attr(fs, fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + STARTBLK_ATTR_OFFSET, STARTBLK_ATTR_LENGTH);
  return fs->volume[fs->FILE_BASE_ADDRESS + file_start_block * fs->STORAGE_BLOCK_SIZE];
}

__device__ u32 addr2block(FileSystem* fs, u32 addr) {
  /* Given an address(in the volume), return the corresponding block ID*/
  printf("addr=%d; base_addr=%d\n; delta=%d", addr, fs->FILE_BASE_ADDRESS, addr - fs->FILE_BASE_ADDRESS);
  return (addr - fs->FILE_BASE_ADDRESS) / fs->STORAGE_BLOCK_SIZE;
}

__device__ void vcb_set(FileSystem* fs, u32 fp, int val) {
  /* Set the corresponding VCB bits to 0 */
  u32 file_fcb_base_addr = fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE;
  int file_start_block = read_file_attr(fs, file_fcb_base_addr + STARTBLK_ATTR_OFFSET, STARTBLK_ATTR_LENGTH);
  int file_size = read_file_attr(fs, file_fcb_base_addr + SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
  u32 file_start_addr = fs->volume[fs->FILE_BASE_ADDRESS + file_start_block * fs->STORAGE_BLOCK_SIZE];
  int file_end_block = addr2block(fs, file_start_addr + file_size);
  int block_start_byte = file_start_block / 8, block_end_byte = file_end_block / 8, block_start_offset = file_start_block % 8, block_end_offset = file_end_block % 8;
  if (val == 0) {
    fs->volume[block_start_byte] &= (0xff >> (8 - block_start_offset)) << (8 - block_start_offset);
    fs->volume[block_end_byte] &= 0xff >> (block_end_offset + 1);
  }
  else {
    fs->volume[block_start_byte] |= 0xff >> block_start_offset;
    fs->volume[block_end_byte] |= (0xff >> (7 - block_end_offset)) << (7 - block_end_offset);
  }
  for (int i = block_start_byte + 1; i < block_end_byte; i++) fs->volume[i] = val;
}

__device__ u32 fs_compress(FileSystem* fs) {
  return (u32)0;
}

__device__ u32 fs_allocate(FileSystem* fs, u32 fp, int block_num) {
  int largest_start_block = -1, largest_size = -1;
  for (int i = 0; i < fs->FCB_ENTRIES; i++) {
    u32 file_fcb_base_addr = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
    int file_start_block = read_file_attr(fs, file_fcb_base_addr + STARTBLK_ATTR_OFFSET, STARTBLK_ATTR_LENGTH);
    if (file_start_block > largest_start_block) {
      largest_start_block = file_start_block;
      largest_size = read_file_attr(fs, file_fcb_base_addr + SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
    }
    int largest_end_block = addr2block(fs, fs->FILE_BASE_ADDRESS + largest_start_block * fs->STORAGE_BLOCK_SIZE + largest_size);
    int new_end_block = largest_end_block + block_num;
    if (new_end_block / 8 > fs->SUPERBLOCK_SIZE) {
      largest_end_block = fs_compress(fs);
      new_end_block = largest_end_block + block_num;
      if (new_end_block / 8 > fs->SUPERBLOCK_SIZE) {
        printf("No enough space for allocation!\n");
        return -1;
      }
      else return largest_end_block + 1;
    }
    else return largest_end_block + 1;
  }

  return (u32)0;
}

__device__ u32 fs_open(FileSystem* fs, char* s, int op)
{
  /* Implement open operation here */
  FCBQuery query = search_file(fs, s);
  if (query.FCB_index != -1) return query.FCB_index;
  if (op == G_READ) {
    return -1;
  }
  else if (op == G_WRITE) {
    if (query.empty_index == -1) return -1; // maximum # of files reached
    else {
      u32 FCB_base_addr = fs->SUPERBLOCK_SIZE + query.empty_index * fs->FCB_SIZE;
      set_file_attr(fs, FCB_base_addr + NAME_ATTR_OFFSET, s); // set file name
      set_file_attr(fs, FCB_base_addr + SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH, 0); // set file size
      set_file_attr(fs, FCB_base_addr + CREATE_TIME_ATTR_OFFSET, CREATE_TIME_ATTR_LENGTH, gtime); // set create time
      set_file_attr(fs, FCB_base_addr + MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH, gtime); // set modify time
      gtime++;
      gfilenum++;
      printf("Currently there are %d files", gfilenum);
      return query.empty_index;
    }
  }
  else {
    printf("Invalid operation code");
  }

}

__device__ void fs_read(FileSystem* fs, uchar* output, u32 size, u32 fp)
{
  /* Implement read operation here */
  if (fp == -1) {
    printf("File not found");
    return;
  }
  int file_size = read_file_attr(fs, fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
  if (size > file_size) {
    printf("Read size exceeds file size");
    return;
  }
  u32 file_start_addr = get_file_base_addr(fs, fp);
  for (int i = 0; i < size; i++) {
    output[i] = fs->volume[file_start_addr + i];
  }
}

__device__ u32 fs_write(FileSystem* fs, uchar* input, u32 size, u32 fp)
{
  /* Implement write operation here */
  if (fp == -1) {
    printf("Invalid fp");
    return;
  }
  u32 file_fcb_base_addr = fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE;
  u32 orgn_file_size = read_file_attr(fs, fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
  u32 orgn_file_base_addr = get_file_base_addr(fs, fp);
  int orgn_pos_max_size = floor((float)orgn_file_size / fs->STORAGE_BLOCK_SIZE) * fs->STORAGE_BLOCK_SIZE; // the maximum size the previous location can hold 
  u32 new_file_base_addr = orgn_file_base_addr;
  u32 new_file_start_block = addr2block(fs, new_file_base_addr);
  printf("originally file is %d Bytes\n", orgn_file_size);
  printf("The original space can store up to %d Bytes of file\n", orgn_pos_max_size);
  if (size < orgn_file_size) {
    vcb_set(fs, fp, 0); // clear the VCB bits
    set_file_attr(fs, file_fcb_base_addr + SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH, size); // update file size
    vcb_set(fs, fp, 1); // set the VCB bits
  }
  else if (size > orgn_pos_max_size)
  {
    int new_block_num = ceil((float)size / fs->STORAGE_BLOCK_SIZE);
    vcb_set(fs, fp, 0);
    set_file_attr(fs, file_fcb_base_addr + SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH, size); // update file size
    new_file_base_addr = fs_allocate(fs, fp, new_block_num);
    new_file_start_block = addr2block(fs, new_file_base_addr);
    set_file_attr(fs, file_fcb_base_addr + STARTBLK_ATTR_OFFSET, STARTBLK_ATTR_LENGTH, new_file_start_block); // update file start block
    vcb_set(fs, fp, 1);
  }
  else {
    set_file_attr(fs, file_fcb_base_addr + SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH, size); // update file size
  }
  // write $size bytes to the new starting position 
  // set_file_attr(fs, fcb_base_addr + SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH, size); // set file size
  for (int i = 0; i < size; i++)
    fs->volume[new_file_base_addr + i] = input[i];
  set_file_attr(fs, file_fcb_base_addr + MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH, gtime); // set modify time
  gtime++;
}

__device__ void fs_gsys(FileSystem* fs, int op)
{
  /* Implement LS_D and LS_S operation here */
  if (op == LS_D) {
    printf("===sort by modified time===\n");
    int prev_oldest_modtime = -1;
    for (int i = 0; i < gfilenum; i++) {
      int curr_oldest_modtime = gtime;
      char* curr_file_name;
      for (int j = 0; j < fs->FCB_ENTRIES; j++) {
        u32 file_fcb_base_addr = fs->SUPERBLOCK_SIZE + j * fs->FCB_SIZE;
        if (read_file_attr(fs, file_fcb_base_addr, 1) == FCB_INVALID) continue;
        int file_modtime = read_file_attr(fs, file_fcb_base_addr + MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH);
        if (file_modtime < prev_oldest_modtime) continue;
        if (file_modtime < curr_oldest_modtime) {
          curr_oldest_modtime = file_modtime;
          curr_file_name = read_file_attr(fs, file_fcb_base_addr + NAME_ATTR_OFFSET);
        }
      }
      printf("%-20s", curr_file_name);
      prev_oldest_modtime = curr_oldest_modtime;
    }
  }
  else if (op == LS_S) {
    int prev_max_size = fs->MAX_FILE_SIZE;
    for (int i = 0; i < gfilenum; i++) {
      int curr_max_size = -1, curr_oldest_create_time = gtime;
      char* curr_file_name;
      for (int j = 0; j < fs->FCB_ENTRIES; j++) {
        u32 file_fcb_base_addr = fs->SUPERBLOCK_SIZE + j * fs->FCB_SIZE;
        if (read_file_attr(fs, file_fcb_base_addr, 1) == FCB_INVALID) continue;
        int file_size = read_file_attr(fs, file_fcb_base_addr + SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
        int file_create_time = read_file_attr(fs, file_fcb_base_addr + CREATE_TIME_ATTR_OFFSET, CREATE_TIME_ATTR_LENGTH);
        if (file_size > prev_max_size) continue;
        if (file_size > curr_max_size || (file_size == curr_max_size && file_create_time < curr_oldest_create_time)) {
          curr_max_size = file_size;
          curr_oldest_create_time = file_create_time;
          curr_file_name = read_file_attr(fs, file_fcb_base_addr + NAME_ATTR_OFFSET);
        }
      }
      printf("%-20s\t %d", curr_file_name, curr_max_size);
      prev_max_size = curr_max_size;
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
  u32 fcb_base_addr = fs->SUPERBLOCK_SIZE + query.FCB_index * fs->FCB_SIZE;
  set_file_attr(fs, fcb_base_addr, FCB_INVALID);
  gfilenum--;
}
