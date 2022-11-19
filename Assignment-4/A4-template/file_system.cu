#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
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

__device__ __managed__ u32 gtime = 0;

__device__ void fcb_init(FileSystem *fs){
  for (u32 i = 0; i < fs->FCB_ENTRIES; i++){
    fs->volume[fs->SUPERBLOCK_SIZE + i*fs->FCB_SIZE] = 0x00; // MSB in the first byte of FCB is valid bit. 0 indicates invalid.
  }
}

__device__ void superblock_init(FileSystem *fs){
  // Initialize superblock. In my implementation, 0 means free and 1 means used.
  for (uchar i = 0; i < fs->SUPERBLOCK_SIZE; i++){
    fs->volume[i] = 0;
  }
}

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
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

__device__ int cmp_str(char *str1, char *str2){
  /* Compare two strings. Return 1 if they are the same. */
  while(*str1 != '\0' && *str2 != '\0'){
    if (*str1 != *str2) return 0;
    str1++;
    str2++;
  }
  if (*str1 == '\0' && *str2 == '\0') return 1;
  else return 0;
}

__device__ char* read_file_attr(FileSystem*fs, u32 fcb_offset_addr){
  int file_name_len = 0;
  while(fs->volume[fcb_offset_addr + file_name_len] != '\0'){
    file_name_len++;
  }
  file_name_len++;
  char file_name[file_name_len];
  memcpy(file_name, fs->volume + fcb_offset_addr , file_name_len);
  return file_name;
}

__device__ int read_file_attr(FileSystem*fs, u32 fcb_offset_addr, int length){
  /* Read file attribute from FCB. */
  int result = 0;
  for (int i = 0; i < length; i++){
    result = result << 8;
    result += fs->volume[fcb_offset_addr + i];
  }
  return result;
}

__device__ void set_file_attr(FileSystem *fs, u32 fcb_offset_addr, int length, int value){
  /* Set file attribute. */
  for (int i=length-1; i>=0; i--){
    fs->volume[fcb_offset_addr + i] = value & 0xFF;
    value = value >> 8;
  }
}

__device__ void set_file_attr(FileSystem *fs, u32 fcb_offset_addr, char *value){
  /* Set file attribute. */
  int count=0;
  while (value != '\0'){
    fs->volume[fcb_offset_addr + count] = value[count];
    count++;
    if (count == fs->MAX_FILENAME_SIZE) break;
  }
}

__device__ FCBQuery find_fcb(FileSystem *fs, char *s){
  /* Find the FCB of the file with name s. 
   * Return a query result containing the fcb index (if found) and first empty fcb index (if found). 
   */
  FCBQuery ret_val = {-1, -1};
  for (u32 i = 0; i < fs->FCB_ENTRIES; i++){
    if (fs->volume[fs->SUPERBLOCK_SIZE + i*fs->FCB_SIZE] & FCB_VALID == FCB_VALID){ // valid bit is set
      if (cmp_str(s, (char *)fs->volume + fs->SUPERBLOCK_SIZE + i*fs->FCB_SIZE + 1)){
        ret_val.FCB_index = i;
      }
    } else if (ret_val.FCB_index == -1){
      ret_val.empty_index = i;
    }
    if (ret_val.FCB_index != -1 && ret_val.empty_index != -1) break;
  }
  return ret_val;
}

__device__ u32 get_file_base_addr(FileSystem *fs, u32 fp){
  u32 file_start_block = read_file_attr(fs, fs->SUPERBLOCK_SIZE + fp*fs->FCB_SIZE + STARTBLK_ATTR_OFFSET, STARTBLK_ATTR_LENGTH);
  return fs->volume[fs->FILE_BASE_ADDRESS + file_start_block * fs->STORAGE_BLOCK_SIZE];
}

__device__ void fs_clear(FileSystem *fs, u32 fp){
  u32 file_base_addr = get_file_base_addr(fs, fp);
  int file_size = read_file_attr(fs, fs->SUPERBLOCK_SIZE + fp*fs->FCB_SIZE + SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);

}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
  FCBQuery query = find_fcb(fs, s);
  if (query.FCB_index != -1) return query.FCB_index;
  if (op == G_READ){
    return -1;
  } else if (op == G_WRITE){
    if (query.empty_index == -1) return -1;
    else {
      u32 FCB_base_addr = fs->SUPERBLOCK_SIZE + query.empty_index*fs->FCB_SIZE;
      set_file_attr(fs, FCB_base_addr+NAME_ATTR_OFFSET, s); // set file name
      set_file_attr(fs, FCB_base_addr+SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH, 0); // set file size
      set_file_attr(fs, FCB_base_addr+CREATE_TIME_ATTR_OFFSET, CREATE_TIME_ATTR_LENGTH, gtime); // set create time
      set_file_attr(fs, FCB_base_addr+MODIFY_TIME_ATTR_OFFSET, MODIFY_TIME_ATTR_LENGTH, gtime); // set modify time
      return query.empty_index;
    }
  } else {
    printf("Invalid operation code");
  }
  
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
  if (fp == -1){
    printf("File not found"); 
    return;
  }
  int file_size = read_file_attr(fs, fs->SUPERBLOCK_SIZE + fp*fs->FCB_SIZE + SIZE_ATTR_OFFSET, SIZE_ATTR_LENGTH);
  if (size > file_size){
    printf("Read size exceeds file size");
    return;
  }
  u32 file_start_addr = get_file_base_addr(fs, fp);
  for (int i=0; i<size; i++){
    output[i] = fs->volume[file_start_addr + i];
  }
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
  if (fp == -1){
    printf("Invalid fp");
    return;
  }

}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
}
