#ifndef FILE_SYSTEM_H
#define FILE_SYSTEM_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

#define G_WRITE 1
#define G_READ 0
#define LS_D 0
#define LS_S 1
#define RM 2

#define LS_DR 3 //display rich info ranking using LS_D

struct FileSystem {
	uchar *volume;
	int SUPERBLOCK_SIZE;
	int FCB_SIZE;
	int FCB_ENTRIES;
	int STORAGE_SIZE;
	int STORAGE_BLOCK_SIZE;
	int MAX_FILENAME_SIZE;
	int MAX_FILE_NUM;
	int MAX_FILE_SIZE;
	int FILE_BASE_ADDRESS;
};

struct FCBQuery{
	int FCB_index;
	int empty_index;
};

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS);

__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem *fs, int op);
__device__ void fs_gsys(FileSystem *fs, int op, char *s);

/* Self-defined function prototypes */
__device__ void fcb_init(FileSystem* fs);
__device__ void superblock_init(FileSystem* fs);
__device__ int is_same_str(char* str1, char* str2);
template <typename T>__device__ T get_file_attr(FileSystem* fs, u32 fp, int attr_offset);
template <typename T> __device__ void set_file_attr(FileSystem* fs, u32 fp, int attr_offset, T value);
__device__ FCBQuery search_file(FileSystem* fs, char* s);
__device__ u32 get_file_base_addr(FileSystem* fs, u32 fp);
__device__ short get_file_end_block(FileSystem* fs, u32 fp);
__device__ u32 get_block_idx(FileSystem* fs, u32 addr);
__device__ void vcb_set(FileSystem* fs, int fp, int val);
__device__ int count_vacant_bits(int VCB_Byte);
__device__ int has_enough_space(FileSystem* fs, int block_size);
__device__ int move_file(FileSystem* fs, u32 fp, int new_start_block_idx);
__device__ int fs_compress(FileSystem* fs);
__device__ u32 fs_allocate(FileSystem* fs, int block_num);
__device__ void fs_diagnose(FileSystem* fs, u32 curr_fp);
#endif