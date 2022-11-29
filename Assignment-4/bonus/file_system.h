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
#define MKDIR 3
#define CD 4
#define CD_P 5
#define RM_RF 6
#define PWD 7

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
__device__ int str_cmp(char* str1, char* str2);
__device__ int str_len(const char* str);
__device__ char* get_file_attr(FileSystem* fs, u32 fp, int attr_offset);
__device__ int get_file_attr(FileSystem* fs, u32 fp, int attr_offset, int attr_length);
__device__ void set_file_attr(FileSystem* fs, u32 fp, int attr_offset, int attr_length, char* value);
__device__ void set_file_attr(FileSystem* fs, u32 fp, int attr_offset, int attr_length, int value);
__device__ void append_parent_content(FileSystem* fs, char* s);
__device__ void pop_parent_content(FileSystem* fs, char* s);
__device__ FCBQuery search_file(FileSystem* fs, char* s);
__device__ u32 get_file_base_addr(FileSystem* fs, u32 fp);
__device__ u32 get_file_end_block(FileSystem* fs, u32 fp);
__device__ void vcb_set(FileSystem* fs, int fp, int val);
__device__ int move_file(FileSystem* fs, u32 fp, int new_start_block_idx);
__device__ int fs_compress(FileSystem* fs);
__device__ u32 fs_allocate(FileSystem* fs, int block_num);
__device__ void fs_diagnose(FileSystem* fs);
__device__ void file_diagnose(FileSystem* fs, u32 fp);
#endif