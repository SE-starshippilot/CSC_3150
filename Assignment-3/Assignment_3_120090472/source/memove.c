#include <stdio.h>
#include <string.h>

typedef struct VM VirtualMemory;
struct VM {
    int PAGE_ENTRIES;
    int INVERT_PAGE_TABLE_SIZE;
    int lru_oldest;
    unsigned int* invert_page_table;
};
unsigned int pt[];
void init_invert_page_table(VirtualMemory* vm) {
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = 0xffffffff;  // Take the second 1K space as the LRU list 
  }
}

void init(VirtualMemory* vm, int page_entries, int invert_table_size, unsigned int* invert_table) {
    vm->PAGE_ENTRIES = page_entries;
    vm->INVERT_PAGE_TABLE_SIZE = invert_table_size;
    vm->lru_oldest = page_entries;
    init_invert_page_table(vm);
}

int update_lru(VirtualMemory* vm, int page_id, int is_oldpage) {
    /* If lru recorded the page, find it and place it at the top of LRU array */
    int move_pos = -1, ret_val = -1;  // index of page_id in LRU array
    if (is_oldpage) {                  // save some time. no need to search LRU array if it is a new page
        for (int i = vm->PAGE_ENTRIES; i < vm->lru_oldest; i++) {
            if (vm->invert_page_table[i] == page_id) {
                move_pos = i;     // if page_id is in the array, then everything (starting at )
                break;
            }
        }//find
    }
    if (move_pos == -1) {
        if (vm->lru_oldest < vm->PAGE_ENTRIES * 2) vm->lru_oldest++;
        else {
            printf("Current vm->lru_oldest: %d; frame: %d\n", vm->lru_oldest, vm->invert_page_table[vm->lru_oldest]);
            ret_val = vm->invert_page_table[vm->lru_oldest];
        };
        move_pos = vm->lru_oldest;
    }
    // printf("Copy Size: %d*%d", (move_pos - vm->PAGE_ENTRIES + 1), sizeof(u32));
    printf("\nBefore:++++++++++++++++++++\n");
    printf("Page entries:%d\n", vm->PAGE_ENTRIES);
    for (int i = vm->PAGE_ENTRIES; i < vm->lru_oldest; i++) {
        printf("%d->", vm->invert_page_table[i]);
    }
    printf("\n+++++++++++++++++++++++++++\n");
    for (int i = vm->PAGE_ENTRIES; i < move_pos; i++) {
        vm->invert_page_table[i + 1] = vm->invert_page_table[i];
    }
    vm->invert_page_table[vm->PAGE_ENTRIES] = page_id;
    // memcpy(&vm->invert_page_table[vm->PAGE_ENTRIES + 1],
    //   &vm->invert_page_table[vm->PAGE_ENTRIES],
    //   (move_pos - vm->PAGE_ENTRIES + 1) * sizeof(u32)
    // );
    return ret_val;
}

int main() {
    VirtualMemory vm;
    init(&vm, 1<<10, 1<<14, pt);
    int ara[20] = { 1,2,3,4,5,6,7,8,9,10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
    //after copy  {1,2,3,4,5,6,7,8,9,10,4, 1, 2, 3, 5, 6, 7, 8, 9, 10};
    int find = 10, index_to_move = -1;
    for (int i = 10; i < 20; i++) {
        if (ara[i] == find) {
            index_to_move = i;
            break;
        }
    }
    int max = 9;
    printf("Before copy:");
    for (int i = 0; i < 20; i++) {
        printf("%d ", ara[i]);
    }
    printf("\n");
    memcpy(ara + 11, ara + 10, (max - 10 + 1) * sizeof(ara[0]));
    ara[10] = find;
    printf("After  copy:");
    for (int i = 0; i < 20; i++) {
        printf("%d ", ara[i]);
    }
    printf("\n");

}