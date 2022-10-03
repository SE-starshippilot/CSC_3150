CFILES:= $(shell ls|grep .c)
PROGS:=$(patsubst %.c,%,$(CFILES))
count:=0
all: $(PROGS)

%:%.c
	$(CC) -o $@ $<

clean:$(PROGS)
	rm $(PROGS)

run: $(PROGS)
	@for testcase in $(PROGS) ; do \
        if [ $$testcase != "program1" ]; then \
			echo "************Performing Testcase For $$testcase ************\n"; \
			./program1 $$testcase; \
			echo "\n"; \
		fi \
	done