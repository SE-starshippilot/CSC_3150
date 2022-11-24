PREV_SESSIONS:=$(shell pidof cuda-gdb)

build:
	@clear
	@for i in $(PREV_SESSIONS); do kill $$i; done
	@nvcc --relocatable-device-code=true main.cu user_program.cu file_system.cu -arch=sm_61 -g -G -o main
run: ./main
	./main
clear:
	rm ./main