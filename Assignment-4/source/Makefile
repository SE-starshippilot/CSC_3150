PREV_SESSIONS:=$(shell pidof cuda-gdb)

run: build
	./main
build:
	@clear
	@for i in $(PREV_SESSIONS); do kill $$i; done
	@nvcc --relocatable-device-code=true main.cu user_program.cu file_system.cu -arch=sm_61 -g -G -o main -std=c++11
clear:
	rm ./main