run: build
	./main
build:
	@nvcc --relocatable-device-code=true -g -G main.cu user_program.cu virtual_memory.cu -o main
clean:
	rm main

