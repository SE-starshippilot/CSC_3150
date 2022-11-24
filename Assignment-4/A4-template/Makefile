build:
	nvcc --relocatable-device-code=true main.cu user_program.cu file_system.cu -o test
run: build
	./test
clear:
	rm ./test