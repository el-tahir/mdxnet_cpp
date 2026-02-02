all:
	@mkdir -p build
	@cd build && cmake .. && make

clean:
	@rm -rf build
	@rm -f separator

.PHONY: all clean