run:
	for f in *.tar.gz; do tar -xzf "$$f"; done
	mkdir test$(NUM)
	mv *.tar.gz test$(NUM)
	mv *.jpg test$(NUM)

run_big:
	for f in *.tar.gz; do tar -xzf "$$f"; done
	mkdir test_big$(NUM)
	mv *.tar.gz test_big$(NUM)
	mv *.jpg test_big$(NUM)
	

