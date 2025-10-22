REMOTE_USER := root
REMOTE_HOST := localhost
PORT := 9191
REMOTE := $(REMOTE_USER)@$(REMOTE_HOST)
REMOTE_DIR := /kaggle/working/cayleypy-cube
KAGGLE_KEY := ../kagglekey

.PHONY: run sync syncback ssh dataset train test

sync:
	rsync -cahvzP --stats --delete -e "ssh -p $(PORT) -F /dev/null -i $(KAGGLE_KEY)" ./ --exclude .git root@localhost:$(REMOTE_DIR)

syncback:
	rsync -cahvzP --stats -e "ssh -p $(PORT) -F /dev/null -i $(KAGGLE_KEY)" --exclude .git root@localhost:$(REMOTE_DIR)/ ./

run: sync
	ssh -i $(KAGGLE_KEY) -p $(PORT) -F /dev/null root@localhost "cd $(REMOTE_DIR) && python $(filter-out run,$(MAKECMDGOALS))"

ssh:
	ssh -i $(KAGGLE_KEY) -p $(PORT) -F /dev/null root@localhost
	
%:
	@:


